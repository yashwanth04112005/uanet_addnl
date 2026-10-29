import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Backwards compatibility patch for Python 3.10+
import collections
import collections.abc
for _attr in ['MutableMapping', 'Mapping', 'Sequence', 'Iterable', 'Callable', 'Container']:
    if not hasattr(collections, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))

from tools.cfg import py2cfg
from train import Supervision_Train
from tools.metric import Evaluator

def compute_uncertainty_map(prob_map):
    """
    Compute uncertainty via: 0.5 - |0.5 - Sigmoid(M)|
    Normalized to [0, 1] for visual contrast:
    Uncertainty = 1.0 - 2.0 * |0.5 - prob|
    """
    raw_uncertainty = 0.5 - np.abs(0.5 - prob_map)
    # Normalize to [0, 1] range: 0 is completely certain, 1 is maximal uncertainty at boundary (p=0.5)
    norm_uncertainty = raw_uncertainty / 0.5
    return np.clip(norm_uncertainty, 0.0, 1.0)

def colorize_uncertainty(uncertainty_norm, colormap=cv2.COLORMAP_TURBO):
    uint8_unc = (uncertainty_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(uint8_unc, colormap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

def evaluate_and_visualize(config_path="config/whubuilding/UANet.py",
                           output_dir="evaluation_results",
                           checkpoint_path=None):
    os.makedirs(output_dir, exist_ok=True)
    config = py2cfg(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running evaluation on device: {device}")

    # Determine checkpoint
    if checkpoint_path is None:
        ckpt_candidate = os.path.join(config.weights_path, config.test_weights_name + ".ckpt")
        if os.path.exists(ckpt_candidate):
            checkpoint_path = ckpt_candidate
        else:
            files = [os.path.join(config.weights_path, f) for f in os.listdir(config.weights_path) if f.endswith(".ckpt")]
            if files:
                # Pick the latest modified checkpoint
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                checkpoint_path = files[0]

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[+] Loading trained checkpoint: {checkpoint_path}")
        model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config)
    else:
        print("[!] No checkpoint found on disk. Initializing model with ImageNet VGG backbone weights for evaluation...")
        model = Supervision_Train(config)

    model.to(device)
    model.eval()

    test_dataset = config.test_dataset
    print(f"[+] Evaluating on {len(test_dataset)} test samples...")

    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    detailed_metrics = []
    
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            img_tensor = sample['img'].unsqueeze(0).to(device)
            img_id = sample['img_id']
            gt_mask = sample['gt_semantic_seg'].cpu().numpy()

            # Multi-level predictions from UANet
            # UANet returns (predict_5, predict_4, predict_3, predict_2, predict_1)
            raw_outputs = model.net(img_tensor)
            
            # Predict_1 is final refined output M_1
            # WHU Building convention: Class 0 is Building, Class 1 is Background
            m5_logits, m4_logits, m3_logits, m2_logits, m1_logits = raw_outputs

            # Softmax probabilities
            probs_m1 = F.softmax(m1_logits, dim=1).squeeze(0).cpu().numpy()
            probs_m2 = F.softmax(m2_logits, dim=1).squeeze(0).cpu().numpy()
            probs_m3 = F.softmax(m3_logits, dim=1).squeeze(0).cpu().numpy()
            probs_m4 = F.softmax(m4_logits, dim=1).squeeze(0).cpu().numpy()
            probs_m5 = F.softmax(m5_logits, dim=1).squeeze(0).cpu().numpy()

            # Pred mask: 0=Building, 1=Background
            pred_mask = np.argmax(probs_m1, axis=0)

            # Update global confusion matrix
            evaluator.add_batch(pre_image=pred_mask, gt_image=gt_mask)

            # Building probability (Class 0)
            building_prob = probs_m1[0]
            
            # Uncertainty map: 0.5 - |0.5 - Sigmoid(M)|
            uncertainty_norm = compute_uncertainty_map(building_prob)
            uncertainty_rgb = colorize_uncertainty(uncertainty_norm)

            # Original RGB image un-normalized for visual display
            raw_img_path = os.path.join(test_dataset.data_root, test_dataset.img_dir, f"{img_id}{test_dataset.img_suffix}")
            if os.path.exists(raw_img_path):
                orig_rgb = np.array(Image.open(raw_img_path).convert('RGB'))
            else:
                orig_rgb = np.clip(sample['img'].permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)

            # Ground Truth RGB representation (White = Building, Black = Background)
            gt_rgb = np.zeros_like(orig_rgb)
            gt_rgb[gt_mask == 0] = [255, 255, 255]

            # Predicted Mask RGB representation
            pred_rgb = np.zeros_like(orig_rgb)
            pred_rgb[pred_mask == 0] = [255, 255, 255]

            # Error overlay: Green = TP, Red = FP, Blue = FN
            error_overlay = orig_rgb.copy()
            tp = (pred_mask == 0) & (gt_mask == 0)
            fp = (pred_mask == 0) & (gt_mask == 1)
            fn = (pred_mask == 1) & (gt_mask == 0)
            error_overlay[tp] = (error_overlay[tp] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
            error_overlay[fp] = (error_overlay[fp] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
            error_overlay[fn] = (error_overlay[fn] * 0.4 + np.array([0, 100, 255]) * 0.6).astype(np.uint8)

            # Multi-level progressive refinement maps (M_5 down to M_1)
            p5_mask = (np.argmax(probs_m5, axis=0) == 0).astype(np.uint8) * 255
            p4_mask = (np.argmax(probs_m4, axis=0) == 0).astype(np.uint8) * 255
            p3_mask = (np.argmax(probs_m3, axis=0) == 0).astype(np.uint8) * 255
            p2_mask = (np.argmax(probs_m2, axis=0) == 0).astype(np.uint8) * 255
            p1_mask = (np.argmax(probs_m1, axis=0) == 0).astype(np.uint8) * 255

            # Save individual files
            sample_sub_dir = os.path.join(output_dir, img_id)
            os.makedirs(sample_sub_dir, exist_ok=True)

            Image.fromarray(orig_rgb).save(os.path.join(sample_sub_dir, "01_input_rgb.png"))
            Image.fromarray(gt_rgb).save(os.path.join(sample_sub_dir, "02_ground_truth.png"))
            Image.fromarray(pred_rgb).save(os.path.join(sample_sub_dir, "03_pred_mask_m1.png"))
            Image.fromarray(uncertainty_rgb).save(os.path.join(sample_sub_dir, "04_uncertainty_map.png"))
            Image.fromarray(error_overlay).save(os.path.join(sample_sub_dir, "05_error_overlay.png"))

            # Save progressive multi-level refinement figure
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            stages = [
                ("M5 (Initial FPN)", p5_mask),
                ("M4 (UAFM Level 4)", p4_mask),
                ("M3 (UAFM Level 3)", p3_mask),
                ("M2 (UAFM Level 2)", p2_mask),
                ("M1 (Final Refined)", p1_mask)
            ]
            for ax, (title, mask_img) in zip(axes, stages):
                ax.imshow(mask_img, cmap='gray')
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(sample_sub_dir, "06_multilevel_refinement.png"), dpi=200, bbox_inches='tight')
            plt.close()

            # Save comprehensive side-by-side presentation figure
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            axes[0].imshow(orig_rgb); axes[0].set_title("(a) Input RS Image (RGB)", fontsize=13, fontweight='bold'); axes[0].axis('off')
            axes[1].imshow(gt_rgb); axes[1].set_title("(b) Ground Truth (GT)", fontsize=13, fontweight='bold'); axes[1].axis('off')
            axes[2].imshow(pred_rgb); axes[2].set_title("(c) UANet Prediction (M1)", fontsize=13, fontweight='bold'); axes[2].axis('off')
            axes[3].imshow(uncertainty_rgb); axes[3].set_title("(d) Uncertainty Map (Turbo)", fontsize=13, fontweight='bold'); axes[3].axis('off')
            axes[4].imshow(error_overlay); axes[4].set_title("(e) Error Overlay (G:TP, R:FP, B:FN)", fontsize=13, fontweight='bold'); axes[4].axis('off')
            plt.tight_layout()
            summary_fig_path = os.path.join(output_dir, f"evaluation_summary_{img_id}.png")
            plt.savefig(summary_fig_path, dpi=200, bbox_inches='tight')
            plt.close()

            # Sample metrics
            s_tp = np.sum(tp)
            s_fp = np.sum(fp)
            s_fn = np.sum(fn)
            s_iou = s_tp / (s_tp + s_fp + s_fn + 1e-8)
            s_prec = s_tp / (s_tp + s_fp + 1e-8)
            s_rec = s_tp / (s_tp + s_fn + 1e-8)
            s_f1 = (2.0 * s_prec * s_rec) / (s_prec + s_rec + 1e-8)
            detailed_metrics.append({
                "sample_id": img_id,
                "IoU": float(s_iou),
                "Precision": float(s_prec),
                "Recall": float(s_rec),
                "F1": float(s_f1)
            })

    # Overall dataset metrics (Class 0: Building)
    iou_per_class = evaluator.Intersection_over_Union()
    precision_per_class = evaluator.Precision()
    recall_per_class = evaluator.Recall()
    f1_per_class = evaluator.F1()
    oa = evaluator.OA()

    building_iou = float(iou_per_class[0])
    building_precision = float(precision_per_class[0])
    building_recall = float(recall_per_class[0])
    building_f1 = float(f1_per_class[0])
    overall_accuracy = float(oa)

    benchmark_results = {
        "model": "UANet (VGG-16 backbone + MBDC + PIGM + URA + UAFM)",
        "dataset": "WHU Building Dataset",
        "num_test_samples": len(test_dataset),
        "metrics": {
            "Building_IoU": building_iou,
            "Building_F1": building_f1,
            "Building_Precision": building_precision,
            "Building_Recall": building_recall,
            "Overall_Accuracy": overall_accuracy
        },
        "per_sample_metrics": detailed_metrics
    }

    # Save metrics JSON
    with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
        json.dump(benchmark_results, f, indent=4)

    # Print clean benchmark table
    table_str = f"""
========================================================================================
                  UANET QUANTITATIVE EVALUATION BENCHMARK REPORT
========================================================================================
 Model Architecture : UANet (VGG-16 BN Backbone + Multi-scale MBDC + PIGM + UAFM)
 Target Dataset     : WHU Building Footprint Segmentation
 Evaluated Samples  : {len(test_dataset)} test image tiles (512x512)
 Multi-Level Stages : M5 (FPN) -> M4 -> M3 -> M2 -> M1 (Final Refinement)
 Device Used        : {device}
----------------------------------------------------------------------------------------
 METRIC                    | VALUE (%)      | RAW SCORE
----------------------------------------------------------------------------------------
 IoU (Intersection over U) | {building_iou * 100:>6.2f}%        | {building_iou:.4f}
 F1-Score (Dice)           | {building_f1 * 100:>6.2f}%        | {building_f1:.4f}
 Precision                 | {building_precision * 100:>6.2f}%        | {building_precision:.4f}
 Recall                    | {building_recall * 100:>6.2f}%        | {building_recall:.4f}
 Overall Accuracy (OA)     | {overall_accuracy * 100:>6.2f}%        | {overall_accuracy:.4f}
========================================================================================
 Visualizations Saved to   : {os.path.abspath(output_dir)}
========================================================================================
"""
    print(table_str)
    with open(os.path.join(output_dir, "metrics_report.txt"), "w") as f:
        f.write(table_str)

    return benchmark_results

if __name__ == "__main__":
    evaluate_and_visualize()
