// Sample metadata with quantitative benchmark scores from evaluation_results/metrics_summary.json
const SAMPLES = {
  "whu_test_000": {
    name: "whu_test_000 (Dense Suburban)",
    iou: "91.68%",
    f1: "95.66%",
    prec: "98.90%",
    rec: "92.63%",
    dir: "/evaluation_results/whu_test_000",
    summary: "/evaluation_results/evaluation_summary_whu_test_000.png"
  },
  "whu_test_001": {
    name: "whu_test_001 (Residential Blocks)",
    iou: "86.98%",
    f1: "93.04%",
    prec: "98.16%",
    rec: "88.43%",
    dir: "/evaluation_results/whu_test_001",
    summary: "/evaluation_results/evaluation_summary_whu_test_001.png"
  },
  "whu_test_002": {
    name: "whu_test_002 (Urban Commercial)",
    iou: "89.31%",
    f1: "94.35%",
    prec: "99.21%",
    rec: "89.95%",
    dir: "/evaluation_results/whu_test_002",
    summary: "/evaluation_results/evaluation_summary_whu_test_002.png"
  },
  "whu_test_003": {
    name: "whu_test_003 (High-Density Rooftops)",
    iou: "43.25%",
    f1: "60.38%",
    prec: "46.20%",
    rec: "87.14%",
    dir: "/evaluation_results/whu_test_003",
    summary: "/evaluation_results/evaluation_summary_whu_test_003.png"
  },
  "whu_test_004": {
    name: "whu_test_004 (Mixed Industrial)",
    iou: "89.83%",
    f1: "94.64%",
    prec: "99.03%",
    rec: "90.63%",
    dir: "/evaluation_results/whu_test_004",
    summary: "/evaluation_results/evaluation_summary_whu_test_004.png"
  },
  "whu_test_005": {
    name: "whu_test_005 (Complex Geometries)",
    iou: "85.60%",
    f1: "92.24%",
    prec: "97.80%",
    rec: "87.27%",
    dir: "/evaluation_results/whu_test_005",
    summary: "/evaluation_results/evaluation_summary_whu_test_005.png"
  }
};

let currentSampleKey = "whu_test_000";
let currentLayer = "mask"; // 'mask', 'gt', 'uncertainty', 'error'
let sliderPos = 50; // percentage
let isDragging = false;

// DOM elements
const compareBox = document.getElementById("compareBox");
const clipContainer = document.getElementById("clipContainer");
const sliderHandle = document.getElementById("sliderHandle");
const imgLeft = document.getElementById("imgLeft");
const imgRight = document.getElementById("imgRight");
const imgUncBlend = document.getElementById("imgUncBlend");
const imgMultilevel = document.getElementById("imgMultilevel");
const currentSampleName = document.getElementById("currentSampleName");
const rightOverlayLabel = document.getElementById("rightOverlayLabel");
const rightLegendText = document.getElementById("rightLegendText");
const rightLegendBadge = document.getElementById("rightLegendBadge");

// Metric elements
const metricIoU = document.getElementById("metricIoU");
const metricF1 = document.getElementById("metricF1");
const metricPrecision = document.getElementById("metricPrecision");
const metricRecall = document.getElementById("metricRecall");

// Sliders
const uncOpacitySlider = document.getElementById("uncOpacitySlider");
const uncOpacityVal = document.getElementById("uncOpacityVal");
const tintOpacitySlider = document.getElementById("tintOpacitySlider");
const tintOpacityVal = document.getElementById("tintOpacityVal");
const tintCanvas = document.getElementById("tintCanvas");

// Initialize Lucide icons
if (window.lucide) {
  lucide.createIcons();
}

// -------------------------------------------------------------
// Interactive Swipe Slider Implementation
// -------------------------------------------------------------
function updateSliderPosition(percent) {
  sliderPos = Math.max(0, Math.min(100, percent));
  sliderHandle.style.left = `${sliderPos}%`;
  clipContainer.style.clipPath = `polygon(${sliderPos}% 0, 100% 0, 100% 100%, ${sliderPos}% 100%)`;
}

function handleDrag(e) {
  if (!isDragging) return;
  const rect = compareBox.getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const offsetX = clientX - rect.left;
  const percent = (offsetX / rect.width) * 100;
  updateSliderPosition(percent);
}

sliderHandle.addEventListener("mousedown", (e) => {
  isDragging = true;
  e.preventDefault();
});

compareBox.addEventListener("mousedown", (e) => {
  isDragging = true;
  handleDrag(e);
});

window.addEventListener("mouseup", () => {
  isDragging = false;
});

window.addEventListener("mousemove", handleDrag);

// Touch Support
compareBox.addEventListener("touchstart", (e) => {
  isDragging = true;
  handleDrag(e);
}, { passive: true });

window.addEventListener("touchend", () => {
  isDragging = false;
});

window.addEventListener("touchmove", handleDrag, { passive: true });

// -------------------------------------------------------------
// Sample Switching
// -------------------------------------------------------------
function selectSample(sampleKey) {
  currentSampleKey = sampleKey;
  const data = SAMPLES[sampleKey];
  if (!data) return;

  // Update button states
  document.querySelectorAll(".preset-btn").forEach(btn => {
    if (btn.dataset.sample === sampleKey) {
      btn.className = "preset-btn px-2.5 py-2 rounded-lg bg-blue-600 text-white font-semibold text-xs transition flex flex-col items-center justify-center border border-blue-400/40";
    } else {
      btn.className = "preset-btn px-2.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium text-xs transition flex flex-col items-center justify-center border border-slate-700";
    }
  });

  // Update images
  imgLeft.src = `${data.dir}/01_input_rgb.png`;
  imgUncBlend.src = `${data.dir}/04_uncertainty_map.png`;
  imgMultilevel.src = `${data.dir}/06_multilevel_refinement.png`;

  // Update Right Layer
  updateRightLayer();

  // Update Metrics
  currentSampleName.innerText = sampleKey;
  metricIoU.innerText = data.iou;
  metricF1.innerText = data.f1;
  metricPrecision.innerText = data.prec;
  metricRecall.innerText = data.rec;

  // Re-draw tint canvas once mask loads
  updateTintCanvas();
}

function updateRightLayer() {
  const data = SAMPLES[currentSampleKey];
  if (!data) return;

  if (currentLayer === "mask") {
    imgRight.src = `${data.dir}/03_pred_mask_m1.png`;
    rightOverlayLabel.innerText = "UANet Mask (M1)";
    rightLegendText.innerText = "Right: Predicted Footprints";
    rightLegendBadge.className = "w-2.5 h-2.5 rounded-sm bg-white";
  } else if (currentLayer === "gt") {
    imgRight.src = `${data.dir}/02_ground_truth.png`;
    rightOverlayLabel.innerText = "Ground Truth";
    rightLegendText.innerText = "Right: Ground Truth";
    rightLegendBadge.className = "w-2.5 h-2.5 rounded-sm bg-slate-200";
  } else if (currentLayer === "uncertainty") {
    imgRight.src = `${data.dir}/04_uncertainty_map.png`;
    rightOverlayLabel.innerText = "Uncertainty Heatmap";
    rightLegendText.innerText = "Right: Boundary Uncertainty";
    rightLegendBadge.className = "w-2.5 h-2.5 rounded-sm bg-gradient-to-r from-blue-500 to-red-500";
  } else if (currentLayer === "error") {
    imgRight.src = `${data.dir}/05_error_overlay.png`;
    rightOverlayLabel.innerText = "Error Overlay (G:TP, R:FP, B:FN)";
    rightLegendText.innerText = "Right: Error (G:TP, R:FP, B:FN)";
    rightLegendBadge.className = "w-2.5 h-2.5 rounded-sm bg-emerald-500";
  }
}

// Preset button click listeners
document.querySelectorAll(".preset-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    selectSample(btn.dataset.sample);
  });
});

// Layer buttons click listeners
document.querySelectorAll(".layer-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".layer-btn").forEach(b => {
      b.className = "layer-btn px-2.5 py-1.5 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium border border-slate-700";
    });
    btn.className = "layer-btn active px-2.5 py-1.5 text-xs rounded-lg bg-blue-600 text-white font-medium border border-blue-400/30";
    currentLayer = btn.dataset.layer;
    updateRightLayer();
  });
});

// -------------------------------------------------------------
// Tint & Uncertainty Opacity Controls
// -------------------------------------------------------------
uncOpacitySlider.addEventListener("input", (e) => {
  const val = e.target.value;
  uncOpacityVal.innerText = `${val}%`;
  imgUncBlend.style.opacity = val / 100;
});

tintOpacitySlider.addEventListener("input", (e) => {
  const val = e.target.value;
  tintOpacityVal.innerText = `${val}%`;
  tintCanvas.style.opacity = val / 100;
});

function updateTintCanvas() {
  const data = SAMPLES[currentSampleKey];
  if (!data) return;
  const maskImg = new Image();
  maskImg.crossOrigin = "anonymous";
  maskImg.src = `${data.dir}/03_pred_mask_m1.png`;
  maskImg.onload = () => {
    tintCanvas.width = 512;
    tintCanvas.height = 512;
    const ctx = tintCanvas.getContext("2d");
    ctx.clearRect(0, 0, 512, 512);

    // Draw mask to temp buffer
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 512;
    tempCanvas.height = 512;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.drawImage(maskImg, 0, 0, 512, 512);
    const imgData = tempCtx.getImageData(0, 0, 512, 512);
    const pixels = imgData.data;

    // Tint building pixels (white pixels) to glowing cyan [6, 182, 212]
    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] > 128) {
        pixels[i] = 6;      // R
        pixels[i + 1] = 182; // G
        pixels[i + 2] = 212; // B
        pixels[i + 3] = 220; // Alpha
      } else {
        pixels[i + 3] = 0;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  };
}

// -------------------------------------------------------------
// Download Handlers
// -------------------------------------------------------------
document.getElementById("btnDownloadMask").addEventListener("click", () => {
  const data = SAMPLES[currentSampleKey];
  const link = document.createElement("a");
  link.href = `${data.dir}/03_pred_mask_m1.png`;
  link.download = `${currentSampleKey}_UANet_prediction_mask.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

document.getElementById("btnDownloadSummary").addEventListener("click", () => {
  const data = SAMPLES[currentSampleKey];
  const link = document.createElement("a");
  link.href = data.summary;
  link.download = `${currentSampleKey}_5Stage_Summary.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
});

// -------------------------------------------------------------
// Custom Image Upload & Live Inference Handling
// -------------------------------------------------------------
const imageUploader = document.getElementById("imageUploader");
imageUploader.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (event) => {
    const dataUrl = event.target.result;
    imgLeft.src = dataUrl;
    currentSampleName.innerText = file.name;
    metricIoU.innerText = "Custom";
    metricF1.innerText = "Evaluating...";
    metricPrecision.innerText = "---";
    metricRecall.innerText = "---";

    // Attempt live API inference
    try {
      const formData = new FormData();
      formData.append("file", file);

      const resp = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (resp.ok) {
        const result = await resp.json();
        imgRight.src = result.prediction_mask_url || dataUrl;
        imgUncBlend.src = result.uncertainty_map_url || "";
        metricIoU.innerText = result.iou ? `${(result.iou * 100).toFixed(1)}%` : "N/A";
        metricF1.innerText = result.building_area_percent ? `${result.building_area_percent}% Area` : "Inferred";
        metricPrecision.innerText = `${result.num_buildings || 0} structures`;
        metricRecall.innerText = "Live PyTorch";
      } else {
        console.warn("Backend inference API not active. Showing uploaded preview.");
        metricF1.innerText = "Uploaded Tile";
      }
    } catch (err) {
      console.warn("Local API not running. Showing uploaded preview.");
      metricF1.innerText = "Uploaded Tile";
    }
  };
  reader.readAsDataURL(file);
});

// -------------------------------------------------------------
// Documentation Modal
// -------------------------------------------------------------
const docsModal = document.getElementById("docsModal");
const btnOpenDocs = document.getElementById("btnOpenDocs");
const btnCloseDocs = document.getElementById("btnCloseDocs");

btnOpenDocs.addEventListener("click", () => docsModal.classList.remove("hidden"));
btnCloseDocs.addEventListener("click", () => docsModal.classList.add("hidden"));
docsModal.addEventListener("click", (e) => {
  if (e.target === docsModal) docsModal.classList.add("hidden");
});

// Initialize on page load
selectSample("whu_test_000");
updateSliderPosition(50);

