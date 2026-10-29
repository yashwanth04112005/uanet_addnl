import os
import cv2
import numpy as np
from PIL import Image

def generate_remote_sensing_tile(seed, img_size=(512, 512)):
    np.random.seed(seed)
    h, w = img_size
    
    # Base terrain color (grass, soil, concrete, asphalt)
    terrain_types = [
        (np.array([45, 80, 55]), np.array([75, 120, 85])),     # Vegetation
        (np.array([120, 115, 100]), np.array([150, 145, 130])), # Soil / dirt
        (np.array([100, 105, 110]), np.array([135, 140, 145])), # Urban concrete
    ]
    t_idx = np.random.randint(0, len(terrain_types))
    low_col, high_col = terrain_types[t_idx]
    
    # Background texture with perlin-like gradient noise
    img = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        base = np.random.uniform(low_col[c], high_col[c])
        noise = cv2.resize(np.random.randn(16, 16).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6) * 30.0 - 15.0
        img[:, :, c] = np.clip(base + noise, 0, 255)
    
    # Add road network
    num_roads = np.random.randint(1, 3)
    road_mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(num_roads):
        pt1 = (np.random.randint(0, w), 0 if np.random.rand() > 0.5 else np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), h if np.random.rand() > 0.5 else np.random.randint(0, h))
        road_width = np.random.randint(12, 24)
        road_color = np.random.uniform(60, 90, size=3)
        cv2.line(img, pt1, pt2, road_color, road_width)
        cv2.line(road_mask, pt1, pt2, 1, road_width)

    # Initialize mask: 1 is Background, 0 is Building (as defined in WHU dataset: classes Building=0, Background=1)
    mask = np.ones((h, w), dtype=np.uint8)
    
    # Generate buildings
    num_buildings = np.random.randint(8, 20)
    for _ in range(num_buildings):
        bw = np.random.randint(30, 90)
        bh = np.random.randint(30, 90)
        bx = np.random.randint(10, w - bw - 10)
        by = np.random.randint(10, h - bh - 10)
        
        # Avoid major overlap with roads
        if np.mean(road_mask[by:by+bh, bx:bx+bw]) > 0.4:
            continue
            
        angle = np.random.uniform(0, 45)
        rect = ((bx + bw / 2, by + bh / 2), (bw, bh), angle)
        box = cv2.boxPoints(rect).astype(np.int32)
        
        # Shadow simulation (dark area adjacent to building)
        shadow_box = box + np.array([np.random.randint(3, 8), np.random.randint(3, 8)])
        cv2.fillPoly(img, [shadow_box], (25, 25, 25))
        
        # Building roof colors: red tile, blue tile, light concrete, galvanized steel
        roof_styles = [
            (180, 70, 60),   # Red / Terracotta
            (70, 110, 160),  # Blue / Navy steel
            (210, 205, 195), # Light flat concrete
            (160, 160, 165), # Galvanized grey
            (230, 180, 120), # Sand/Beige
        ]
        rc = roof_styles[np.random.randint(0, len(roof_styles))]
        
        # Building body
        cv2.fillPoly(img, [box], rc)
        
        # Rooftop edge / ridge detail
        cv2.polylines(img, [box], isClosed=True, color=(int(rc[0]*0.7), int(rc[1]*0.7), int(rc[2]*0.7)), thickness=2)
        
        # Set Building class (0 = Building)
        cv2.fillPoly(mask, [box], 0)
        
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, mask

def create_dataset_splits(base_dir="data/whubuilding"):
    splits = {
        'train': 16,
        'val': 4,
        'test': 6
    }
    
    seed = 100
    for split, count in splits.items():
        img_dir = os.path.join(base_dir, split, "images")
        mask_dir = os.path.join(base_dir, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        print(f"Generating {count} samples for {split}...")
        for i in range(count):
            img, mask = generate_remote_sensing_tile(seed=seed)
            seed += 1
            
            img_id = f"whu_{split}_{i:03d}"
            img_path = os.path.join(img_dir, f"{img_id}.tif")
            mask_path = os.path.join(mask_dir, f"{img_id}.png")
            
            # Save TIFF image and PNG mask
            Image.fromarray(img).save(img_path)
            Image.fromarray(mask).save(mask_path)
            
    print("Dataset generated successfully!")

if __name__ == "__main__":
    create_dataset_splits()
