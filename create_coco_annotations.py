import json
from pathlib import Path
import yaml
from PIL import Image
from tqdm import tqdm

def create_coco_annotations(data_yaml_path, output_json_path):
    """Convert YOLO format to COCO format annotations"""
    
    print(f"📂 Loading dataset config: {data_yaml_path}")
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)
    
    # Get paths
    val_rgb_path = Path(data['val_rgb'])
    labels_path = val_rgb_path.parent.parent / 'labels' / val_rgb_path.name
    
    print(f"📂 Images path: {val_rgb_path}")
    print(f"📂 Labels path: {labels_path}")
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(data['names'])]
    }
    
    ann_id = 1
    img_files = sorted(val_rgb_path.glob('*.jpg')) + sorted(val_rgb_path.glob('*.png'))
    
    print(f"📊 Processing {len(img_files)} images...")
    
    for img_id, img_file in enumerate(tqdm(img_files)):
        try:
            img = Image.open(img_file)
            width, height = img.size
        except:
            print(f"⚠️  Skipping {img_file.name} - cannot open")
            continue
        
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_file.name,
            "width": width,
            "height": height
        })
        
        # Read corresponding label file
        label_file = labels_path / (img_file.stem + '.txt')
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])
                        
                        # Convert YOLO (normalized) to COCO (pixel, top-left)
                        x_min = (x_center - w/2) * width
                        y_min = (y_center - h/2) * height
                        box_w = w * width
                        box_h = h * height
                        
                        coco_format["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [x_min, y_min, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0
                        })
                        ann_id += 1
    
    # Save
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"\n✅ Created COCO annotations: {output_path}")
    print(f"   📸 Images: {len(coco_format['images'])}")
    print(f"   📦 Annotations: {len(coco_format['annotations'])}")
    print(f"   🏷️  Categories: {len(coco_format['categories'])}")
    
    # Size distribution
    areas = [ann['area'] for ann in coco_format['annotations']]
    if areas:
        import numpy as np
        areas = np.array(areas)
        small = (areas < 32**2).sum()
        medium = ((areas >= 32**2) & (areas < 96**2)).sum()
        large = (areas >= 96**2).sum()
        print(f"\n   📏 Size distribution:")
        print(f"      Small:  {small} ({small/len(areas)*100:.1f}%)")
        print(f"      Medium: {medium} ({medium/len(areas)*100:.1f}%)")
        print(f"      Large:  {large} ({large/len(areas)*100:.1f}%)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR_aligned.yaml')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        # Auto-determine output path
        data_path = Path(args.data).parent
        args.output = str(data_path / 'annotations.json')
    
    create_coco_annotations(args.data, args.output)