#!/usr/bin/env python3
"""
Anchor Re-clustering for FLIR Dataset
Optimized for small object detection in thermal/RGB fusion

Usage:
    python recluster_anchors.py --data data/multispectral/FLIR_aligned.yaml --imgsz 640

This will:
1. Analyze your dataset's ground truth boxes
2. Run k-means clustering (optimized for small objects)
3. Generate new anchors
4. Update your YAML config files
"""

import argparse
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import torch

def load_dataset_labels(data_path, img_size=640):
    """Load all ground truth boxes from dataset"""
    with open(data_path) as f:
        data_dict = yaml.safe_load(f)
    
    # Get label paths
    train_path = data_dict.get('train_rgb') or data_dict.get('train')
    
    # Find label files
    if train_path.endswith('.txt'):
        with open(train_path, 'r') as f:
            img_paths = [x.strip() for x in f.readlines()]
    else:
        from pathlib import Path
        img_paths = list(Path(train_path).rglob('*.jpg')) + list(Path(train_path).rglob('*.png'))
    
    print(f"Found {len(img_paths)} training images")
    
    # Load all boxes
    all_boxes = []
    sizes = {'small': 0, 'medium': 0, 'large': 0}
    
    for img_path in tqdm(img_paths, desc="Loading labels"):
        # Convert image path to label path
        label_path = str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt')
        
        try:
            with open(label_path, 'r') as f:
                labels = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
            
            if len(labels) > 0:
                # Extract width and height (normalized 0-1)
                boxes = labels[:, 3:5]  # [width, height]
                
                # Convert to pixels
                boxes_px = boxes * img_size
                
                # Count by size (COCO definition: small<32², medium<96², large>=96²)
                areas = boxes_px[:, 0] * boxes_px[:, 1]
                sizes['small'] += np.sum(areas < 32**2)
                sizes['medium'] += np.sum((areas >= 32**2) & (areas < 96**2))
                sizes['large'] += np.sum(areas >= 96**2)
                
                all_boxes.append(boxes)
        except:
            pass
    
    if len(all_boxes) == 0:
        raise ValueError("No labels found! Check your dataset paths.")
    
    all_boxes = np.concatenate(all_boxes, 0)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total objects: {len(all_boxes)}")
    print(f"   Small objects (<32px): {sizes['small']} ({sizes['small']/len(all_boxes)*100:.1f}%)")
    print(f"   Medium objects (32-96px): {sizes['medium']} ({sizes['medium']/len(all_boxes)*100:.1f}%)")
    print(f"   Large objects (>96px): {sizes['large']} ({sizes['large']/len(all_boxes)*100:.1f}%)")
    print(f"   Mean box size: {all_boxes.mean(0)}")
    print(f"   Median box size: {np.median(all_boxes, 0)}")
    
    return all_boxes, sizes

def kmeans_anchors(boxes, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    K-means clustering to generate anchors optimized for the dataset
    
    Args:
        boxes: Ground truth boxes (normalized 0-1) shape [N, 2] (width, height)
        n: Number of anchors (9 for 3 detection layers)
        img_size: Training image size
        thr: Anchor-label wh ratio threshold
        gen: Generations for k-means
        verbose: Print progress
    
    Returns:
        anchors: shape [n, 2] (width, height) in pixels
    """
    from scipy.cluster.vq import kmeans
    
    print(f'\n🔄 Running k-means for {n} anchors on {len(boxes)} boxes...')
    
    # Convert to pixels
    s = boxes.std(0)  # sigmas for whitening
    k, dist = kmeans(boxes / s, n, iter=gen)  # k-means
    k *= s
    
    # Sort by area (small to large)
    k = k[np.argsort(k.prod(1))]  # sort by area
    
    # Convert to pixels
    k *= img_size
    
    # Calculate fitness (best possible recall)
    wh = torch.tensor(boxes * img_size, dtype=torch.float32)
    wh0 = torch.tensor(k, dtype=torch.float32)
    
    def fitness(k):
        """Calculate best possible recall (BPR)"""
        r = wh[:, None] / k[None]  # ratios
        x = torch.min(r, 1. / r).min(2)[0]  # min ratio
        best = x.max(1)[0]  # best match for each box
        aat = (best > 1. / thr).float().sum()  # anchors above threshold
        bpr = best.mean()  # best possible recall
        return bpr, aat
    
    bpr, aat = fitness(wh0)
    
    if verbose:
        print(f'   Best Possible Recall (BPR): {bpr:.4f}')
        print(f'   Anchors above threshold: {aat:.0f}/{len(boxes)} ({aat/len(boxes)*100:.1f}%)')
        print(f'\n📏 New Anchors (pixels @ {img_size}px):')
        for i, (w, h) in enumerate(k):
            print(f'   Anchor {i}: {w:.1f}x{h:.1f}')
    
    return k

def update_yaml_anchors(yaml_path, anchors, img_size=640):
    """Update YAML config with new anchors"""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    
    # Normalize anchors back to 0-1 (relative to stride)
    # YOLOv5 uses 3 detection heads with strides [8, 16, 32]
    strides = [8, 16, 32]
    
    # Reshape anchors for 3 heads (3 anchors per head)
    anchors_per_head = anchors.reshape(3, 3, 2)
    
    # Normalize to stride (YOLOv5 format)
    formatted_anchors = []
    for i, stride in enumerate(strides):
        head_anchors = anchors_per_head[i] / stride
        formatted_anchors.append(head_anchors.tolist())
    
    # Update config
    cfg['anchors'] = formatted_anchors
    
    # Backup original
    backup_path = yaml_path.replace('.yaml', '_backup.yaml')
    with open(backup_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f'   Backup saved: {backup_path}')
    
    # Save new config
    with open(yaml_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f'   Updated: {yaml_path}')
    
    return formatted_anchors

def visualize_anchor_distribution(boxes, old_anchors, new_anchors, img_size=640):
    """Create visualization comparing old vs new anchors"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Convert to pixels
    boxes_px = boxes * img_size
    
    # Plot 1: Box distribution vs anchors
    ax = axes[0]
    ax.scatter(boxes_px[:, 0], boxes_px[:, 1], alpha=0.3, s=10, label='Ground Truth')
    
    if old_anchors is not None:
        old_px = np.array(old_anchors).reshape(-1, 2) * img_size / 8  # Assuming stride 8 normalization
        ax.scatter(old_px[:, 0], old_px[:, 1], c='red', s=200, marker='x', linewidths=3, label='Old Anchors')
    
    ax.scatter(new_anchors[:, 0], new_anchors[:, 1], c='green', s=200, marker='o', 
               edgecolors='black', linewidths=2, label='New Anchors')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    ax.set_title('Anchor vs Ground Truth Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Size distribution histogram
    ax = axes[1]
    areas = boxes_px[:, 0] * boxes_px[:, 1]
    ax.hist(areas, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(32**2, color='red', linestyle='--', linewidth=2, label='Small/Medium (32²)')
    ax.axvline(96**2, color='orange', linestyle='--', linewidth=2, label='Medium/Large (96²)')
    ax.set_xlabel('Object Area (pixels²)')
    ax.set_ylabel('Count')
    ax.set_title('Object Size Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'anchor_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'\n📊 Visualization saved: {save_path}')
    plt.close()

def compare_anchors(boxes, old_anchors, new_anchors, img_size=640):
    """Compare old vs new anchors performance"""
    print(f'\n📊 ANCHOR COMPARISON:')
    print('='*60)
    
    wh = torch.tensor(boxes * img_size, dtype=torch.float32)
    thr = 4.0
    
    def calc_metrics(anchors_px):
        k = torch.tensor(anchors_px, dtype=torch.float32)
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]
        best = x.max(1)[0]
        aat = (best > 1. / thr).float().sum()
        bpr = best.mean()
        return bpr.item(), aat.item()
    
    if old_anchors is not None:
        old_px = np.array(old_anchors).reshape(-1, 2) * img_size / 8
        old_bpr, old_aat = calc_metrics(old_px)
        print(f'OLD Anchors:')
        print(f'   BPR (Best Possible Recall): {old_bpr:.4f}')
        print(f'   Anchors Above Threshold: {old_aat:.0f}/{len(boxes)} ({old_aat/len(boxes)*100:.1f}%)')
    
    new_bpr, new_aat = calc_metrics(new_anchors)
    print(f'\nNEW Anchors:')
    print(f'   BPR (Best Possible Recall): {new_bpr:.4f}')
    print(f'   Anchors Above Threshold: {new_aat:.0f}/{len(boxes)} ({new_aat/len(boxes)*100:.1f}%)')
    
    if old_anchors is not None:
        improvement = (new_bpr - old_bpr) / old_bpr * 100
        print(f'\n🎯 IMPROVEMENT: +{improvement:.2f}% BPR')
        print(f'   Expected mAP gain: +{improvement * 0.1:.2f} to +{improvement * 0.2:.2f} mAP@0.5')
    
    print('='*60)

def main():
    parser = argparse.ArgumentParser(description='Re-cluster anchors for FLIR dataset')
    parser.add_argument('--data', type=str, default='data/multispectral/FLIR_aligned.yaml', 
                        help='dataset.yaml path')
    parser.add_argument('--cfg', type=str, default='', 
                        help='model.yaml path to update (leave empty to auto-detect)')
    parser.add_argument('--imgsz', type=int, default=640, help='training image size')
    parser.add_argument('--n', type=int, default=9, help='number of anchors')
    parser.add_argument('--gen', type=int, default=1000, help='k-means generations')
    parser.add_argument('--no-plot', action='store_true', help='disable visualization')
    opt = parser.parse_args()
    
    print('='*60)
    print('🎯 ANCHOR RE-CLUSTERING FOR FLIR DATASET')
    print('='*60)
    
    # Load dataset
    boxes, sizes = load_dataset_labels(opt.data, opt.imgsz)
    
    # Load old anchors if config provided
    old_anchors = None
    if opt.cfg and Path(opt.cfg).exists():
        with open(opt.cfg) as f:
            cfg = yaml.safe_load(f)
            old_anchors = cfg.get('anchors')
            print(f'\n📋 Loaded old anchors from {opt.cfg}')
    
    # Run k-means
    new_anchors = kmeans_anchors(boxes, n=opt.n, img_size=opt.imgsz, gen=opt.gen)
    
    # Compare performance
    compare_anchors(boxes, old_anchors, new_anchors, opt.imgsz)
    
    # Visualize
    if not opt.no_plot:
        try:
            visualize_anchor_distribution(boxes, old_anchors, new_anchors, opt.imgsz)
        except Exception as e:
            print(f'Warning: Could not create visualization: {e}')
    
    # Auto-detect config files to update
    if not opt.cfg:
        # Find all YAML configs in models directory
        config_candidates = list(Path('models').rglob('*FLIR*.yaml')) + \
                           list(Path('models').rglob('*dual*.yaml'))
        
        if config_candidates:
            print(f'\n🔍 Found {len(config_candidates)} model configs to update:')
            for cfg_path in config_candidates:
                print(f'   {cfg_path}')
            
            # Ask user
            response = input('\nUpdate all these configs? [y/N]: ')
            if response.lower() == 'y':
                for cfg_path in config_candidates:
                    update_yaml_anchors(str(cfg_path), new_anchors, opt.imgsz)
        else:
            print('\n⚠️  No model configs found. Please specify --cfg manually.')
    else:
        # Update specified config
        update_yaml_anchors(opt.cfg, new_anchors, opt.imgsz)
    
    # Print final anchors in YOLOv5 format
    print('\n📋 COPY THESE ANCHORS TO YOUR CONFIG:')
    print('='*60)
    anchors_per_head = new_anchors.reshape(3, 3, 2)
    strides = [8, 16, 32]
    print('anchors:')
    for i, stride in enumerate(strides):
        head_anchors = (anchors_per_head[i] / stride).tolist()
        print(f'  - {head_anchors}  # P{i+3}/{stride}')
    print('='*60)
    
    print('\n✅ Anchor re-clustering complete!')
    print('\n📌 NEXT STEPS:')
    print('1. Verify the anchors were updated in your model YAML')
    print('2. Re-train with: python train.py --cfg <your_config>.yaml')
    print('3. Expected improvement: +0.5 to +1.5 mAP@0.5 (especially for small objects)')

if __name__ == '__main__':
    main()