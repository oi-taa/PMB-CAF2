import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader, create_dataloader_rgb_ir
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

def compute_efficiency_metrics(model, device, input_shape=(640, 640)):
    """Compute FPS, Params, GFLOPs"""
    import time
    try:
        from thop import profile
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return {'params_M': 0, 'gflops': 0, 'fps': 0}
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Create dummy inputs
    rgb_input = torch.randn(1, 3, *input_shape).to(device)
    thermal_input = torch.randn(1, 3, *input_shape).to(device)
    
    # GFLOPs
    try:
        macs, _ = profile(model, inputs=(rgb_input, thermal_input), verbose=False)
        gflops = macs / 1e9
    except:
        gflops = 0.0
    
    # FPS measurement
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(rgb_input, thermal_input)
        
        if device.type != 'cpu':
            torch.cuda.synchronize()
        
        # Timing
        start_time = time.time()
        num_runs = 50
        for _ in range(num_runs):
            _ = model(rgb_input, thermal_input)
        
        if device.type != 'cpu':
            torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / num_runs
        fps = 1.0 / avg_time
    
    return {
        'params_M': total_params,
        'gflops': gflops,
        'fps': fps,
        'inference_time_ms': avg_time * 1000
    }

def compute_size_based_ap_safe(stats_with_areas, all_gt_areas, img_wh=None):
    """
    Size-based AP using official COCO methodology
    NOTE: This is approximate - for reportable metrics use --save-json + pycocotools
    """
    import numpy as np
    from utils.metrics import ap_per_class

    default_metrics = {
        'mAP_small': 0.0, 'mAP_medium': 0.0, 'mAP_large': 0.0,
        'small_count': 0, 'medium_count': 0, 'large_count': 0
    }

    if not stats_with_areas or len(stats_with_areas) < 6:
        print("⚠️  Size-based AP: stats missing")
        return default_metrics

    try:
        tp, conf, pred_cls, target_cls, matched_gt_areas, matched_gt_idx = stats_with_areas
        tp = np.asarray(tp)
        conf = np.asarray(conf)
        pred_cls = np.asarray(pred_cls)
        target_cls = np.asarray(target_cls)
        all_gt_areas = np.asarray(all_gt_areas).astype(float)
        matched_gt_idx = np.asarray(matched_gt_idx).astype(int)

        if tp.size == 0 or len(all_gt_areas) == 0:
            return default_metrics

        # COCO thresholds
        small_thresh = 32 ** 2
        large_thresh = 96 ** 2

        small_gt_mask = all_gt_areas < small_thresh
        medium_gt_mask = (all_gt_areas >= small_thresh) & (all_gt_areas < large_thresh)
        large_gt_mask = all_gt_areas >= large_thresh

        small_gt_count = int(small_gt_mask.sum())
        medium_gt_count = int(medium_gt_mask.sum())
        large_gt_count = int(large_gt_mask.sum())

        print("\n📏 Ground Truth Size Distribution:")
        print(f"   Small:  {small_gt_count} objects (< 32²)")
        print(f"   Medium: {medium_gt_count} objects (32²-96²)")
        print(f"   Large:  {large_gt_count} objects (> 96²)")

        def compute_for_size(gt_mask, size_name):
            gt_idx_this_size = np.where(gt_mask)[0]
            num_gts = len(gt_idx_this_size)
            
            if num_gts == 0:
                print(f"   {size_name}: No GTs → skip")
                return 0.0
            
            # Find predictions matched to this size
            pred_matched_this_size = np.array([
                (matched_gt_idx[i] in gt_idx_this_size) if matched_gt_idx[i] >= 0 else False
                for i in range(len(matched_gt_idx))
            ])
            
            if not pred_matched_this_size.any():
                print(f"   {size_name}: No matched predictions → 0.0")
                return 0.0
            
            # Subset to matched predictions only
            tp_subset = tp[pred_matched_this_size]
            conf_subset = conf[pred_matched_this_size]
            pred_cls_subset = pred_cls[pred_matched_this_size]
            tcls_subset = target_cls[gt_mask]
            
            # Compute AP
            p, r, ap, f1, ap_class = ap_per_class(
                tp_subset, conf_subset, pred_cls_subset, tcls_subset, plot=False
            )
            
            if ap is None or ap.size == 0:
                return 0.0
            
            # mAP@[0.5:0.95]
            ap = np.asarray(ap)
            mean_ap = float(np.nanmean(ap.mean(1) if ap.ndim == 2 else ap))
            
            tp_count = int((tp_subset[:, 0] if tp_subset.ndim == 2 else tp_subset).sum())
            print(f"   {size_name}: {tp_count}/{num_gts} TPs → mAP@[0.5:0.95] = {mean_ap:.3f}")
            
            return mean_ap

        return {
            'small_count': small_gt_count,
            'medium_count': medium_gt_count,
            'large_count': large_gt_count,
            'mAP_small': compute_for_size(small_gt_mask, "Small"),
            'mAP_medium': compute_for_size(medium_gt_mask, "Medium"),
            'mAP_large': compute_for_size(large_gt_mask, "Large")
        }

    except Exception as e:
        import traceback
        print(f"❌ Size AP error: {e}")
        traceback.print_exc()
        return default_metrics

def evaluate_coco_size_based_official(pred_json, anno_json):
    """
    Official COCO evaluation with size-based metrics
    This produces REPORTABLE results for publications
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        print("\n" + "="*80)
        print("📊 OFFICIAL COCO SIZE-BASED EVALUATION (REPORTABLE)")
        print("="*80)
        
        coco_gt = COCO(anno_json)
        coco_dt = coco_gt.loadRes(pred_json)
        
        results = {}
        
        # Overall
        print("\n🎯 OVERALL:")
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        results['mAP_overall'] = float(coco_eval.stats[0])
        results['mAP50_overall'] = float(coco_eval.stats[1])
        
        # Size-specific
        for area_name, area_range in [('small', [0, 32**2]), 
                                      ('medium', [32**2, 96**2]), 
                                      ('large', [96**2, 1e5**2])]:
            print(f"\n📏 {area_name.upper()} (area: {area_range[0]}-{area_range[1]} px²):")
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.params.areaRng = [area_range]
            coco_eval.params.areaRngLbl = [area_name]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            results[f'mAP_{area_name}'] = float(coco_eval.stats[0])
            results[f'mAP50_{area_name}'] = float(coco_eval.stats[1])
        
        print("\n" + "="*80)
        print("✅ OFFICIAL REPORTABLE METRICS:")
        print("="*80)
        print(f"Overall  mAP@[0.5:0.95]: {results['mAP_overall']:.3f}")
        print(f"Small    mAP@[0.5:0.95]: {results['mAP_small']:.3f}  ← REPORTABLE")
        print(f"Medium   mAP@[0.5:0.95]: {results['mAP_medium']:.3f}  ← REPORTABLE")
        print(f"Large    mAP@[0.5:0.95]: {results['mAP_large']:.3f}  ← REPORTABLE")
        print("="*80 + "\n")
        
        return results
        
    except Exception as e:
        print(f"❌ Official COCO evaluation failed: {e}")
        print("   Install: pip install pycocotools")
        import traceback
        traceback.print_exc()
        return {}

def compute_efficiency_metrics(model, device, input_shape=(640, 640)):
    """Compute FPS, Params, GFLOPs for efficiency analysis"""
    import time
    try:
        from thop import profile
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return {'params_M': 0, 'gflops': 0, 'fps': 0, 'inference_time_ms': 0}
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # Create dummy inputs
    rgb_input = torch.randn(1, 3, *input_shape).to(device)
    thermal_input = torch.randn(1, 3, *input_shape).to(device)
    
    # GFLOPs calculation
    try:
        macs, _ = profile(model, inputs=(rgb_input, thermal_input), verbose=False)
        gflops = macs / 1e9
    except Exception as e:
        print(f"Warning: GFLOPs calculation failed: {e}")
        gflops = 0.0
    
    # FPS measurement
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(rgb_input, thermal_input)
        
        if device.type != 'cpu':
            torch.cuda.synchronize()
        
        # Timing
        start_time = time.time()
        num_runs = 50
        for _ in range(num_runs):
            _ = model(rgb_input, thermal_input)
        
        if device.type != 'cpu':
            torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / num_runs
        fps = 1.0 / avg_time
    
    return {
        'params_M': total_params,
        'gflops': gflops,
        'fps': fps,
        'inference_time_ms': avg_time * 1000
    }

def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=True,  # save auto-label confidences
         plots=False,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         opt=None):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        if opt.cfg:
            # If config provided, build model and load weights
            from models.yolo_test import Model
            import yaml
            with open(opt.cfg) as f:
                cfg = yaml.safe_load(f)
            model = Model(cfg=cfg).to(device)
            ckpt = torch.load(weights[0], map_location=device, weights_only=False)
            state_dict = ckpt.get('ema') if ckpt.get('ema') else ckpt.get('model')
            model.load_state_dict(state_dict, strict=False)
            model.float().eval()
        else:
            # Use attempt_load (for old checkpoints)
            model = attempt_load(weights, map_location=device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params/1e6:.1f}M")
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        print(opt.task)
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        val_path_rgb = data['val_rgb']
        val_path_ir = data['val_ir']
        dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    all_gt_areas = [] 
    gt_offset = 0

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        img_rgb = img[:, :3, :, :]
        img_ir = img[:, 3:, :, :]

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            output = model(img_rgb, img_ir, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t
            if isinstance(output, tuple):
                if len(output) == 2:
                    first, second = output
                    
                    # Check if second is obj_clean (tensor) or train_out (list)
                    if isinstance(second, torch.Tensor) and second.dim() == 4 and second.shape[1] == 1:
                        # (Detect_output, obj_clean)
                        detect_out = first
                        obj_clean = second
                        
                        # Detect itself might return tuple in eval mode
                        if isinstance(detect_out, tuple):
                            out = detect_out[0]  # Get concatenated predictions
                            train_out = detect_out[1] if len(detect_out) > 1 else None
                        else:
                            out = detect_out
                            train_out = None
                            
                    elif isinstance(second, list):
                        # (out, train_out) - normal training
                        out = first
                        train_out = second
                    else:
                        # Fallback
                        out = first
                        train_out = None
                else:
                    out = output[0]
                    train_out = None
            else:
                out = output
                train_out = None

            # Compute loss
            if compute_loss and train_out is not None:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    # No predictions but targets exist - add empty stats with matched_areas
                    stats.append((
                        torch.zeros(0, niou, dtype=torch.bool), 
                        torch.Tensor(), 
                        torch.Tensor(), 
                        tcls,
                        torch.zeros(0)  # ✅ Empty matched_areas
                    ))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                "class_id": int(cls),
                                "box_caption": "%s %.3f" % (names[cls], conf),
                                "scores": {"class_score": conf},
                                "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    rgb_img = img[si][:3] if img[si].shape[0] > 3 else img[si]  # Take first 3 channels
                    wandb_images.append(wandb_logger.wandb.Image(rgb_img, boxes=boxes, caption=path.name))
                    
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                'bbox': [round(x, 3) for x in b],
                                'score': round(p[4], 5)})

            # =====================================================================
            # ✅ CRITICAL ADDITION: Track matched GT areas for size-based metrics
            # =====================================================================
            
            matched_areas = torch.zeros(pred.shape[0], device=device, dtype=torch.float32)
            matched_gt_idx = torch.full((pred.shape[0],), -1, dtype=torch.long, device=device) 
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)

            if nl:
                detected = set()  # set of detected target indices (python ints)
                tcls_tensor = labels[:, 0]

                # target boxes in native (pixel) coords
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if nl > 0:
                    gt_widths = (tbox[:, 2] - tbox[:, 0]).clamp(min=0.0)
                    gt_heights = (tbox[:, 3] - tbox[:, 1]).clamp(min=0.0)
                    gt_areas_batch = (gt_widths * gt_heights).cpu().numpy()
                    all_gt_areas.extend(gt_areas_batch)

                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # iterate per-class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=True)[0]  # target indices (tensor 1D)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=True)[0]   # prediction indices (tensor 1D)

                    if pi.numel() == 0 or ti.numel() == 0:
                        continue

                    # compute IoUs between these preds and targets (predn already scaled)
                    ious_all, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best IoUs, indices into ti

                    # indices of predictions that exceed the primary IoU threshold (iouv[0])
                    pass_inds = (ious_all > iouv[0]).nonzero(as_tuple=True)[0]  # 1D tensor of indices into ious_all/pi

                    for idx in pass_inds.tolist():
                        pred_idx = pi[idx].item()
                        tgt_idx = ti[i[idx].item()].item()
                        
                        # ✅ FIX: Check detected BEFORE assigning anything
                        if tgt_idx in detected:
                            continue
                        detected.add(tgt_idx)
                        
                        # Now assign (only if GT wasn't already matched)
                        correct[pred_idx] = ious_all[idx] > iouv
                        
                        # ✅ FIX: Compute area here (after the continue check)
                        matched_gt_box = tbox[tgt_idx]
                        mw = (matched_gt_box[2] - matched_gt_box[0]).clamp(min=0.0)
                        mh = (matched_gt_box[3] - matched_gt_box[1]).clamp(min=0.0)
                        matched_areas[pred_idx] = (mw * mh)
                        matched_gt_idx[pred_idx] = gt_offset + tgt_idx 
                        
                        if len(detected) == nl:
                            break
                gt_offset += nl

            # ✅ Append statistics WITH matched areas
            stats.append((
                correct.cpu(), 
                pred[:, 4].cpu(), 
                pred[:, 5].cpu(), 
                tcls,
                matched_areas.cpu(),  # ✅ Include matched GT areas
                matched_gt_idx.cpu() 
            ))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # =====================================================================
    # ✅ COMPUTE STATISTICS WITH SIZE-BASED METRICS
    # =====================================================================

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy

    if len(stats) and stats[0].any():
        # Standard AP computation (first 4 elements)
        p, r, ap, f1, ap_class = ap_per_class(*stats[:4], plot=plots, save_dir=save_dir, names=names)
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.75, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
        mp, mr, map50, map75, map = 0., 0., 0., 0., 0.

    # ✅ Compute size-based metrics with correct signature
    print("\n" + "="*80)
    print("📏 Computing Size-Based Performance Metrics...")
    print("="*80)
    # Add this RIGHT BEFORE: size_metrics = compute_size_based_ap_safe(stats, img_wh=imgsz)

    
    size_metrics = compute_size_based_ap_safe(stats, all_gt_areas, img_wh=imgsz)
    print(f"\n{'='*80}")
    print(f"📊 GROUND TRUTH SIZE DISTRIBUTION (COCO Thresholds)")
    print(f"{'='*80}")
    total_gts = len(all_gt_areas)
    print(f"Total GTs processed: {total_gts}")
    if total_gts > 0:
        areas_arr = np.array(all_gt_areas)
        print(f"  Area range: [{areas_arr.min():.1f}, {areas_arr.max():.1f}] pixels²")
        print(f"  Mean area: {areas_arr.mean():.1f} pixels²")
        print(f"  Median area: {np.median(areas_arr):.1f} pixels²")
    print(f"{'='*80}\n")

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 5  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))

    # Print size-based results
    print(f"\nSize-based Performance:")
    print(f"  Small objects:  mAP@0.5 = {size_metrics['mAP_small']:.3f} ({size_metrics['small_count']} objects)")
    print(f"  Medium objects: mAP@0.5 = {size_metrics['mAP_medium']:.3f} ({size_metrics['medium_count']} objects)")
    print(f"  Large objects:  mAP@0.5 = {size_metrics['mAP_large']:.3f} ({size_metrics['large_count']} objects)")

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''
        
        # Determine annotation file path
        if 'coco' in data.get('val_rgb', '').lower():
            anno_json = '../coco/annotations/instances_val2017.json'
        else:
            # For custom datasets, look for annotations.json in data directory
            data_path = Path(data.get('path', data.get('val_rgb', '.'))).parent
            anno_json = str(data_path / 'annotations.json')
        
        pred_json = str(save_dir / f"{w}_predictions.json")
        print(f'\n💾 Saving predictions to {pred_json}')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        # Try official COCO evaluation
        if Path(anno_json).exists():
            print(f"📂 Using GT annotations: {anno_json}")
            official_results = evaluate_coco_size_based_official(pred_json, anno_json)
            
            # Update size metrics with official results if available
            if official_results:
                if 'mAP_small' in official_results:
                    size_metrics['mAP_small'] = official_results['mAP_small']
                    size_metrics['mAP_medium'] = official_results['mAP_medium']
                    size_metrics['mAP_large'] = official_results['mAP_large']
                    print("✅ Using OFFICIAL size-based metrics")
        else:
            print(f"⚠️  GT annotations not found: {anno_json}")
            print("   Create with: python create_coco_annotations.py")
            print("   Using approximate size metrics instead")
            
            # Try standard COCO eval for overall metrics
            try:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval

                if 'coco' in data.get('val_rgb', '').lower():
                    anno = COCO(anno_json)
                    pred = anno.loadRes(pred_json)
                    eval = COCOeval(anno, pred, 'bbox')
                    if is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    map, map50 = eval.stats[:2]
            except Exception as e:
                print(f'Standard pycocotools unable to run: {e}')

    # Return results with size metrics
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    
    # Enhanced return with size metrics
    enhanced_results = (mp, mr, map50, map75, map, *(loss.cpu() / len(dataloader)).tolist(), 
                       size_metrics['mAP_small'], size_metrics['mAP_medium'], size_metrics['mAP_large'])
    
    return enhanced_results, maps, t

'''
def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=True,  # save auto-label confidences
         plots=False,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         is_coco=False,
         opt=None):
    # Initialize/load model and set device
    training = model is not None
    
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        print(opt.task)
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        val_path_rgb = data['val_rgb']
        val_path_ir = data['val_ir']
        dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    all_target_areas = [] 
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map75, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    # for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(testloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        if len(targets) > 0:
            # Convert target format [batch_idx, class, x_center, y_center, width, height] to areas
            target_widths = targets[:, 4] * width  # denormalize width
            target_heights = targets[:, 5] * height  # denormalize height  
            target_areas = target_widths * target_heights
            all_target_areas.extend(target_areas.cpu().numpy())

        img_rgb = img[:, :3, :, :]
        img_ir = img[:, 3:, :, :]

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img_rgb, img_ir, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    rgb_img = img[si][:3] if img[si].shape[0] > 3 else img[si]  # Take first 3 channels
                    wandb_images.append(wandb_logger.wandb.Image(rgb_img, boxes=boxes, caption=path.name))
                    
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()
    
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        # print("mAP75", ap[:, 5].mean(-1))
        ap50, ap75, ap = ap[:, 0], ap[:, 5], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map75, map = p.mean(), r.mean(), ap50.mean(), ap75.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    targets_info = {'areas': all_target_areas}
    size_metrics = compute_size_based_metrics(stats, targets_info, imgsz)
    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 5  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map75, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap75[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)

        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map75, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, size_metrics
'''

def print_publication_ready_results(results, size_metrics, efficiency_metrics, dataset_name, method_name="PMB-CAF"):
    """Print results in publication-ready format"""
    
    print("\n" + "="*80)
    print(f"🎯 {method_name} - {dataset_name} Results")
    print("="*80)
    
    # Main metrics table
    mp, mr, map50, map75, map_avg = results[:5]
    
    print(f"\n📊 Detection Performance:")
    print(f"  mAP@50:       {map50:.3f}")
    print(f"  mAP@75:       {map75:.3f}")  
    print(f"  mAP@[0.5:0.95]: {map_avg:.3f}")
    print(f"  Precision:    {mp:.3f}")
    print(f"  Recall:       {mr:.3f}")
    
    # Size-based metrics
    print(f"\n📏 Size-based Performance:")
    print(f"  mAP_small:    {size_metrics.get('mAP_small', 0.0):.3f} ({size_metrics.get('small_count', 0)} objects)")
    print(f"  mAP_medium:   {size_metrics.get('mAP_medium', 0.0):.3f} ({size_metrics.get('medium_count', 0)} objects)")
    print(f"  mAP_large:    {size_metrics.get('mAP_large', 0.0):.3f} ({size_metrics.get('large_count', 0)} objects)")
    
    # Efficiency metrics
    print(f"\n⚡ Efficiency:")
    print(f"  Params:       {efficiency_metrics.get('params_M', 0.0):.2f}M")
    print(f"  GFLOPs:       {efficiency_metrics.get('gflops', 0.0):.2f}")
    print(f"  FPS:          {efficiency_metrics.get('fps', 0.0):.1f}")
    
    # Summary table for paper
    print(f"\n📋 Summary Table (for paper):")
    print(f"Method: {method_name}")
    print(f"Dataset: {dataset_name}")
    print(f"mAP@50: {map50:.3f} | mAP@75: {map75:.3f} | mAP@[0.5:0.95]: {map_avg:.3f}")
    print(f"mAP_S: {size_metrics.get('mAP_small', 0.0):.3f} | mAP_M: {size_metrics.get('mAP_medium', 0.0):.3f} | mAP_L: {size_metrics.get('mAP_large', 0.0):.3f}")
    print(f"Params: {efficiency_metrics.get('params_M', 0.0):.2f}M | GFLOPs: {efficiency_metrics.get('gflops', 0.0):.2f} | FPS: {efficiency_metrics.get('fps', 0.0):.1f}")
    
    print("="*80)
    
def test_with_comprehensive_metrics(data, weights=None, **kwargs):
    """Enhanced test function with comprehensive metrics including size-based mAP"""
    
    # Run the original test function
    results, maps, times = test(data, weights, **kwargs)
    
    # Get efficiency metrics if model is available
    if kwargs.get('model') is not None:
        model = kwargs['model']
        device = next(model.parameters()).device
        efficiency_metrics = compute_efficiency_metrics(model, device)
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("📊 Comprehensive Evaluation Results")
        print("="*60)
        
        # Main performance
        mp, mr, map50, map75, map_avg = results[:5]
        print(f"Overall Performance:")
        print(f"  mAP@0.5:      {map50:.3f}")
        print(f"  mAP@0.5:0.95: {map_avg:.3f}")
        print(f"  mAP@0.75:     {map75:.3f}")
        print(f"  Precision:    {mp:.3f}")
        print(f"  Recall:       {mr:.3f}")
        
        # Efficiency metrics
        print(f"\nEfficiency Metrics:")
        print(f"  Parameters:   {efficiency_metrics['params_M']:.2f}M")
        print(f"  GFLOPs:       {efficiency_metrics['gflops']:.2f}")
        print(f"  FPS:          {efficiency_metrics['fps']:.1f}")
        print(f"  Inference:    {efficiency_metrics['inference_time_ms']:.2f}ms")
        
        # Speed breakdown
        inf_time, nms_time, total_time = times[:3]
        print(f"\nSpeed Breakdown:")
        print(f"  Inference:    {inf_time:.1f}ms")
        print(f"  NMS:          {nms_time:.1f}ms") 
        print(f"  Total:        {total_time:.1f}ms")
        
        print("="*60)
        
        # Create comprehensive metrics dictionary
        comprehensive_metrics = {
            'mAP@50': float(map50),
            'mAP@75': float(map75), 
            'mAP@[0.5:0.95]': float(map_avg),
            'precision': float(mp),
            'recall': float(mr),
            'params_M': efficiency_metrics['params_M'],
            'gflops': efficiency_metrics['gflops'],
            'fps': efficiency_metrics['fps'],
            'inference_time_ms': efficiency_metrics['inference_time_ms'],
            'nms_time_ms': inf_time,
            'total_time_ms': total_time
        }
        
        return results, maps, times, comprehensive_metrics
    else:
        print("⚠️  Model not available for efficiency metrics")
        return results, maps, times, {}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='/home/fqy/proj/multispectral-object-detection/best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='./data/multispectral/FLIR_aligned.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=64, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')  # ← ADD THIS LINE
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', default=True, action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    print(opt.data)
    check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             opt=opt
             )
    # results, maps, times = test.test(data_dict,
    #                                  batch_size=batch_size * 2,
    #                                  imgsz=imgsz_test,
    #                                  model=ema.ema,
    #                                  single_cls=opt.single_cls,
    #                                  dataloader=testloader,
    #                                  save_dir=save_dir,
    #                                  verbose=nc < 50 and final_epoch,
    #                                  plots=plots and final_epoch,
    #                                  wandb_logger=wandb_logger,
    #                                  compute_loss=compute_loss,
    #                                  is_coco=is_coco)

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, opt=opt)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, opt=opt)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
