# Model validation metrics

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import general


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    # print("p[:, i]", p[:, i])
    # print("p[:, i]", p[:, i].shape())
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general.box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        except Exception as e:
            pass

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


# Plots ----------------------------------------------------------------------------------------------------------------

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
def ap_per_class_with_size_analysis(tp, conf, pred_cls, target_cls, target_areas, img_size=640, plot=False, save_dir='.', names=()):
    """
    Enhanced ap_per_class with size-based analysis
    
    Args:
        target_areas: Array of target bounding box areas (normalized 0-1)
        img_size: Image size for converting normalized areas to pixels
    """
    
    # Convert normalized areas to pixel areas
    pixel_areas = target_areas * (img_size ** 2)
    
    # Define size categories based on your dataset's distribution
    area_percentiles = np.percentile(pixel_areas, [33, 67])
    small_thresh, large_thresh = area_percentiles[0], area_percentiles[1]
    
    # Categorize targets
    size_mask_small = pixel_areas < small_thresh
    size_mask_medium = (pixel_areas >= small_thresh) & (pixel_areas < large_thresh)
    size_mask_large = pixel_areas >= large_thresh
    
    # Compute AP for each size category
    results = {}
    
    # Overall AP (existing function)
    p_all, r_all, ap_all, f1_all, unique_classes = ap_per_class(
        tp, conf, pred_cls, target_cls, plot, save_dir, names
    )
    
    results['overall'] = {
        'precision': p_all,
        'recall': r_all, 
        'ap': ap_all,
        'f1': f1_all,
        'classes': unique_classes
    }
    
    # Size-specific AP
    for size_name, size_mask in [('small', size_mask_small), 
                                 ('medium', size_mask_medium), 
                                 ('large', size_mask_large)]:
        if size_mask.sum() > 0:  # Only if we have objects in this size category
            # Filter targets by size
            size_indices = np.where(size_mask)[0]
            
            # Create mask for predictions that match these targets
            pred_mask = np.isin(np.arange(len(pred_cls)), size_indices)
            
            if pred_mask.sum() > 0:
                p_size, r_size, ap_size, f1_size, _ = ap_per_class(
                    tp[pred_mask], conf[pred_mask], pred_cls[pred_mask], 
                    target_cls[size_indices], False, save_dir, names
                )
                
                results[size_name] = {
                    'precision': p_size,
                    'recall': r_size,
                    'ap': ap_size, 
                    'f1': f1_size,
                    'count': size_mask.sum()
                }
            else:
                # No predictions for this size category
                results[size_name] = {
                    'precision': np.zeros_like(p_all),
                    'recall': np.zeros_like(r_all),
                    'ap': np.zeros_like(ap_all),
                    'f1': np.zeros_like(f1_all),
                    'count': size_mask.sum()
                }
    
    return results

def compute_efficiency_metrics(model, input_shape=(640, 640), device='cuda'):
    """
    Compute efficiency metrics: FPS, Params, GFLOPs
    """
    import time
    from thop import profile
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # Millions
    
    # Create dummy inputs 
    rgb_input = torch.randn(1, 3, *input_shape).to(device)
    thermal_input = torch.randn(1, 3, *input_shape).to(device)
    
    # GFLOPs calculation
    try:
        macs, _ = profile(model, inputs=(rgb_input, thermal_input), verbose=False)
        gflops = macs / 1e9
    except:
        gflops = 0.0
        print("Warning: GFLOPs calculation failed")
    
    # FPS calculation
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(rgb_input, thermal_input)
        
        # Timing
        torch.cuda.synchronize()
        start_time = time.time()
        
        num_runs = 100
        for _ in range(num_runs):
            _ = model(rgb_input, thermal_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time
    
    return {
        'params_M': total_params,
        'gflops': gflops, 
        'fps': fps,
        'inference_time_ms': avg_time * 1000
    }

def compute_fusion_analysis(rgb_results, thermal_results, fusion_results):
    """
    Analyze fusion benefits vs individual modalities
    """
    fusion_gain = {}
    
    # Extract mAP values (assuming results format from your test function)
    rgb_map = rgb_results[2] if len(rgb_results) > 2 else 0  # mAP@0.5
    thermal_map = thermal_results[2] if len(thermal_results) > 2 else 0
    fusion_map = fusion_results[2] if len(fusion_results) > 2 else 0
    
    rgb_map95 = rgb_results[3] if len(rgb_results) > 3 else 0  # mAP@0.5:0.95  
    thermal_map95 = thermal_results[3] if len(thermal_results) > 3 else 0
    fusion_map95 = fusion_results[3] if len(fusion_results) > 3 else 0
    
    # Compute gains
    fusion_gain['map50_gain'] = fusion_map - max(rgb_map, thermal_map)
    fusion_gain['map50_95_gain'] = fusion_map95 - max(rgb_map95, thermal_map95)
    
    fusion_gain['rgb_map50'] = rgb_map
    fusion_gain['thermal_map50'] = thermal_map
    fusion_gain['fusion_map50'] = fusion_map
    
    fusion_gain['rgb_map50_95'] = rgb_map95
    fusion_gain['thermal_map50_95'] = thermal_map95  
    fusion_gain['fusion_map50_95'] = fusion_map95
    
    return fusion_gain

def print_comprehensive_results(overall_results, size_results, efficiency_metrics, fusion_analysis, class_names):
    """
    Print results in publication-ready format
    """
    print("\n" + "="*80)
    print("PMB-CAF Comprehensive Evaluation Results")
    print("="*80)
    
    # Main performance table
    print(f"\n📊 Overall Performance:")
    print(f"  mAP@0.5:      {overall_results[2]:.3f}")
    print(f"  mAP@0.5:0.95: {overall_results[3]:.3f}")  
    print(f"  Precision:    {overall_results[0]:.3f}")
    print(f"  Recall:       {overall_results[1]:.3f}")
    
    # Size-based analysis
    print(f"\n📏 Performance by Object Size:")
    for size in ['small', 'medium', 'large']:
        if size in size_results:
            ap50 = size_results[size]['ap'][:, 0].mean() if len(size_results[size]['ap']) > 0 else 0
            ap95 = size_results[size]['ap'][:, 1].mean() if size_results[size]['ap'].shape[1] > 1 else 0
            count = size_results[size]['count']
            print(f"  {size.capitalize()} Objects: mAP@0.5={ap50:.3f}, mAP@0.5:0.95={ap95:.3f}, Count={count}")
    
    # Per-class results
    print(f"\n🎯 Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        if i < len(overall_results[0]):
            print(f"  {class_name}: P={overall_results[0][i]:.3f}, R={overall_results[1][i]:.3f}")
    
    # Efficiency metrics  
    print(f"\n⚡ Efficiency Metrics:")
    print(f"  Parameters:   {efficiency_metrics['params_M']:.2f}M")
    print(f"  GFLOPs:       {efficiency_metrics['gflops']:.2f}")
    print(f"  FPS:          {efficiency_metrics['fps']:.1f}")
    print(f"  Inference:    {efficiency_metrics['inference_time_ms']:.2f}ms")
    
    # Fusion analysis
    print(f"\n🔥 Fusion Analysis:")
    print(f"  RGB-only mAP@0.5:     {fusion_analysis['rgb_map50']:.3f}")
    print(f"  Thermal-only mAP@0.5:  {fusion_analysis['thermal_map50']:.3f}")
    print(f"  Fused mAP@0.5:        {fusion_analysis['fusion_map50']:.3f}")
    print(f"  Fusion Gain:          +{fusion_analysis['map50_gain']:.3f}")
    
    print("="*80)