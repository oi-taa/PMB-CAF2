import warnings
# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized

from torch.nn import init, Sequential


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                # (B,C,1,1)
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # init final bias to negative to start gate small
        nn.init.constant_(self.net[-2].bias, -3.0)  # bias of last Conv2d before Sigmoid

    def forward(self, x):
        """
        x: (B, C, H, W) or the 'enhanced' features you want to base gating on
        returns: gate weights (B, C, 1, 1) to be broadcast-multiplied
        """
        return self.net(x)   # (B, C, 1, 1)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


import torch
import torch.nn as nn
import torch.nn.functional as F

class SCP_Enhanced_Upsample(nn.Module):
    """
    SCP module that:
    - Computes confidence gamma from P5 fused features 
    - Applies gamma to projected P5 context, then upsamples to P4 size
    
    Usage: Takes two inputs [p5_fused, p5_ctx_proj] and returns weighted upsampled context
    """
    
    def __init__(self, p5_channels=1024, ctx_channels=64, hidden=256, scale_factor=2, mode='nearest'):
        super(SCP_Enhanced_Upsample, self).__init__()
        self.p5_channels = p5_channels
        self.ctx_channels = ctx_channels
        self.scale_factor = scale_factor
        self.mode = mode
        
        # MLP for gamma: input = p5_channels (1024), output = scalar per sample
        self.gamma_mlp = nn.Sequential(
            nn.Linear(p5_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
        
        # Initialize gamma_mlp with small weights and conservative start
        last_linear = None
        for m in self.gamma_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                last_linear = m
        
        # Initialize final linear bias to negative for conservative start
        if last_linear is not None:
            nn.init.constant_(last_linear.bias, -2.0)  # sigmoid(-3) ≈ 0.047
        
        self._last_confidence = None
    
    def forward(self, x):
        """
        Args:
            x: List/tuple of [p5_fused, p5_ctx_proj]
            p5_fused: [B, 1024, H5, W5] - used to compute gamma
            p5_ctx_proj: [B, 64, H5, W5] - projected context at P5 resolution
            
        Returns:
            weighted_ctx_upsampled: [B, 64, H4, W4] - upsampled weighted context
        """
        if isinstance(x, (list, tuple)) and len(x) == 2:
            p5_fused, p5_ctx_proj = x
        else:
            raise ValueError("SCP_Enhanced_Upsample expects [p5_fused, p5_ctx_proj]")
        
        # Compute gamma from P5 fused features
        b = p5_fused.shape[0]
        g = p5_fused.mean(dim=(2, 3))                    # Global average pool [B, 1024]
        g = self.gamma_mlp(g)                            # [B, 1]
        gamma = torch.sigmoid(g).view(b, 1, 1, 1)        # [B, 1, 1, 1]
        
        # Store for logging
        self._last_confidence = gamma.detach()
        
        # Upsample projected context to P4 size
        p5_ctx_up = F.interpolate(
            p5_ctx_proj, 
            scale_factor=self.scale_factor, 
            mode=self.mode,
            recompute_scale_factor=False
        )  # [B, 64, H4, W4]
        
        # Apply confidence weighting
        weighted_ctx = gamma * p5_ctx_up                 # [B, 64, H4, W4]
        
        return weighted_ctx
    
    def get_confidence_stats(self):
        """Helper for monitoring gamma distribution"""
        if self._last_confidence is None:
            return None
        c = self._last_confidence
        return {
            'mean_confidence': float(c.mean().item()),
            'std_confidence': float(c.std().item()),
            'min_confidence': float(c.min().item()),
            'max_confidence': float(c.max().item())
        }

    
class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)
            

class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        
        
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        
        # return torch.add(x[0], x[1])



class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label, color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x
# Add this to models/common.py - Replace the existing Concat class


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out
    

class BCAM(nn.Module):
    """
    Bidirectional Cross-Attention Module (BCAM)
    
    **Return Values**: Always returns 3-tuple `(rgb_final, thermal_final, fused_final)`
    - External pos override internal per-modality pos embeddings
    - Optional return_attn to get attention weights for debugging/figures
    - Consistent 3-tuple output for debugging and orchestration flexibility
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, 
                 vert_anchors=8, horz_anchors=8, learnable_scale = False):
        super(BCAM, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        assert d_model % num_heads == 0
        #og weight avging - self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.fusion_proj = nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.fusion_proj.weight)
        # spatial handling
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.avgpool = nn.AdaptiveAvgPool2d((vert_anchors, horz_anchors))

        # internal pos embeddings (fallback)
        self.spatial_pos_embed = nn.Parameter(
        torch.randn(1, d_model, vert_anchors, horz_anchors) * 0.02
        )
        self.rgb_modality_token = nn.Parameter(torch.randn(1, d_model, 1, 1) * 0.02)
        self.thermal_modality_token = nn.Parameter(torch.randn(1, d_model, 1, 1) * 0.02)

        # separate norms for RGB vs Thermal
        self.rgb_norm1 = nn.LayerNorm(d_model)
        self.thermal_norm1 = nn.LayerNorm(d_model)
        self.rgb_norm2 = nn.LayerNorm(d_model)
        self.thermal_norm2 = nn.LayerNorm(d_model)

        # projections
        self.rgb_q = nn.Linear(d_model, d_model, bias=False)
        self.thermal_kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.thermal_q = nn.Linear(d_model, d_model, bias=False)
        self.rgb_kv = nn.Linear(d_model, d_model * 2, bias=False)

        # outputs + ffn
        self.rgb_out = nn.Linear(d_model, d_model, bias=False)
        self.thermal_out = nn.Linear(d_model, d_model, bias=False)
        self.learnable_scale = learnable_scale
        if learnable_scale:
            # Initialize at your empirically good value (0.1)
            self.rgb_attn_scale = nn.Parameter(torch.tensor(0.1))
            self.thermal_attn_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.register_buffer('rgb_attn_scale', torch.tensor(0.1))
            self.register_buffer('thermal_attn_scale', torch.tensor(0.1))
        self.rgb_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.thermal_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _add_pos_encoding(self, features, pos_embed):
        """Interpolate and move pos embed to features device/dtype, then add."""
        if pos_embed is None:
            return features
        # move/embed to same device/dtype as features
        pos = pos_embed.to(device=features.device, dtype=features.dtype)
        if pos.shape[-2:] != features.shape[-2:]:
            pos = F.interpolate(pos, size=features.shape[-2:], mode='bilinear', align_corners=False)
        return features + pos

    def _multi_head_attention(self, query, key, value, return_attn=False):
        """
        Multi-head attention.
        query/key/value: (B, L, C)
        If return_attn=True returns (attended (B,L,C), attn_weights_avg (B, L, L))
        attn_weights_avg is attention averaged across heads.
        """
        B, seq_len = query.shape[:2]
        Q = query.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        K = key.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = value.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, L, L)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)  # (B, H, L, D)
        attended = attended.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)  # (B, L, C)

        if return_attn:
            # average attention across heads -> (B, L, L)
            if return_attn:
                attn_avg = attn_weights.mean(dim=1)
               
                
                return attended, attn_avg
        return attended

    def forward(self, x, pos_rgb=None, pos_thermal=None, return_attn=False):
        """
        CANONICAL RULE: This method adds positional encoding exactly once.
        - External pos (pos_rgb/thermal) overrides internal pos embeddings  
        - Caller must NOT pre-add positional encoding to input features
        - This method will add pos after avgpooling
        
        x: tuple (rgb_fea, thermal_fea) each (B, C, H, W)
        pos_rgb/pos_thermal: optional external pos embeddings (or None)
        return_attn: if True, return ((rgb, thermal, fused), {'rgb_to_th':..., 'th_to_rgb':...})
        
        **Return Values**: Always returns 3-tuple `(rgb_final, thermal_final, fused_final)`
        When return_attn=True: returns ((rgb, thermal, fused), attn_info)
        """
        rgb_fea, thermal_fea = x[0], x[1]
        bs, c, h, w = rgb_fea.shape
        
        # Input validation
        assert rgb_fea.shape[1] == self.d_model, \
            f"RGB channels {rgb_fea.shape[1]} != d_model {self.d_model}"
        assert thermal_fea.shape[1] == self.d_model, \
            f"Thermal channels {thermal_fea.shape[1]} != d_model {self.d_model}"

        # avgpool once, reuse
        rgb_pooled = self.avgpool(rgb_fea)
        thermal_pooled = self.avgpool(thermal_fea)

        if pos_rgb is not None and pos_thermal is not None:
            # External override
            rgb_pooled = self._add_pos_encoding(rgb_pooled, pos_rgb)
            thermal_pooled = self._add_pos_encoding(thermal_pooled, pos_thermal)
        else:
            # Use internal shared spatial + modality tokens
            spatial_pos = self.spatial_pos_embed.to(device=rgb_pooled.device, dtype=rgb_pooled.dtype)
            
            if spatial_pos.shape[-2:] != rgb_pooled.shape[-2:]:
                spatial_pos = F.interpolate(spatial_pos, size=rgb_pooled.shape[-2:], 
                                        mode='bilinear', align_corners=False)
            
            # Shared spatial + modality-specific tokens
            rgb_pooled = rgb_pooled + spatial_pos + self.rgb_modality_token
            thermal_pooled = thermal_pooled + spatial_pos + self.thermal_modality_token

        # Convert to tokens ONCE: (B, C, H, W) -> (B, L, C)
        rgb_tokens = rgb_pooled.view(bs, c, -1).permute(0, 2, 1)
        thermal_tokens = thermal_pooled.view(bs, c, -1).permute(0, 2, 1)

        # Bidirectional cross-attention on TOKENS
        # RGB -> Thermal attention (pre-norm)
        rgb_normed = self.rgb_norm1(rgb_tokens)      # (B, L, C)
        thermal_normed = self.thermal_norm1(thermal_tokens)  # (B, L, C)

        rgb_q = self.rgb_q(rgb_normed)               # (B, L, C)
        thermal_kv = self.thermal_kv(thermal_normed) # (B, L, 2C)
        thermal_k, thermal_v = torch.chunk(thermal_kv, 2, dim=-1)  # (B, L, C) each

        if return_attn:
            rgb_attended, attn_rgb_to_th = self._multi_head_attention(rgb_q, thermal_k, thermal_v, return_attn=True)
        else:
            rgb_attended = self._multi_head_attention(rgb_q, thermal_k, thermal_v, return_attn=False)

        rgb_attended = self.rgb_out(rgb_attended)    # (B, L, C)
        rgb_tokens = rgb_tokens + self.rgb_attn_scale * rgb_attended     # Residual connection

        # Thermal -> RGB attention
        thermal_q = self.thermal_q(thermal_normed)   # (B, L, C)
        rgb_kv = self.rgb_kv(rgb_normed)            # (B, L, 2C)
        rgb_k, rgb_v = torch.chunk(rgb_kv, 2, dim=-1)  # (B, L, C) each

        if return_attn:
            thermal_attended, attn_th_to_rgb = self._multi_head_attention(thermal_q, rgb_k, rgb_v, return_attn=True)
        else:
            thermal_attended = self._multi_head_attention(thermal_q, rgb_k, rgb_v, return_attn=False)

        thermal_attended = self.thermal_out(thermal_attended)  # (B, L, C)
        thermal_tokens = thermal_tokens + self.thermal_attn_scale * thermal_attended   # Residual connection
        '''print(f"🎯 ATTENTION DEBUG:")
        print(f"  RGB features: min={rgb_q.min():.3f}, max={rgb_q.max():.3f}, mean={rgb_q.mean():.3f}")
        print(f"  Thermal K: min={thermal_k.min():.3f}, max={thermal_k.max():.3f}, mean={thermal_k.mean():.3f}")
        print(f"  RGB attended: min={rgb_attended.min():.3f}, max={rgb_attended.max():.3f}, mean={rgb_attended.mean():.3f}")
        print(f"  Thermal attended: min={thermal_attended.min():.3f}, max={thermal_attended.max():.3f}, mean={thermal_attended.mean():.3f}")'''

        # Feed-forward networks on TOKENS (pre-norm)
        rgb_tokens = rgb_tokens + self.rgb_ffn(self.rgb_norm2(rgb_tokens))          # (B, L, C)
        thermal_tokens = thermal_tokens + self.thermal_ffn(self.thermal_norm2(thermal_tokens))  # (B, L, C)

        # Convert back to spatial format ONCE at the end
        rgb_spatial = rgb_tokens.transpose(1, 2).view(bs, c, self.vert_anchors, self.horz_anchors)  # (B, C, H, W)
        thermal_spatial = thermal_tokens.transpose(1, 2).view(bs, c, self.vert_anchors, self.horz_anchors)  # (B, C, H, W)

        rgb_final = F.interpolate(rgb_spatial, size=(h, w), mode='bilinear', align_corners=False)
        thermal_final = F.interpolate(thermal_spatial, size=(h, w), mode='bilinear', align_corners=False)
        
        '''fusion_w = F.softmax(self.fusion_weights, dim=0)
        cross_modal = rgb_final * thermal_final
        fused_final = (fusion_w[0] * rgb_final + 
                    fusion_w[1] * thermal_final + 
                    0.1 * cross_modal)'''
        fused_concat = torch.cat([rgb_final, thermal_final], dim=1) 
        if not hasattr(self, 'fusion_proj'):
            # Add this to __init__ instead:
            # self.fusion_proj = nn.Conv2d(d_model * 2, d_model, 1, bias=False)
            pass
            
        fused_final = self.fusion_proj(fused_concat)  # (B, C, H, W)

        if return_attn:
            attn_info = {
                'rgb_to_thermal': attn_rgb_to_th,   # (B, L, L)
                'thermal_to_rgb': attn_th_to_rgb    # (B, L, L)
            }
            return (rgb_final, thermal_final, fused_final), attn_info

        # Always return 3-tuple for consistency
        return rgb_final, thermal_final, fused_final

    
class BCAM_SingleOutput(BCAM):
    """BCAM variant that returns single fused tensor for clean comparison"""
    def __init__(self, d_model, output_mode='fused', **kwargs):
        super().__init__(d_model, **kwargs)
        self.output_mode = output_mode  # 'fused', 'rgb', 'thermal'
    
    def forward(self, x, **kwargs):
        rgb_final, thermal_final, fused_final = super().forward(x, **kwargs)
        
        output = {'fused': fused_final, 'rgb': rgb_final, 'thermal': thermal_final}[self.output_mode]
        
        assert isinstance(output, torch.Tensor) and not isinstance(output, tuple)
        return output
        
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (YOLOv5 & YOLOX)
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        
class UCAM(nn.Module):
    """
    Unidirectional Cross-Attention Module (UCAM)
    - Only RGB attends to Thermal (for ablation studies vs BCAM)
    - Same interface as BCAM (tuple in -> 3-tuple out)
    - Follows same tokenization → norm → proj → attention → FFN → un-tokenize pattern
    
    **Return Values**: Always returns 3-tuple `(rgb_final, thermal_final, fused_final)`
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, 
                 vert_anchors=8, horz_anchors=8):
        super(UCAM, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        assert d_model % num_heads == 0

        # Spatial handling - same as BCAM
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.avgpool = nn.AdaptiveAvgPool2d((vert_anchors, horz_anchors))

        # Internal pos embeddings (fallback)
        self.rgb_pos_embed = nn.Parameter(torch.randn(1, d_model, vert_anchors, horz_anchors) * 0.02)
        self.thermal_pos_embed = nn.Parameter(torch.randn(1, d_model, vert_anchors, horz_anchors) * 0.02)

        # Separate norms for RGB vs Thermal
        self.rgb_norm1 = nn.LayerNorm(d_model)
        self.thermal_norm1 = nn.LayerNorm(d_model)
        self.rgb_norm2 = nn.LayerNorm(d_model)
        self.thermal_norm2 = nn.LayerNorm(d_model)

        # RGB → Thermal projections only (unidirectional)
        self.rgb_q = nn.Linear(d_model, d_model, bias=False)
        self.thermal_kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.rgb_out = nn.Linear(d_model, d_model, bias=False)

        # Feed-forward for both modalities
        self.rgb_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.thermal_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _add_pos_encoding(self, features, pos_embed):
        """Same as BCAM - interpolate and add pos encoding"""
        if pos_embed is None:
            return features
        pos = pos_embed.to(device=features.device, dtype=features.dtype)
        if pos.shape[-2:] != features.shape[-2:]:
            pos = F.interpolate(pos, size=features.shape[-2:], mode='bilinear', align_corners=False)
        return features + pos

    def _multi_head_attention(self, query, key, value, return_attn=False):
        """Same multi-head attention as BCAM"""
        B, seq_len = query.shape[:2]
        Q = query.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = key.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = value.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)

        if return_attn:
            attn_avg = attn_weights.mean(dim=1)
            return attended, attn_avg
        return attended

    def forward(self, x, pos_rgb=None, pos_thermal=None, return_attn=False):
        """
        Unidirectional forward: only RGB attends to Thermal
        
        **Return Values**: Always returns 3-tuple `(rgb_final, thermal_final, fused_final)`
        When return_attn=True: returns ((rgb, thermal, fused), attn_info)
        """
        rgb_fea, thermal_fea = x[0], x[1]
        bs, c, h, w = rgb_fea.shape
        
        # Input validation
        assert rgb_fea.shape[1] == self.d_model, \
            f"RGB channels {rgb_fea.shape[1]} != d_model {self.d_model}"
        assert thermal_fea.shape[1] == self.d_model, \
            f"Thermal channels {thermal_fea.shape[1]} != d_model {self.d_model}"

        # Avgpool once, reuse
        rgb_pooled = self.avgpool(rgb_fea)
        thermal_pooled = self.avgpool(thermal_fea)

        # Positional encoding policy: external overrides internal
        use_rgb_pos = pos_rgb if pos_rgb is not None else getattr(self, 'rgb_pos_embed', None)
        use_thermal_pos = pos_thermal if pos_thermal is not None else getattr(self, 'thermal_pos_embed', None)

        if use_rgb_pos is not None:
            rgb_pooled = self._add_pos_encoding(rgb_pooled, use_rgb_pos)
        if use_thermal_pos is not None:
            thermal_pooled = self._add_pos_encoding(thermal_pooled, use_thermal_pos)

        # Convert to tokens: (B, C, H, W) -> (B, L, C)
        rgb_tokens = rgb_pooled.view(bs, c, -1).permute(0, 2, 1)
        thermal_tokens = thermal_pooled.view(bs, c, -1).permute(0, 2, 1)

        # === Unidirectional Cross-Attention: RGB attends to Thermal only ===
        rgb_normed = self.rgb_norm1(rgb_tokens)
        thermal_normed = self.thermal_norm1(thermal_tokens)

        rgb_q = self.rgb_q(rgb_normed)
        thermal_kv = self.thermal_kv(thermal_normed)
        thermal_k, thermal_v = torch.chunk(thermal_kv, 2, dim=-1)

        if return_attn:
            rgb_attended, attn_rgb_to_thermal = self._multi_head_attention(rgb_q, thermal_k, thermal_v, return_attn=True)
        else:
            rgb_attended = self._multi_head_attention(rgb_q, thermal_k, thermal_v, return_attn=False)

        rgb_attended = self.rgb_out(rgb_attended)
        rgb_tokens = rgb_tokens + rgb_attended

        # Feed-forward for both modalities (even though thermal didn't get attention)
        rgb_tokens = rgb_tokens + self.rgb_ffn(self.rgb_norm2(rgb_tokens))
        thermal_tokens = thermal_tokens + self.thermal_ffn(self.thermal_norm2(thermal_tokens))

        # Convert back to spatial
        rgb_spatial = rgb_tokens.transpose(1, 2).view(bs, c, self.vert_anchors, self.horz_anchors)
        thermal_spatial = thermal_tokens.transpose(1, 2).view(bs, c, self.vert_anchors, self.horz_anchors)

        rgb_final = F.interpolate(rgb_spatial, size=(h, w), mode='bilinear')
        thermal_final = F.interpolate(thermal_spatial, size=(h, w), mode='bilinear')
        
        # Always compute fused output
        fused_final = rgb_final + thermal_final

        if return_attn:
            attn_info = {
                'rgb_to_thermal': attn_rgb_to_thermal,   # (B, L, L)
                'thermal_to_rgb': None  # No thermal->RGB attention in UCAM
            }
            return (rgb_final, thermal_final, fused_final), attn_info

        return rgb_final, thermal_final, fused_final



    
class Adapter1x1(nn.Module):
    """Project upsampled coarse features to target channels"""
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)]
        if use_bn:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.proj = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.proj(x)

class Adapter1x1(nn.Module):
    """Project upsampled coarse features to target channels"""
    def __init__(self, in_ch, out_ch, use_bn=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)]
        if use_bn:
            # Safe GroupNorm with dynamic num_groups
            num_groups = min(32, out_ch) if out_ch >= 32 else out_ch
            # Ensure divisibility
            while out_ch % num_groups != 0:
                num_groups -= 1
            layers += [nn.GroupNorm(num_groups, out_ch), nn.ReLU(inplace=True)]
        self.proj = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.proj(x)


class FiLMModulation(nn.Module):
    """Produce per-channel scale and shift from coarse context"""
    def __init__(self, in_ch, channels, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or channels // 2
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, channels * 2, 1, bias=True)
        )
        
        # Zero-init BOTH convs for identity at start
        nn.init.constant_(self.net[0].bias, 0.0)
        nn.init.normal_(self.net[0].weight, 0.0, 0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)
        nn.init.normal_(self.net[-1].weight, 0.0, 0.01)
    
    def forward(self, coarse):
        pooled = self.pool(coarse)
        params = self.net(pooled)
        scale, shift = params.chunk(2, dim=1)
        return scale, shift


class BCAM_Progressive(BCAM):
    """Progressive BCAM with channel-wise FiLM and gated residuals"""
    def __init__(self, d_model, num_heads=4, dropout=0.1, 
                 vert_anchors=8, horz_anchors=8, coarse_channels=None):
        super().__init__(d_model, num_heads, dropout, vert_anchors, horz_anchors)
        
        if coarse_channels:
            self.adapter = Adapter1x1(coarse_channels, d_model, use_bn=False)
            self.rgb_film = FiLMModulation(d_model, d_model)
            self.thermal_film = FiLMModulation(d_model, d_model)
            # Single gated residual (not per-stream - simpler)
            self.g_ctx = nn.Parameter(torch.zeros(1))
        else:
            self.adapter = None
    
    def forward(self, x, coarse_context=None, pos_rgb=None, pos_thermal=None, return_attn=False):
        rgb_fea, thermal_fea = x[0], x[1]
        
        dev = next(self.parameters()).device
        dt = rgb_fea.dtype
        
        if rgb_fea.shape[1] != self.d_model or thermal_fea.shape[1] != self.d_model:
            raise RuntimeError(f"Input channels must equal d_model ({self.d_model})")
        if coarse_context is None:
            raise RuntimeError("BCAM_Progressive requires coarse_context")
        if self.adapter is None:
            raise RuntimeError("BCAM_Progressive not configured with adapter")
        
        # Coerce types
        coarse_context = coarse_context.to(dev, dt)
        rgb_fea = rgb_fea.to(dev, dt)
        thermal_fea = thermal_fea.to(dev, dt)
        
        if coarse_context.shape[1] != self.adapter.in_ch:
            raise RuntimeError(
                f"Adapter expects {self.adapter.in_ch} coarse channels, "
                f"got {coarse_context.shape[1]}"
            )
        
        # Upsample
        if coarse_context.shape[-2:] != rgb_fea.shape[-2:]:
            coarse_up = F.interpolate(coarse_context, size=rgb_fea.shape[-2:],
                                    mode='bilinear', align_corners=False)
        else:
            coarse_up = coarse_context
        
        # Project context
        ctx_proj = self.adapter(coarse_up).to(dt)
        
        # Gated residual injection
        gamma = torch.sigmoid(self.g_ctx)
        rgb_fea = rgb_fea + gamma * ctx_proj
        thermal_fea = thermal_fea + gamma * ctx_proj
        
        # Channel-wise FiLM
        rgb_scale, rgb_shift = self.rgb_film(ctx_proj)
        thermal_scale, thermal_shift = self.thermal_film(ctx_proj)
        
        # Bound scales
        rgb_scale = torch.tanh(rgb_scale)
        thermal_scale = torch.tanh(thermal_scale)
        
        # Apply FiLM
        rgb_fea = (1 + rgb_scale) * rgb_fea + rgb_shift
        thermal_fea = (1 + thermal_scale) * thermal_fea + thermal_shift
        
        # Call parent
        parent_out = super().forward((rgb_fea, thermal_fea), 
                                    pos_rgb=pos_rgb, pos_thermal=pos_thermal, 
                                    return_attn=return_attn)
        
        if return_attn:
            (rgb_out, thermal_out, fused_out), attn_info = parent_out
            return (rgb_out, thermal_out, fused_out), attn_info
        else:
            rgb_out, thermal_out, fused_out = parent_out
            return rgb_out, thermal_out, fused_out

class Progressive_SimpleAdd(nn.Module):
    """
    Progressive context injection with simple addition (no FiLM)
    For ablation testing against FiLM approach
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, 
                 vert_anchors=8, horz_anchors=8, coarse_channels=None):
        super().__init__()
        self.d_model = d_model
        
        # Project coarse context
        if coarse_channels:
            self.adapter = Adapter1x1(coarse_channels, d_model, use_bn=False)
            # Learnable scalar gate (same as BCAM_Progressive)
            self.g_ctx = nn.Parameter(torch.zeros(1))
        else:
            self.adapter = None
        
        # Standard BCAM for attention
        self.bcam = BCAM(d_model, num_heads, dropout, vert_anchors, horz_anchors)
    
    def forward(self, x, coarse_context=None, pos_rgb=None, pos_thermal=None, return_attn=False):
        rgb_fea, thermal_fea = x[0], x[1]
        
        # Device/dtype coercion
        dev = next(self.parameters()).device
        dt = rgb_fea.dtype
        
        if coarse_context is None:
            raise RuntimeError("Progressive_SimpleAdd requires coarse_context")
        if self.adapter is None:
            raise RuntimeError("Progressive_SimpleAdd not configured with adapter")
        
        # Coerce types
        coarse_context = coarse_context.to(dev, dt)
        rgb_fea = rgb_fea.to(dev, dt)
        thermal_fea = thermal_fea.to(dev, dt)
        
        # Check channels
        if coarse_context.shape[1] != self.adapter.in_ch:
            raise RuntimeError(
                f"Adapter expects {self.adapter.in_ch} coarse channels, "
                f"got {coarse_context.shape[1]}"
            )
        
        # Upsample coarse to fine resolution
        if coarse_context.shape[-2:] != rgb_fea.shape[-2:]:
            coarse_up = F.interpolate(coarse_context, size=rgb_fea.shape[-2:],
                                    mode='bilinear', align_corners=False)
        else:
            coarse_up = coarse_context
        
        # Project context
        ctx_proj = self.adapter(coarse_up).to(dt)
        
        # Simple gated addition (no FiLM modulation)
        gamma = torch.sigmoid(self.g_ctx)
        rgb_fea = rgb_fea + gamma * ctx_proj
        thermal_fea = thermal_fea + gamma * ctx_proj
        
        # Standard BCAM attention on context-enriched features
        return self.bcam((rgb_fea, thermal_fea), pos_rgb, pos_thermal, return_attn)

class Progressive_Projection(nn.Module):
    """
    Progressive context injection WITHOUT attention - for P3 scale
    Applies gated residual + FiLM modulation + projection, skips cross-attention
    
    Returns single fused tensor (not 3-tuple like BCAM)
    """
    def __init__(self, d_model, coarse_channels):
        super().__init__()
        self.d_model = d_model
        
        # Adapter to project coarse context
        self.adapter = Adapter1x1(coarse_channels, d_model, use_bn=False)
        
        # FiLM for each modality (channel-wise)
        self.rgb_film = FiLMModulation(d_model, d_model)
        self.thermal_film = FiLMModulation(d_model, d_model)
        
        # Gated residual (same as BCAM_Progressive)
        self.g_ctx = nn.Parameter(torch.zeros(1))
        
        # Simple fusion projection (no attention)
        self.fusion_proj = nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.fusion_proj.weight)
    
    def forward(self, x, coarse_context=None):
        """
        x: tuple of (rgb_features, thermal_features)
        coarse_context: context from P4
        
        Returns: single fused tensor (B, d_model, H, W)
        """
        rgb_fea, thermal_fea = x[0], x[1]
        
        # Device/dtype coercion
        dev = next(self.parameters()).device
        dt = rgb_fea.dtype
        
        if rgb_fea.shape[1] != self.d_model or thermal_fea.shape[1] != self.d_model:
            raise RuntimeError(f"Input channels must equal d_model ({self.d_model})")
        if coarse_context is None:
            raise RuntimeError("Progressive_Projection requires coarse_context")
        
        # Coerce types
        coarse_context = coarse_context.to(dev, dt)
        rgb_fea = rgb_fea.to(dev, dt)
        thermal_fea = thermal_fea.to(dev, dt)
        
        # Check channels
        if coarse_context.shape[1] != self.adapter.in_ch:
            raise RuntimeError(
                f"Adapter expects {self.adapter.in_ch} coarse channels, "
                f"got {coarse_context.shape[1]}"
            )
        
        # Upsample coarse to fine resolution
        if coarse_context.shape[-2:] != rgb_fea.shape[-2:]:
            coarse_up = F.interpolate(coarse_context, size=rgb_fea.shape[-2:],
                                    mode='bilinear', align_corners=False)
        else:
            coarse_up = coarse_context
        
        # Project context
        ctx_proj = self.adapter(coarse_up).to(dt)
        
        # Gated residual injection (same as BCAM_Progressive)
        gamma = torch.sigmoid(self.g_ctx)
        rgb_fea = rgb_fea + gamma * ctx_proj
        thermal_fea = thermal_fea + gamma * ctx_proj
        
        # Channel-wise FiLM modulation
        rgb_scale, rgb_shift = self.rgb_film(ctx_proj)
        thermal_scale, thermal_shift = self.thermal_film(ctx_proj)
        
        # Bound scales for stability
        rgb_scale = torch.tanh(rgb_scale)
        thermal_scale = torch.tanh(thermal_scale)
        
        # Apply FiLM
        rgb_modulated = (1 + rgb_scale) * rgb_fea + rgb_shift
        thermal_modulated = (1 + thermal_scale) * thermal_fea + thermal_shift
        
        # Concatenation + projection (NO attention)
        fused_concat = torch.cat([rgb_modulated, thermal_modulated], dim=1)
        fused = self.fusion_proj(fused_concat)
        
        return fused
    
class ScaleAdaptiveFusion(nn.Module):
    """
    Direct weighted fusion without cross-attention.
    Used at P3 where attention on context-injected features degrades performance.
    
    Args:
        channels: Number of input/output channels
        w_thermal: Weight for thermal modality (default 0.6 for P3)
        learnable: If True, weights are learnable parameters
    
    Returns:
        Single fused tensor (B, C, H, W)
    """
    def __init__(self, channels, w_thermal=0.6, learnable=False):
        super().__init__()
        self.channels = channels
        self.learnable = learnable
        
        if learnable:
            # Initialize as parameters
            self.w_thermal = nn.Parameter(torch.tensor(w_thermal))
            self.w_rgb = nn.Parameter(torch.tensor(1.0 - w_thermal))
        else:
            # Fixed weights
            self.register_buffer('w_thermal', torch.tensor(w_thermal))
            self.register_buffer('w_rgb', torch.tensor(1.0 - w_thermal))
    
    def forward(self, x):
        """
        x: tuple/list of (rgb_features, thermal_features)
           Each is (B, C, H, W)
        
        Returns: (B, C, H, W) weighted fusion
        """
        if isinstance(x, (tuple, list)) and len(x) == 2:
            rgb, thermal = x[0], x[1]
        else:
            raise ValueError(f"ScaleAdaptiveFusion expects tuple of 2 tensors, got {type(x)}")
        
        # Validate shapes
        assert rgb.shape == thermal.shape, \
            f"RGB and Thermal shapes must match: {rgb.shape} vs {thermal.shape}"
        assert rgb.shape[1] == self.channels, \
            f"Expected {self.channels} channels, got {rgb.shape[1]}"
        
        # Weighted fusion
        if self.learnable:
            # Normalize weights to sum to 1
            w_sum = self.w_rgb + self.w_thermal
            w_rgb_norm = self.w_rgb / w_sum
            w_thermal_norm = self.w_thermal / w_sum
            fused = w_rgb_norm * rgb + w_thermal_norm * thermal
        else:
            fused = self.w_rgb * rgb + self.w_thermal * thermal
        
        return fused


class BCAM_ScaleAdaptive(BCAM_SingleOutput):
    """
    BCAM with scale-dependent thermal weighting.
    Used at P5 and P4 where cross-attention is effective.
    
    Extends BCAM_SingleOutput to add weighted fusion after attention.
    
    Args:
        d_model: Feature dimension
        scale: 'P5', 'P4', or 'P3' (though P3 should use ScaleAdaptiveFusion)
        learnable_weights: If True, fusion weights are learnable
        **kwargs: Passed to parent BCAM
    """
    def __init__(self, d_model, scale='P5', learnable_weights=False, **kwargs):
        super().__init__(d_model, output_mode='fused', **kwargs)
        
        self.pyramid_scale = scale
        self.learnable_weights = learnable_weights
        
        # Scale-dependent thermal weights
        thermal_weight_map = {
            'P5': 0.4,  # Large objects - emphasize RGB texture
            'P4': 0.5,  # Medium objects - balanced
            'P3': 0.6   # Small objects - emphasize thermal shape (shouldn't use this class though)
        }
        
        if scale not in thermal_weight_map:
            raise ValueError(f"Scale must be P3/P4/P5, got {scale}")
        
        w_thermal_init = thermal_weight_map[scale]
        
        if learnable_weights:
            self.w_thermal = nn.Parameter(torch.tensor(w_thermal_init))
            self.w_rgb = nn.Parameter(torch.tensor(1.0 - w_thermal_init))
        else:
            self.register_buffer('w_thermal', torch.tensor(w_thermal_init))
            self.register_buffer('w_rgb', torch.tensor(1.0 - w_thermal_init))
    
    def forward(self, x, **kwargs):
        """
        Forward pass with scale-adaptive weighting applied to BCAM output.
        
        BCAM_SingleOutput returns single fused tensor.
        We re-weight the internal rgb/thermal outputs before final fusion.
        """
        # Get parent class output - but we need to intercept before final fusion
        # So we call BCAM (grandparent) instead to get 3-tuple
        rgb_final, thermal_final, _ = BCAM.forward(self, x, **kwargs)
        
        # Apply scale-adaptive weighting
        if self.learnable_weights:
            w_sum = self.w_rgb + self.w_thermal
            w_rgb_norm = self.w_rgb / w_sum
            w_thermal_norm = self.w_thermal / w_sum
            weighted_fused = w_rgb_norm * rgb_final + w_thermal_norm * thermal_final
        else:
            weighted_fused = self.w_rgb * rgb_final + self.w_thermal * thermal_final
        
        return weighted_fused