import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels,  bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: [B,C,1,1]
        b, c, _, _ = x.size()
        v = x.view(b, c)                   # [B,C]
        w = self.fc(v).view(b, c, 1, 1)    # [B,C,1,1]
        return x * w                       # channel re‐weight


class BottleneckDW(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4):
        super().__init__()
        mid_ch = out_ch // expansion
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.downsample = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride!=1 or in_ch!=out_ch) else None)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)

class DepthwiseSeparable(nn.Module):
    """
    A single depthwise‑separable block: 3×3 depthwise → 1×1 pointwise.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x



class CNN(nn.Module):
    def __init__(self, num_classes=2, use_bottleneck=True, use_se=True):
        super().__init__()
        block = BottleneckDW if use_bottleneck else DepthwiseSeparable  # or swap in your separable

        # stem + 4 strided blocks: output goes [B,512,8,8]
        self.layer0 = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = block(32,  64, stride=2)
        self.layer2 = block(64, 128, stride=2)
        self.layer3 = block(128,256, stride=2)
        self.layer4 = block(256,512, stride=2)

        # global pool to [B,512,1,1]
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.se   = SEBlock(512) if use_se else nn.Identity()

        # *big* MLP on the pooled 512-dim vector
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # [B,512]
            nn.Linear(512, 2048, bias=False),  # +~1M params
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024, bias=False), # +~2M params
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)       # + small
        )

    def forward(self, x):
        x = self.layer0(x)   # 256→128
        x = self.layer1(x)   # 128→64
        x = self.layer2(x)   # 64→32
        x = self.layer3(x)   # 32→16
        x = self.layer4(x)   # 16→8
        x = self.pool(x)     # → [B,512,1,1]
        x = self.se(x)       # optional SE
        return self.classifier(x)
    