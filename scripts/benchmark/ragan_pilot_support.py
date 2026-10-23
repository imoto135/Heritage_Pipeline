"""Local RA-GAN adapter; keep the author's checkout unchanged.

The functional differential kernel preserves the author's detached derivation.
The otherwise unused diff_conv.weight stays in state_dict for compatibility,
but is frozen before optimizer construction.
"""
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'tmp_work/ra_gan_reference_20261021'


def reference_classes():
    for p in (REFERENCE / 'dependencies', REFERENCE / 'source'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from models.Generator import RA_GAN
    from models.Conv import GGC
    from models.Discriminator import Discriminator
    return RA_GAN, GGC, Discriminator


def stable_ggc_forward(self, image):
    normal = self.conv(image)
    theta = self.theta_net(image).mean()
    if abs(float(theta.detach())) < 1e-8:
        combined = normal
    else:
        # Preserve stop-gradient semantics in the author's torch.no_grad block.
        kernel = self.conv.weight.detach().sum(2, keepdim=True).sum(3, keepdim=True)
        difference = F.conv2d(image, kernel, self.diff_conv.bias,
                              stride=self.diff_conv.stride,
                              padding=self.diff_conv.padding,
                              dilation=self.diff_conv.dilation,
                              groups=self.diff_conv.groups)
        combined = normal - theta * difference
    value, gate = torch.chunk(combined, 2, 1)
    return self.activation(value) * torch.sigmoid(gate)


def stabilize_ggc(model):
    _, GGC, _ = reference_classes()
    names = []
    for name, module in model.named_modules():
        if isinstance(module, GGC):
            module.diff_conv.weight.requires_grad_(False)
            module.forward = types.MethodType(stable_ggc_forward, module)
            names.append(name)
    return names


def auxiliaries(rgb):
    """Recompute after the native crop; never use clean or target pixels."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.rint(np.clip(cv2.magnitude(dx, dy), 0, 255)).astype(np.uint8)
    return np.concatenate((rgb, otsu[..., None], sobel[..., None]), axis=2)


class Block(nn.Sequential):
    def __init__(self, ci, co):
        super().__init__(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.ReLU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.ReLU())


class PilotUNet(nn.Module):
    """Five-channel baseline, native pixels; paper probability at output."""
    def __init__(self, base=24):
        super().__init__()
        self.encoders = nn.ModuleList([Block(5, base)] +
            [Block(base * 2**i, base * 2**(i+1)) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(base * 2**(i+1), base * 2**i, 2, 2)
                                for i in reversed(range(4))])
        self.decoders = nn.ModuleList([Block(base * 2**(i+1), base * 2**i)
                                      for i in reversed(range(4))])
        self.output = nn.Conv2d(base, 1, 1)

    def forward(self, image):
        skips = []
        for i, encoder in enumerate(self.encoders):
            image = encoder(image)
            if i < 4:
                skips.append(image)
                image = F.max_pool2d(image, 2)
        for up, decoder, skip in zip(self.up, self.decoders, reversed(skips)):
            image = decoder(torch.cat((up(image), skip), dim=1))
        return torch.sigmoid(self.output(image))


def make_model(kind):
    if kind == 'unet':
        return PilotUNet()
    RA_GAN, _, _ = reference_classes()
    model = RA_GAN(5)
    assert len(stabilize_ggc(model)) == 4
    return model


def predict(model, kind, image):
    return model(image) if kind == 'unet' else model(image[:, :3], image[:, 3:4], image[:, 4:5])


def supervised_loss(paper_probability, paper_target, valid):
    """Partial-label loss. The ink overlap terms differ from author paper terms.

    All compared neural models share exactly this loss. Unknown target values
    have zero influence, including in denominators.
    """
    p = paper_probability.float().clamp(1e-6, 1-1e-6)
    t = paper_target.float()
    n = valid.sum().clamp_min(1)
    l1 = ((p-t).abs() * valid).sum() / n
    bce = (F.binary_cross_entropy(p, t, reduction='none') * valid).sum() / n
    ink = (1-p) * valid
    truth = (1-t) * valid
    intersection = (ink * truth).sum()
    dice = 1 - (2*intersection + 1e-6) / (ink.sum() + truth.sum() + 1e-6)
    jaccard = 1 - (intersection + 1e-6) / (ink.sum() + truth.sum() - intersection + 1e-6)
    return 10*l1 + bce + dice + jaccard


def critic_input(paper_probability, valid):
    # The critic sees the same neutral unknown pixels for real and fake labels.
    return paper_probability * valid + .5 * (1-valid)
