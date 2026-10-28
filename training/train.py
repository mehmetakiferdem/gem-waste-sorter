#!/usr/bin/env python3
"""Fine-tunes MobileNetV2 (ImageNet) on dataset/ and exports ONNX for TIDL.

    python3 train.py --data dataset --out ../models/waste_mnv2 --epochs 20

MobileNetV2 is chosen because every layer of it runs on the C7x-MMA under
TIDL; normalisation is folded into the exported graph so the board feeds raw
RGB 0..255.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class ChannelGain:
    """Scales R, G, B independently: a white balance the ISP got wrong."""

    def __init__(self, low=0.75, high=1.3):
        self.low, self.high = low, high

    def __call__(self, x):
        g = torch.empty(3, 1, 1).uniform_(self.low, self.high)
        return (x * g).clamp(0, 1)


class Deployed(nn.Module):
    """RGB 0..255 NCHW in, logits out."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def loaders(root, size, batch, workers):
    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([
        # Tighter crops too: live, the classifier sees the detector's box.
        transforms.RandomResizedCrop(size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        # The IMX219 + ISP colours differ from the dataset cameras: the live
        # picture came out pink in daylight, and items are held, so blurred.
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.08),
        transforms.RandomGrayscale(0.05),
        transforms.RandomApply([transforms.GaussianBlur(7, (0.1, 2.5))], p=0.3),
        transforms.ToTensor(),
        transforms.RandomApply([ChannelGain()], p=0.5),
        norm])
    eval_tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    tr = datasets.ImageFolder(os.path.join(root, "train"), train_tf)
    va = datasets.ImageFolder(os.path.join(root, "val"), eval_tf)
    # Classes are unbalanced (lots of plastic, little glass): sample evenly.
    counts = torch.bincount(torch.tensor(tr.targets))
    w = (1.0 / counts.float())[tr.targets]
    sampler = WeightedRandomSampler(w, len(tr), replacement=True)
    return (DataLoader(tr, batch, sampler=sampler, num_workers=workers, pin_memory=True),
            DataLoader(va, batch, num_workers=workers), tr.classes)


def evaluate(net, dl, dev, n_cls):
    net.eval()
    conf = torch.zeros(n_cls, n_cls, dtype=torch.long)
    with torch.no_grad():
        for x, y in dl:
            p = net(x.to(dev)).argmax(1).cpu()
            for t, q in zip(y, p):
                conf[t, q] += 1
    return conf.diag().sum().item() / conf.sum().item(), conf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="../models/waste_mnv2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, classes = loaders(args.data, args.size, args.batch, args.workers)
    print(f"cihaz={dev} sınıflar={classes} train={len(tr.dataset)} val={len(va.dataset)}")

    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    net.classifier[1] = nn.Linear(net.last_channel, len(classes))
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs, steps_per_epoch=len(tr))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(args.out, exist_ok=True)
    best = 0.0
    for ep in range(args.epochs):
        net.train()
        t0, total = time.time(), 0.0
        for x, y in tr:
            opt.zero_grad()
            loss = loss_fn(net(x.to(dev)), y.to(dev))
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        acc, conf = evaluate(net, va, dev, len(classes))
        print(f"epoch {ep + 1:2d}/{args.epochs} loss={total / len(tr):.3f} val_acc={acc:.3f} "
              f"({time.time() - t0:.0f} s)")
        if acc > best:
            best = acc
            torch.save(net.state_dict(), os.path.join(args.out, "best.pt"))

    net.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=dev))
    acc, conf = evaluate(net, va, dev, len(classes))
    print(f"\nen iyi val_acc={acc:.3f}\nkarışıklık matrisi (satır=gerçek):")
    print("         " + " ".join(f"{c[:6]:>6s}" for c in classes))
    for c, row in zip(classes, conf):
        print(f"{c:8s} " + " ".join(f"{v:6d}" for v in row.tolist()))

    deployed = Deployed(net.cpu()).eval()
    onnx_path = os.path.join(args.out, "model.onnx")
    # Opset 11 and a fixed batch of 1: what TIDL's ONNX import handles best.
    torch.onnx.export(deployed, torch.zeros(1, 3, args.size, args.size), onnx_path,
                      input_names=["input"], output_names=["logits"], opset_version=11,
                      dynamo=False)
    with open(os.path.join(args.out, "labels.txt"), "w") as f:
        f.write("\n".join(classes) + "\n")
    print(f"\n{onnx_path} ve labels.txt yazıldı. Sonraki adım: compile_tidl.py")


if __name__ == "__main__":
    main()
