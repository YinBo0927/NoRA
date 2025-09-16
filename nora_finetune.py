import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from replace_activation import replace_activation, replace_activation_with_nora
from nora import NoRA
from rational_group import Rational_Group1d, Rational_Group2d

def only_nora_params(model):
    for m in model.modules():
        if isinstance(m, NoRA):
            for p in m.parameters():
                if p.requires_grad:
                    yield p

def freeze_all_but_nora(model, unfreeze_bias_and_norm=True):
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_bias_and_norm:
        norm_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)
        for m in model.modules():
            if isinstance(m, norm_types):
                for p in m.parameters(recurse=False):
                    p.requires_grad = True
            if hasattr(m, "bias") and isinstance(m.bias, torch.Tensor):
                m.bias.requires_grad = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--preset", type=str, default="gelu")
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--save", type=str, default="./nora_vit_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 transforms
    tf_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616]),
    ])
    tf_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2470,0.2435,0.2616]),
    ])
    train_ds = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tf_train)
    test_ds  = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load ViT
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, 10)  # CIFAR-10

    # Replace GELU with NoRA-LoRA
    replace_activation_with_nora(
        model,
        target_types=(Rational_Group1d,),

    )

    # Freeze backbone
    freeze_all_but_nora(model, unfreeze_bias_and_norm=True)
    model.to(device)

    opt = torch.optim.AdamW(list(only_nora_params(model)), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best = 0.0

    @torch.no_grad()
    def evaluate():
        model.eval(); tot=0; correct=0; loss_sum=0.0
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits, y, reduction="sum")
            loss_sum += loss.item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            tot += y.size(0)
        return loss_sum/tot, correct/tot

    for ep in range(args.epochs):
        model.train(); run_loss=0.0
        for i,(x,y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_loss += loss.item()
            if (i+1) % 20 == 0:
                print(f"ep {ep+1}/{args.epochs} step {i+1}/{len(train_loader)} loss={run_loss/(i+1):.4f}")

        val_loss, val_acc = evaluate()
        print(f"[Val] ep {ep+1}: loss={val_loss:.4f} acc={val_acc*100:.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "acc": best}, args.save)
            print(f"Saved best -> {args.save} (acc={best*100:.2f}%).")

    print(f"Done. Best acc: {best*100:.2f}%")

if __name__ == "__main__":
    main()

