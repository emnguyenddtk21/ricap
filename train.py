import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from tqdm.auto import tqdm

from utils import AverageMeter, accuracy, count_params, seed_everything
from wide_resnet import WideResNet


CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

DATASET_CONFIG = {
    "cifar10": {"dataset": datasets.CIFAR10, "num_classes": 10},
    "cifar100": {"dataset": datasets.CIFAR100, "num_classes": 100},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train WideResNet with RICAP/Mixup on CIFAR.")

    parser.add_argument("--name", default=None, help="Experiment name. Defaults to an auto-generated name.")
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASET_CONFIG.keys()), help="Dataset name.")
    parser.add_argument("--data-dir", default="./data", help="Directory used to store torchvision datasets.")
    parser.add_argument("--output-dir", default="./models", help="Directory used to store logs and checkpoints.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    parser.add_argument("--depth", default=28, type=int, help="WideResNet depth.")
    parser.add_argument("--width", default=10, type=int, help="WideResNet widen factor.")
    parser.add_argument("--dropout", default=0.3, type=float, help="WideResNet dropout ratio.")

    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--milestones", default="60,120,160", type=str)
    parser.add_argument("--gamma", default=0.2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num-workers", default=4, type=int, help="Number of DataLoader workers.")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable mixed precision on CUDA.")

    parser.add_argument("--ricap", action=argparse.BooleanOptionalAction, default=False, help="Use RICAP.")
    parser.add_argument("--ricap-beta", default=0.3, type=float, help="Beta parameter of RICAP.")
    parser.add_argument("--random-erase", action=argparse.BooleanOptionalAction, default=False, help="Use Random Erasing.")
    parser.add_argument("--random-erase-prob", default=0.5, type=float)
    parser.add_argument("--random-erase-sl", default=0.02, type=float)
    parser.add_argument("--random-erase-sh", default=0.4, type=float)
    parser.add_argument("--random-erase-r", default=0.3, type=float)
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, default=False, help="Use Mixup.")
    parser.add_argument("--mixup-alpha", default=1.0, type=float)

    args = parser.parse_args()
    enabled_augments = sum([args.ricap, args.mixup, args.random_erase])
    if enabled_augments > 1:
        raise ValueError("Enable at most one of --ricap, --mixup, or --random-erase at a time.")

    if args.ricap_beta <= 0:
        raise ValueError("--ricap-beta must be > 0.")
    if args.mixup and args.mixup_alpha <= 0:
        raise ValueError("--mixup-alpha must be > 0 when mixup is enabled.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")

    return args


def build_experiment_name(args):
    name = f"{args.dataset}_WideResNet{args.depth}-{args.width}"
    if args.ricap:
        name += "_wRICAP"
    if args.random_erase:
        name += "_wRandomErasing"
    if args.mixup:
        name += "_wMixup"
    return name


def build_transforms(args):
    mean, std = CIFAR_STATS[args.dataset]
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if args.random_erase:
        train_transforms.append(
            transforms.RandomErasing(
                p=args.random_erase_prob,
                scale=(args.random_erase_sl, args.random_erase_sh),
                ratio=(args.random_erase_r, 1.0 / args.random_erase_r),
                value="random",
            )
        )
    train_transforms.append(transforms.Normalize(mean, std))

    test_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def build_dataloaders(args):
    dataset_cls = DATASET_CONFIG[args.dataset]["dataset"]
    train_transform, test_transform = build_transforms(args)
    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    train_set = dataset_cls(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = dataset_cls(root=data_dir, train=False, download=True, transform=test_transform)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def apply_ricap(inputs, targets, beta, device):
    input_height, input_width = inputs.size(2), inputs.size(3)
    w = int(np.round(input_width * np.random.beta(beta, beta)))
    h = int(np.round(input_height * np.random.beta(beta, beta)))

    widths = [w, input_width - w, w, input_width - w]
    heights = [h, h, input_height - h, input_height - h]

    cropped_images = []
    target_slices = []
    area_weights = []

    for k in range(4):
        indices = torch.randperm(inputs.size(0), device=inputs.device)
        x_k = np.random.randint(0, input_width - widths[k] + 1)
        y_k = np.random.randint(0, input_height - heights[k] + 1)
        cropped = inputs[indices][:, :, y_k : y_k + heights[k], x_k : x_k + widths[k]]
        cropped_images.append(cropped)
        target_slices.append(targets[indices].to(device, non_blocking=True))
        area_weights.append(widths[k] * heights[k] / (input_width * input_height))

    patched_images = torch.cat(
        (
            torch.cat((cropped_images[0], cropped_images[1]), dim=3),
            torch.cat((cropped_images[2], cropped_images[3]), dim=3),
        ),
        dim=2,
    ).to(device, non_blocking=True)

    return patched_images, target_slices, area_weights


def apply_mixup(inputs, targets, alpha, device):
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(inputs.size(0), device=inputs.device)

    input_a, input_b = inputs, inputs[indices]
    target_a = targets.to(device, non_blocking=True)
    target_b = targets[indices].to(device, non_blocking=True)
    mixed_inputs = lam * input_a + (1.0 - lam) * input_b

    return mixed_inputs.to(device, non_blocking=True), target_a, target_b, lam


def train_one_epoch(args, train_loader, model, criterion, optimizer, scaler, device):
    losses = AverageMeter()
    scores = AverageMeter()
    model.train()

    progress = tqdm(train_loader, desc="Train", leave=False)
    for inputs, targets in progress:
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=scaler.is_enabled()):
            if args.ricap:
                patched_inputs, ricap_targets, area_weights = apply_ricap(inputs, targets, args.ricap_beta, device)
                outputs = model(patched_inputs)
                loss = sum(weight * criterion(outputs, target) for weight, target in zip(area_weights, ricap_targets))
                acc = sum(weight * accuracy(outputs, target)[0] for weight, target in zip(area_weights, ricap_targets))
                batch_size = patched_inputs.size(0)
            elif args.mixup:
                mixed_inputs, target_a, target_b, lam = apply_mixup(inputs, targets, args.mixup_alpha, device)
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, target_a) + (1.0 - lam) * criterion(outputs, target_b)
                acc = lam * accuracy(outputs, target_a)[0] + (1.0 - lam) * accuracy(outputs, target_b)[0]
                batch_size = mixed_inputs.size(0)
            else:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc = accuracy(outputs, targets)[0]
                batch_size = inputs.size(0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), batch_size)
        scores.update(acc.item(), batch_size)
        progress.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{scores.avg:.2f}")

    return OrderedDict(loss=losses.avg, acc=scores.avg)


def validate(val_loader, model, criterion, device):
    losses = AverageMeter()
    scores = AverageMeter()
    model.eval()

    with torch.no_grad():
        progress = tqdm(val_loader, desc="Eval", leave=False)
        for inputs, targets in progress:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc1 = accuracy(outputs, targets, topk=(1,))[0]

            losses.update(loss.item(), inputs.size(0))
            scores.update(acc1.item(), inputs.size(0))
            progress.set_postfix(loss=f"{losses.avg:.4f}", acc=f"{scores.avg:.2f}")

    return OrderedDict(loss=losses.avg, acc=scores.avg)


def save_checkpoint(experiment_dir, model, optimizer, scheduler, args, epoch, best_acc, is_best):
    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }
    latest_path = experiment_dir / "checkpoint_latest.pth"
    torch.save(checkpoint, latest_path)
    if is_best:
        torch.save(checkpoint, experiment_dir / "checkpoint_best.pth")
        torch.save(model.state_dict(), experiment_dir / "model_best_state_dict.pth")


def main():
    args = parse_args()
    args.name = args.name or build_experiment_name(args)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = device.type == "cuda"

    experiment_dir = Path(args.output_dir) / args.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    args_dict = vars(args).copy()
    args_dict["device"] = device.type

    print("Config -----")
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    print("------------")

    (experiment_dir / "args.json").write_text(json.dumps(args_dict, indent=2), encoding="utf-8")

    train_loader, test_loader = build_dataloaders(args)
    num_classes = DATASET_CONFIG[args.dataset]["num_classes"]

    model = WideResNet(args.depth, args.width, num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"Trainable params: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(epoch) for epoch in args.milestones.split(",") if epoch],
        gamma=args.gamma,
    )
    scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    logs = []
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch}/{args.epochs}] - lr: {current_lr:.6f}", flush=True)

        train_log = train_one_epoch(args, train_loader, model, criterion, optimizer, scaler, device)
        val_log = validate(test_loader, model, criterion, device)
        scheduler.step()

        print(
            f"loss {train_log['loss']:.4f} - acc {train_log['acc']:.2f} - "
            f"val_loss {val_log['loss']:.4f} - val_acc {val_log['acc']:.2f}"
        , flush=True)

        logs.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "loss": train_log["loss"],
                "acc": train_log["acc"],
                "val_loss": val_log["loss"],
                "val_acc": val_log["acc"],
            }
        )
        pd.DataFrame(logs).to_csv(experiment_dir / "log.csv", index=False)

        is_best = val_log["acc"] > best_acc
        if is_best:
            best_acc = val_log["acc"]
            print("=> saved best model", flush=True)

        save_checkpoint(experiment_dir, model, optimizer, scheduler, args, epoch, best_acc, is_best)


if __name__ == "__main__":
    main()
