#!/usr/bin/env python3
import argparse
from cv_core import TrainConfig, run_cv

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/banana")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-pct", type=float, default=0.1)
    args = p.parse_args()

    cfg = TrainConfig(
        data=args.data, out=args.out, variant="bam",
        epochs=args.epochs, batch_size=args.batch_size,
        img_size=args.img_size, lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed, val_pct=args.val_pct
    )
    print(run_cv(cfg))

if __name__ == "__main__":
    main()
