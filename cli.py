from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import torch

from csihar.core import (
    Trainer,
    build_loaders_from_manifests,
    build_model_from_config,
    config_fingerprint,
    evaluate_split,
    get_env_info,
    load_checkpoint,
    make_run_dir,
    parse_kv_overrides,
    resolve_config,
    save_resolved_config,
    set_seed,
    setup_logger,
)


def _common_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--out-dir", default=None, help="Override output.out_dir")
    ap.add_argument("--device", default=None, help="Override trainer.device")
    ap.add_argument("--seed", type=int, default=None, help="Override trainer.seed")
    ap.add_argument("--set", action="append", default=[], help="Override config with key=value (supports JSON)")
    return ap


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)

    # --set key=value
    kv_over = parse_kv_overrides(args.set)
    if kv_over:
        # deep merge (internal utility)
        from csihar.core import _deep_update
        _deep_update(cfg, kv_over)

    if args.out_dir is not None:
        cfg.setdefault("output", {})
        cfg["output"]["out_dir"] = str(args.out_dir)
    if args.device is not None:
        cfg.setdefault("trainer", {})
        cfg["trainer"]["device"] = str(args.device)
    if args.seed is not None:
        cfg.setdefault("trainer", {})
        cfg["trainer"]["seed"] = int(args.seed)

    return cfg


def train_main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(parents=[_common_parser()])
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    args = ap.parse_args(argv)

    cfg = resolve_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    out_dir = str(cfg.get("output", {}).get("out_dir", "./runs"))
    exp_name = str(cfg.get("exp_name", "exp"))
    run_dir = make_run_dir(out_dir, exp_name)
    logger = setup_logger(run_dir)

    seed = int(cfg.get("trainer", {}).get("seed", 42))
    set_seed(seed, deterministic=bool(cfg.get("trainer", {}).get("deterministic", False)))
    save_resolved_config(cfg, run_dir)

    logger.info(f"[env] {get_env_info()}")
    logger.info(f"[cfg] fingerprint={config_fingerprint(cfg)}")
    logger.info(f"[cfg] resolved saved: {os.path.join(run_dir, 'config.resolved.yaml')}")

    dev_str = str(cfg.get("trainer", {}).get("device", "cuda"))
    device = torch.device(dev_str if torch.cuda.is_available() else "cpu")
    logger.info(f"[device] {device}")

    train_loader, val_loader, test_loader, remap, counts = build_loaders_from_manifests(
        cfg, args.train_manifest, args.val_manifest, args.test_manifest
    )
    n_classes = len(remap)
    logger.info(f"[data] n_classes={n_classes} | counts={counts}")

    class_weights = None
    if bool(cfg.get("loss", {}).get("ce", {}).get("class_balanced", False)):
        tot = sum(counts.values())
        w = []
        for c in range(n_classes):
            w.append(tot / (n_classes * max(1, counts.get(c, 1))))
        class_weights = torch.tensor(w, dtype=torch.float32)

    model = build_model_from_config(cfg["model"], n_classes=n_classes)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        n_classes=n_classes,
        device=device,
        run_dir=run_dir,
        logger=logger,
        class_weights=class_weights,
    )
    trainer.fit(train_loader, val_loader, test_loader)


def eval_main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(parents=[_common_parser()])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-manifest", required=True)
    args = ap.parse_args(argv)

    cfg = resolve_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    out_dir = str(cfg.get("output", {}).get("out_dir", "./runs"))
    exp_name = str(cfg.get("exp_name", "exp"))
    run_dir = make_run_dir(out_dir, exp_name)
    logger = setup_logger(run_dir)

    seed = int(cfg.get("trainer", {}).get("seed", 42))
    set_seed(seed, deterministic=bool(cfg.get("trainer", {}).get("deterministic", False)))
    save_resolved_config(cfg, run_dir)

    dev_str = str(cfg.get("trainer", {}).get("device", "cuda"))
    device = torch.device(dev_str if torch.cuda.is_available() else "cpu")

    # Eval-only mode: use the test manifest as all splits (for remap consistency)
    train_loader, val_loader, test_loader, remap, _ = build_loaders_from_manifests(
        cfg, args.test_manifest, args.test_manifest, args.test_manifest
    )
    n_classes = len(remap)

    model = build_model_from_config(cfg["model"], n_classes=n_classes).to(device)
    ckpt = load_checkpoint(args.ckpt, map_location=str(device))
    model.load_state_dict(ckpt["model"], strict=False)

    res = evaluate_split(model, test_loader, device, n_classes, return_confusion=True)
    logger.info(f"[EVAL] top1={float(res.get('top1', 0.0)):.4f} | total={int(res.get('total', 0))}")

    import numpy as np
    conf = res.get("confusion", None)
    if conf is not None:
        path = os.path.join(run_dir, "confusion.npy")
        np.save(path, conf)
        logger.info(f"[EVAL] confusion saved: {path}")


def cache_main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(parents=[_common_parser()])
    ap.add_argument("--train-manifest", required=True)
    ap.add_argument("--val-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    args = ap.parse_args(argv)

    cfg = resolve_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    out_dir = str(cfg.get("output", {}).get("out_dir", "./runs"))
    exp_name = str(cfg.get("exp_name", "exp"))
    run_dir = make_run_dir(out_dir, exp_name)
    logger = setup_logger(run_dir)

    seed = int(cfg.get("trainer", {}).get("seed", 42))
    set_seed(seed, deterministic=bool(cfg.get("trainer", {}).get("deterministic", False)))
    save_resolved_config(cfg, run_dir)

    train_loader, val_loader, test_loader, _, _ = build_loaders_from_manifests(
        cfg, args.train_manifest, args.val_manifest, args.test_manifest
    )

    logger.info("[cache] building preprocessing cache by iterating loaders...")
    for _ in train_loader:
        pass
    for _ in val_loader:
        pass
    for _ in test_loader:
        pass
    logger.info("[cache] done.")


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add_common(p: argparse.ArgumentParser):
        p.add_argument("--config", required=True)
        p.add_argument("--out-dir", default=None)
        p.add_argument("--device", default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--set", action="append", default=[])

    p_tr = sub.add_parser("train")
    _add_common(p_tr)
    p_tr.add_argument("--train-manifest", required=True)
    p_tr.add_argument("--val-manifest", required=True)
    p_tr.add_argument("--test-manifest", required=True)

    p_ev = sub.add_parser("eval")
    _add_common(p_ev)
    p_ev.add_argument("--ckpt", required=True)
    p_ev.add_argument("--test-manifest", required=True)

    p_ca = sub.add_parser("cache")
    _add_common(p_ca)
    p_ca.add_argument("--train-manifest", required=True)
    p_ca.add_argument("--val-manifest", required=True)
    p_ca.add_argument("--test-manifest", required=True)

    args = ap.parse_args(argv)

    def _pack_sets(ss):
        out = []
        for s in ss:
            out += ["--set", s]
        return out

    if args.cmd == "train":
        train_main(
            argv=[
                f"--config={args.config}",
                f"--train-manifest={args.train_manifest}",
                f"--val-manifest={args.val_manifest}",
                f"--test-manifest={args.test_manifest}",
            ]
            + ([f"--out-dir={args.out_dir}"] if args.out_dir else [])
            + ([f"--device={args.device}"] if args.device else [])
            + ([f"--seed={args.seed}"] if args.seed is not None else [])
            + _pack_sets(args.set)
        )
    elif args.cmd == "eval":
        eval_main(
            argv=[
                f"--config={args.config}",
                f"--ckpt={args.ckpt}",
                f"--test-manifest={args.test_manifest}",
            ]
            + ([f"--out-dir={args.out_dir}"] if args.out_dir else [])
            + ([f"--device={args.device}"] if args.device else [])
            + ([f"--seed={args.seed}"] if args.seed is not None else [])
            + _pack_sets(args.set)
        )
    else:
        cache_main(
            argv=[
                f"--config={args.config}",
                f"--train-manifest={args.train_manifest}",
                f"--val-manifest={args.val_manifest}",
                f"--test-manifest={args.test_manifest}",
            ]
            + ([f"--out-dir={args.out_dir}"] if args.out_dir else [])
            + ([f"--device={args.device}"] if args.device else [])
            + ([f"--seed={args.seed}"] if args.seed is not None else [])
            + _pack_sets(args.set)
        )


if __name__ == "__main__":
    main()
