#!/usr/bin/env python3
"""REINFORCE policy-gradient post-training for the eFrog frog-call classifier.

This is a TRUE reinforcement-learning fine-tuning loop (not supervised
cross-entropy). The pre-trained ONNX classifier is loaded into a trainable
torch.nn.Module via `onnx2torch` and treated as a stochastic policy:

    policy   pi_theta(a | s) = softmax(classifier_logits(s))
    state    s = a user observation's log-mel spectrogram, shape [1, 1, 64, 157]
    action   a = a class index sampled from Categorical(pi_theta(.|s))
    reward   r = +1 if a == ground-truth label (species_name), else -1
    objective  maximise E[r]  via  REINFORCE:
               loss = -(r - b) * log pi_theta(a|s) - beta * H[pi_theta(.|s)]
               where b is a running-mean baseline (variance reduction) and the
               entropy term H encourages exploration.

After training, the updated policy network is exported back to ONNX with the
same input/output names and [1,1,64,157] -> [1,19] signature.

Run `python train_rl.py --smoke` to exercise the whole pipeline end-to-end on
fabricated data, with no real model or observations required.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local contract (works whether run as a module or a script from this folder).
try:
    from labels import (
        LABEL_CLASSES,
        LABEL_TO_IDX,
        MEL_LENGTH,
        N_CLASSES,
        N_FRAMES,
        N_MELS,
    )
except ImportError:  # pragma: no cover - executed only when imported as package
    from .labels import (  # type: ignore
        LABEL_CLASSES,
        LABEL_TO_IDX,
        MEL_LENGTH,
        N_CLASSES,
        N_FRAMES,
        N_MELS,
    )

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "frog_classifier_rl.onnx"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def decode_mel(b64: str) -> np.ndarray | None:
    """Decode a base64 little-endian float32 mel spectrogram into [64, 157].

    Returns None if the payload is missing or the wrong size.
    """
    if not isinstance(b64, str) or not b64.strip():
        return None
    try:
        raw = base64.b64decode(b64, validate=False)
        arr = np.frombuffer(raw, dtype="<f4")
    except Exception:
        return None
    if arr.size != MEL_LENGTH:
        return None
    return arr.reshape(N_MELS, N_FRAMES).astype(np.float32)


def _truthy(value) -> bool:
    """Robustly parse CSV/Supabase booleans (True/true/t/1/yes)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value) and not (isinstance(value, float) and np.isnan(value))
    s = str(value).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}


def load_observations(csv_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a Supabase `Version_1.observations` CSV export.

    Keeps only rows where `included_feedback` is true and the row has a valid
    mel spectrogram plus a `species_name` present in the label list.

    Returns (states, labels, stats) where:
      states  : float32 array [N, 1, 64, 157]
      labels  : int64 array [N]   (ground-truth class index from species_name)
      stats   : dict with bookkeeping counts
    """
    df = pd.read_csv(csv_path)
    stats = {
        "total_rows": int(len(df)),
        "with_feedback": 0,
        "agreed": 0,
        "disputed": 0,
        "skipped_no_feedback": 0,
        "skipped_bad_mel": 0,
        "skipped_bad_label": 0,
    }

    required = {"mel_spectrogram", "species_name", "included_feedback"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_cols)}. "
            f"Found columns: {list(df.columns)}"
        )

    states: list[np.ndarray] = []
    labels: list[int] = []
    feedback_col = "feedback" if "feedback" in df.columns else None

    for row in df.itertuples(index=False):
        rd = row._asdict()
        if not _truthy(rd.get("included_feedback")):
            stats["skipped_no_feedback"] += 1
            continue
        stats["with_feedback"] += 1

        name = rd.get("species_name")
        name = name.strip() if isinstance(name, str) else name
        if name not in LABEL_TO_IDX:
            stats["skipped_bad_label"] += 1
            continue

        mel = decode_mel(rd.get("mel_spectrogram"))
        if mel is None:
            stats["skipped_bad_mel"] += 1
            continue

        states.append(mel[None, :, :])  # add channel dim -> [1, 64, 157]
        labels.append(LABEL_TO_IDX[name])

        if feedback_col is not None and _truthy(rd.get(feedback_col)):
            stats["agreed"] += 1
        else:
            stats["disputed"] += 1

    if not states:
        return (
            np.zeros((0, 1, N_MELS, N_FRAMES), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            stats,
        )
    return (
        np.stack(states).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Policy network (load ONNX -> trainable torch module)
# ─────────────────────────────────────────────────────────────────────────────
def load_policy(onnx_path: Path) -> nn.Module:
    """Load an ONNX classifier into a trainable torch.nn.Module via onnx2torch."""
    # Side-effect import: registers opset-18 reduce converters that onnx2torch
    # 1.5.x lacks (the eFrog model is exported at opset 18). See module docstring.
    try:
        import onnx2torch_compat  # noqa: F401
    except ImportError:  # pragma: no cover
        from . import onnx2torch_compat  # type: ignore  # noqa: F401
    from onnx2torch import convert

    model = convert(str(onnx_path))
    # The classifier outputs raw logits over the 19 classes; softmax(logits) is
    # the categorical policy distribution.
    model.train()
    return model


def _policy_logits(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """Forward pass returning logits [B, N_CLASSES]."""
    logits = model(states)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return logits


# ─────────────────────────────────────────────────────────────────────────────
# REINFORCE training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(
    model: nn.Module,
    states: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    entropy_coef: float,
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Fine-tune the policy with REINFORCE. Returns the trained model (on CPU)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    states_t = torch.from_numpy(states).to(device)
    labels_t = torch.from_numpy(labels).to(device)
    n = states_t.shape[0]

    # Running-mean reward baseline for variance reduction (b in the loss).
    baseline = 0.0
    baseline_momentum = 0.9

    print(
        f"REINFORCE: {n} feedback observations, {epochs} epochs, "
        f"lr={lr}, batch={batch_size}, entropy_coef={entropy_coef}, "
        f"device={device}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_reward, ep_correct, ep_loss, ep_seen = 0.0, 0, 0.0, 0
        t0 = time.time()

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            s = states_t[idx]
            y = labels_t[idx]

            logits = _policy_logits(model, s)
            dist = torch.distributions.Categorical(logits=logits)

            # Sample an action (class) from the policy — this is the RL step.
            actions = dist.sample()
            # Reward: +1 for a correct action, -1 otherwise.
            rewards = torch.where(
                actions == y,
                torch.ones_like(actions, dtype=torch.float32),
                -torch.ones_like(actions, dtype=torch.float32),
            )

            # Advantage = reward - baseline (running mean).
            advantage = rewards - baseline
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            # REINFORCE objective (minimise negative expected return).
            pg_loss = -(advantage.detach() * log_prob).mean()
            ent_loss = -entropy_coef * entropy.mean()
            loss = pg_loss + ent_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # Update baseline toward this batch's mean reward.
            batch_mean = float(rewards.mean().item())
            baseline = baseline_momentum * baseline + (1 - baseline_momentum) * batch_mean

            bs = s.shape[0]
            ep_reward += float(rewards.sum().item())
            # Greedy accuracy (argmax of policy) — a stable progress metric.
            ep_correct += int((logits.argmax(dim=1) == y).sum().item())
            ep_loss += float(loss.item()) * bs
            ep_seen += bs

        mean_reward = ep_reward / max(1, ep_seen)
        greedy_acc = ep_correct / max(1, ep_seen)
        print(
            f"  epoch {epoch:3d}/{epochs}  "
            f"mean_reward={mean_reward:+.4f}  "
            f"greedy_acc={greedy_acc:.4f}  "
            f"loss={ep_loss / max(1, ep_seen):+.4f}  "
            f"baseline={baseline:+.4f}  "
            f"({time.time() - t0:.2f}s)",
            flush=True,
        )

    model.eval().cpu()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────
def export_onnx(model: nn.Module, out_path: Path) -> None:
    """Export the trained policy back to ONNX with the original signature."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.zeros(1, 1, N_MELS, N_FRAMES, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "time"},
            "output": {0: "batch_size"},
        },
        opset_version=18,
        dynamo=False,
    )


def verify_onnx(out_path: Path) -> None:
    """Sanity-check that the exported ONNX loads and produces [1, 19] logits."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(out_path)))
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 1, N_MELS, N_FRAMES), dtype=np.float32)
    out = sess.run(None, {name: dummy})[0]
    assert out.shape[-1] == N_CLASSES, f"Expected last dim {N_CLASSES}, got {out.shape}"
    print(f"Verified ONNX: input '{name}', output shape {out.shape}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (fabricate model + data, run end-to-end)
# ─────────────────────────────────────────────────────────────────────────────
def run_smoke(args: argparse.Namespace) -> int:
    """Generate a tiny synthetic ONNX model + observations CSV, then train."""
    try:
        from make_synthetic_data import make_synthetic_model, make_synthetic_csv
    except ImportError:
        from .make_synthetic_data import make_synthetic_model, make_synthetic_csv  # type: ignore

    work = HERE / "_smoke"
    work.mkdir(exist_ok=True)
    model_path = work / "synthetic_classifier.onnx"
    csv_path = work / "synthetic_observations.csv"
    # Keep the smoke artifact inside _smoke/ unless the user overrode --out away
    # from the default (argparse always populates args.out with the default).
    if args.out and Path(args.out) != DEFAULT_OUT:
        out_path = Path(args.out)
    else:
        out_path = work / "frog_classifier_rl.onnx"

    print("[smoke] fabricating synthetic ONNX model and observations...", flush=True)
    make_synthetic_model(model_path)
    make_synthetic_csv(csv_path, n_rows=48, seed=args.seed)

    args.model = str(model_path)
    args.observations = str(csv_path)
    args.out = str(out_path)
    rc = run_pipeline(args)
    if rc == 0:
        print(f"[smoke] SUCCESS — new ONNX written to {out_path}", flush=True)
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline driver
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    csv_path = Path(args.observations)
    out_path = Path(args.out) if args.out else DEFAULT_OUT

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2
    if not csv_path.exists():
        print(f"ERROR: observations CSV not found: {csv_path}", file=sys.stderr)
        return 2

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )

    print(f"Loading observations from {csv_path} ...", flush=True)
    states, labels, stats = load_observations(csv_path)
    print("Observation stats:", flush=True)
    for k, v in stats.items():
        print(f"  {k:22s}: {v}", flush=True)

    if states.shape[0] == 0:
        print(
            "ERROR: no usable feedback observations after filtering. "
            "Need rows with included_feedback=true, a valid mel_spectrogram, "
            "and a species_name in the 19-class label list.",
            file=sys.stderr,
        )
        return 3

    print(f"Loading policy network from {model_path} (onnx2torch) ...", flush=True)
    model = load_policy(model_path)

    model = train(
        model,
        states,
        labels,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        device=device,
    )

    print(f"Exporting trained policy to ONNX: {out_path} ...", flush=True)
    export_onnx(model, out_path)
    verify_onnx(out_path)
    print("Done.", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="REINFORCE post-training for the eFrog ONNX frog-call classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", help="Path to the input ONNX classifier.")
    p.add_argument(
        "--observations",
        help="Path to a Supabase Version_1.observations CSV export.",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output ONNX path (defaults inside post-training/).",
    )
    p.add_argument("--epochs", type=int, default=8, help="Number of RL epochs.")
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    p.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    p.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient (exploration).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA exists.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run a self-contained end-to-end smoke test on synthetic data.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        return run_smoke(args)
    if not args.model or not args.observations:
        print(
            "ERROR: --model and --observations are required (or use --smoke).",
            file=sys.stderr,
        )
        return 2
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
