#!/usr/bin/env python3
"""Fabricate a tiny synthetic ONNX classifier and observation CSV.

This lets the whole RL post-training pipeline run end-to-end with NO real data,
purely to prove the plumbing works. The synthetic model mirrors the eFrog
architecture conventions (GroupNorm, time-only mean pooling, dB->[-1,1] input
scaling, [1,1,64,157] -> [1,19] signature) from `EDA-Master.ipynb`.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from labels import LABEL_CLASSES, N_CLASSES, N_FRAMES, N_MELS
except ImportError:  # pragma: no cover
    from .labels import LABEL_CLASSES, N_CLASSES, N_FRAMES, N_MELS  # type: ignore


class _ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.GroupNorm(8, c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class _FrogCNN(nn.Module):
    """Small stand-in for the real FrogCNN, same I/O contract.

    Input: dB mel (batch, 1, 64, time), values ~ [-80, 0].
    Output: one logit per class. Time is pooled away; frequency is preserved.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.blocks = nn.Sequential(
            _ConvBlock(1, 16),
            _ConvBlock(16, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 64),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * (N_MELS // 16), n_classes),
        )

    def forward(self, x):
        x = x / 40.0 + 1.0          # [-80, 0] dB -> ~[-1, 1]
        x = self.blocks(x)
        x = x.mean(dim=3)           # pool over time only
        return self.head(x.flatten(1))


def make_synthetic_model(out_path: Path) -> Path:
    """Build, randomly init, and export a synthetic FrogCNN to ONNX."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = _FrogCNN(N_CLASSES).eval()
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
    return out_path


def _encode_mel(mel: np.ndarray) -> str:
    """Encode a [64, 157] float32 mel as base64 little-endian raw bytes."""
    return base64.b64encode(mel.astype("<f4").tobytes()).decode("ascii")


def make_synthetic_csv(out_path: Path, n_rows: int = 48, seed: int = 42) -> Path:
    """Write a synthetic Supabase-style observations CSV.

    Each row carries a class-conditioned mel spectrogram so the classifier has
    a real (if noisy) signal to learn. A mix of agree/dispute/no-feedback rows
    and a couple of intentionally-bad rows exercise the filtering logic.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_rows):
        true_idx = int(rng.integers(0, N_CLASSES))
        true_name = LABEL_CLASSES[true_idx]

        # Class-conditioned mel: a frequency band lit up per class, in dB range.
        mel = rng.normal(-60.0, 5.0, size=(N_MELS, N_FRAMES)).astype(np.float32)
        band = (true_idx * (N_MELS // N_CLASSES + 1)) % N_MELS
        lo, hi = band, min(N_MELS, band + 6)
        mel[lo:hi, :] += rng.normal(35.0, 3.0, size=(hi - lo, N_FRAMES)).astype(np.float32)
        mel = np.clip(mel, -80.0, 0.0)

        # Roughly 70% feedback; of those ~60% agree, ~40% dispute.
        has_feedback = rng.random() < 0.7
        if has_feedback:
            agreed = rng.random() < 0.6
            if agreed:
                pred_name = true_name           # model was right
                feedback = True
            else:
                # Model predicted something else; user corrected to true_name.
                wrong_idx = (true_idx + 1 + int(rng.integers(0, N_CLASSES - 1))) % N_CLASSES
                pred_name = LABEL_CLASSES[wrong_idx]
                feedback = False
            species_name = true_name
        else:
            pred_name = true_name
            species_name = true_name            # present but should be skipped
            feedback = ""                        # null

        rows.append(
            {
                "id": i + 1,
                "mel_spectrogram": _encode_mel(mel),
                "species": pred_name,
                "species_name": species_name,
                "included_feedback": bool(has_feedback),
                "feedback": feedback,
            }
        )

    # Two intentionally-malformed rows to exercise the skip paths.
    rows.append(
        {
            "id": n_rows + 1,
            "mel_spectrogram": "",                # blank mel -> skipped
            "species": LABEL_CLASSES[0],
            "species_name": LABEL_CLASSES[0],
            "included_feedback": True,
            "feedback": True,
        }
    )
    rows.append(
        {
            "id": n_rows + 2,
            "mel_spectrogram": _encode_mel(
                np.zeros((N_MELS, N_FRAMES), dtype=np.float32)
            ),
            "species": "Not A Real Frog",
            "species_name": "Not A Real Frog",    # label not in list -> skipped
            "included_feedback": True,
            "feedback": False,
        }
    )

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Generate synthetic eFrog RL test data.")
    ap.add_argument("--model-out", default=str(here / "_smoke" / "synthetic_classifier.onnx"))
    ap.add_argument("--csv-out", default=str(here / "_smoke" / "synthetic_observations.csv"))
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    m = make_synthetic_model(Path(a.model_out))
    c = make_synthetic_csv(Path(a.csv_out), n_rows=a.rows, seed=a.seed)
    print(f"Wrote synthetic model -> {m}")
    print(f"Wrote synthetic CSV   -> {c}")
