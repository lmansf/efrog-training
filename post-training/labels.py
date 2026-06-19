"""Shared label/feature contract for the eFrog RL post-training pipeline.

These constants MUST stay in sync with the training notebook (`EDA-Master.ipynb`)
and the eFrog server's feature extractor. The 19 class labels are listed in the
exact order the classifier emits logits (index 0..18).
"""

from __future__ import annotations

# ── Feature contract — MUST match EDA-Master.ipynb / efrog/server.py ──────────
N_MELS = 64
N_FRAMES = 157  # 1 + (16000 * 5.0) // 512
MEL_LENGTH = N_MELS * N_FRAMES  # 10048 float32 values per spectrogram

# ── The 19 class labels, IN ORDER (logit index 0..18) ────────────────────────
LABEL_CLASSES = [
    "American Bullfrog",            # 0
    "Barking Tree Frog",            # 1
    "Cane Toad",                    # 2
    "Coastal Plains Leopard Frog",  # 3
    "Cuban Tree Frog",              # 4
    "Eastern Narrow-mouthed Toad",  # 5
    "Eastern Spadefoot",            # 6
    "Green Frog",                   # 7
    "Green Treefrog",               # 8
    "Greenhouse Frog",              # 9
    "Little Grass Frog",            # 10
    "Oak Toad",                     # 11
    "Pig Frog",                     # 12
    "Pine Woods Tree Frog",         # 13
    "Southern Chorus Frog",         # 14
    "Southern Cricket Frog",        # 15
    "Southern Leopard Frog",        # 16
    "Southern Toad",                # 17
    "Squirrel Tree Frog",           # 18
]
N_CLASSES = len(LABEL_CLASSES)
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_CLASSES)}

assert N_CLASSES == 19, "Expected exactly 19 frog classes"
