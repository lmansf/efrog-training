# eFrog RL Post-Training

A **reinforcement-learning** fine-tuning pipeline for the eFrog Florida
frog-call classifier. It takes the existing ONNX classifier plus a batch of
real user observations (with feedback) and produces a **new ONNX model** that
has been nudged by a policy-gradient (REINFORCE) update toward the corrections
users actually made in the field.

This is *not* supervised cross-entropy fine-tuning. The classifier is treated as
a stochastic **policy**, actions (predicted species) are **sampled**, a scalar
**reward** is computed from whether the sampled species matched the user's
ground truth, and the network is updated with the REINFORCE gradient estimator.

---

## RL formulation

| RL concept | eFrog instantiation |
|------------|---------------------|
| Policy `π_θ` | The classifier itself, loaded from ONNX into a trainable `torch.nn.Module` via [`onnx2torch`](https://github.com/ENOT-AutoML/onnx2torch). The 19 raw output logits define a categorical policy `π_θ(a\|s) = softmax(logits(s))`. |
| State `s` | One observation's log-mel spectrogram, tensor shape `[1, 1, 64, 157]` of dB values (range ≈ `[-80, 0]`). |
| Action `a` | A class index **sampled** from `Categorical(softmax(logits))` — a genuine stochastic action, not an argmax. |
| Reward `r` | `+1` if `a == ground-truth label` (the user-confirmed `species_name`), else `-1`. |
| Objective | Maximise expected reward `E[r]` via REINFORCE. |

### Loss

```
loss = -(r - b) · log π_θ(a|s)  -  β · H[π_θ(·|s)]
```

* `(r - b)` is the **advantage**: reward minus a **baseline** `b` (a running
  mean of recent rewards) for variance reduction. This is the standard
  REINFORCE-with-baseline trick.
* `log π_θ(a|s)` is the log-probability of the sampled action — increasing it
  when advantage is positive makes good actions more likely, and vice versa.
* `H[π_θ(·|s)]` is the policy **entropy**; the `-β · H` term is a small
  **entropy bonus** (`--entropy-coef`, default `0.01`) that keeps the policy
  exploring rather than collapsing prematurely.
* Optimised with **Adam** at a small learning rate (default `1e-4`), a few
  epochs, in mini-batches. Gradients are clipped at norm 5.

The greedy accuracy (`argmax` of the policy) is logged each epoch as a stable
progress signal alongside the noisier mean reward.

---

## Data contract

The pipeline reads a CSV export of the eFrog Supabase table
`Version_1.observations`. Only these columns are used:

| Column | Type | Role |
|--------|------|------|
| `mel_spectrogram` | TEXT | base64 of a little-endian `float32` array of length `64*157 = 10048`, row-major as `mel[mel_bin*157 + frame]`. Decoded with `np.frombuffer(base64.b64decode(s), dtype='<f4').reshape(64, 157)` → model input `[1,1,64,157]`. |
| `species` | TEXT | The model's predicted top species at capture time (one of the 19 names). Informational only. |
| `species_name` | TEXT | **The training label** — the user-confirmed/corrected ground-truth species. On *agree* it equals the prediction; on *dispute* it is the user's correction. |
| `included_feedback` | BOOLEAN | `true` when the user gave feedback (agree or dispute). **Rows are filtered to `true`.** |
| `feedback` | BOOLEAN (nullable) | `true` = agreed, `false` = disputed, `null` = no feedback. Used only for reporting agree/dispute counts. |

**Filtering rules** (applied in `load_observations`):

1. Keep only rows where `included_feedback` is true.
2. Skip rows whose `mel_spectrogram` is blank or not exactly 10048 float32s.
3. Skip rows whose `species_name` is blank or not one of the 19 labels.
4. Map `species_name` → class index via the ordered label list in `labels.py`.

The 19 labels (index 0..18) live in [`labels.py`](./labels.py) and must stay in
sync with the classifier's output order.

### Exporting observations from Supabase to CSV

* **Dashboard:** Supabase Studio → Table Editor → `Version_1.observations` →
  *Export* → *Export to CSV*.
* **SQL / psql:**
  ```sql
  \copy (
    select mel_spectrogram, species, species_name, included_feedback, feedback
    from "Version_1".observations
    where included_feedback = true
  ) to 'observations.csv' with csv header;
  ```
* **CLI:** `supabase db dump` / the REST endpoint with `Accept: text/csv` also
  work; only the five columns above are required (extra columns are ignored).

---

## Install

```bash
pip install -r requirements.txt        # this folder, or the repo-root one
```

Key dependency added for this pipeline: **`onnx2torch`** (plus `torch`, `onnx`,
`onnxruntime`, `numpy`, `pandas`, already in the repo).

---

## Run

```bash
python train_rl.py \
  --model ../artifacts/frog_classifier.onnx \
  --observations observations.csv \
  --out frog_classifier_rl.onnx \
  --epochs 8 --lr 1e-4 --batch-size 16 --entropy-coef 0.01 --seed 42
```

Output: a new ONNX model with the **same** signature as the input
(`input` `[1,1,64,157]` → `output` `[1,19]`, opset 18), written to `--out`
(default `post-training/frog_classifier_rl.onnx`). The script verifies the
exported model loads in ONNX Runtime and emits `[1, 19]` logits.

### Smoke test (no real data needed)

```bash
python train_rl.py --smoke
```

This fabricates a tiny synthetic ONNX classifier and ~50 synthetic observation
rows (via `make_synthetic_data.py`), runs the full load → REINFORCE → export →
verify pipeline, and writes a new ONNX file under `post-training/_smoke/`. Use
it to confirm the plumbing works end-to-end. You can also generate the synthetic
inputs standalone:

```bash
python make_synthetic_data.py            # writes _smoke/synthetic_*.{onnx,csv}
```

---

## Files

| File | Purpose |
|------|---------|
| `train_rl.py` | The CLI REINFORCE pipeline (load → train → export → verify). |
| `labels.py` | The 19-class ordered label list and the feature contract constants. |
| `make_synthetic_data.py` | Fabricates a synthetic ONNX model + observations CSV for the smoke test. |
| `onnx2torch_compat.py` | Registers opset-18 reduce-op converters that `onnx2torch` 1.5.x lacks (the eFrog model is exported at opset 18). |
| `rl_post_training.ipynb` | Notebook walkthrough of the same pipeline. |
| `requirements.txt` | Standalone dependency list for this folder. |

---

## Limitations & caveats

* **REINFORCE is high-variance.** With only a handful of observations the reward
  signal is noisy; the running-mean baseline and entropy bonus help, but expect
  jittery per-epoch rewards. Treat this as a *gentle nudge*, not a retrain.
* **Small-data regime.** Field feedback batches are tiny relative to the
  original training set. Use a small learning rate and few epochs to avoid
  catastrophic forgetting of the pre-trained behaviour. Validate the new model
  against a held-out set before shipping it.
* **Disputed labels are the learning signal.** *Agree* rows reinforce existing
  correct behaviour; *dispute* rows carry the real corrective signal (the user
  told the model it was wrong and supplied the right species). A batch that is
  all-agree mostly reinforces the status quo.
* **Reward is 0/1-style (`±1`) on a single sampled action.** It does not use
  partial credit or class similarity. Confusable species pairs are treated as
  fully wrong.
* **Trusting user labels.** `species_name` is taken as ground truth. Erroneous
  user corrections become erroneous reward — consider gating on confidence or
  multiple confirmations upstream.
* **No reward shaping for the app's sigmoid head.** The eFrog app applies a
  per-class sigmoid to the logits at inference; this RL loop optimises the
  softmax policy over classes. The exported logits remain compatible with the
  app's sigmoid usage, but the training objective is the categorical policy.
* **onnx2torch opset coverage.** `onnx2torch_compat.py` patches the opset-18
  reduce operators used by the eFrog architecture. A model using other
  unsupported opset-18 ops may need additional shims.
