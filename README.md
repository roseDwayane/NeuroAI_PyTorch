# Deep Learning for EEG — Hands-on Colab Notebooks

Practicum materials for the **Cutting-EEG Workshop, September 2026, UCSD**
(*Deep Learning EEG Methods and Practice*).

Four self-contained Google Colab notebooks taking you from "I have preprocessed EEG"
to fine-tuning foundation models and fusing multiple modalities. Everything runs on a
free Colab GPU with no local setup and no manual downloads.

**Kung-Yi Chang** · National Tsing Hua University

---

## Notebooks

| # | Notebook | Level | Runtime | Open |
|---|---|---|---|---|
| 1 | **From Data to Model: The Core Braindecode Pipeline** | Beginner | ~8 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roseDwayane/NeuroAI_PyTorch/blob/main/notebooks/01_braindecode_pipeline.ipynb) |
| 2 | **Adding Attention: Transformers for EEG** | Intermediate | ~12 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roseDwayane/NeuroAI_PyTorch/blob/main/notebooks/02_transformers_for_eeg.ipynb) |
| 3 | **Standing on the Shoulders of Giants: Fine-tuning EEG Foundation Models** | Core | ~15 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roseDwayane/NeuroAI_PyTorch/blob/main/notebooks/03_finetuning_foundation_models.ipynb) |
| 4 | **Broadening the Scope: Multimodal Alignment and Fusion** | Bonus | ~10 min | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roseDwayane/NeuroAI_PyTorch/blob/main/notebooks/04_multimodal_fusion.ipynb) |

Runtimes are for a free Colab **T4 GPU**. Everything also runs on CPU, roughly 3–5×
slower. Notebook 1 is a prerequisite for 2–4; the rest are independent of each other.

> **Note on the Colab badges:** they point at `roseDwayane/NeuroAI_PyTorch` on `main`.
> If you fork or rename the repo, update the URLs. Until the repo is public, use
> *File → Upload notebook* in Colab instead.

---

## What each notebook covers

### 1 · The core pipeline
The data-format problem beginners hit first. Converting MNE/EEGLAB-preprocessed data
into PyTorch `Dataset` and `DataLoader` objects, then training `EEGNet` and
`ShallowFBCSPNet` with a plain, readable training loop.

Ends with a **copy-paste template** for your own lab's data.

Also covers the four traps: the `[moabb]` extra, random-split leakage, the three-element
batch, and the renamed model arguments.

### 2 · Transformers
Tokenizing continuous EEG into patches — the step that makes attention applicable, and
the same trick every EEG foundation model uses. Builds a ViT-style encoder from scratch
(no hidden library code), visualizes what the model attended to, and compares against
`EEGConformer`.

Sets up Notebook 3 by showing *why* pure Transformers struggle on small EEG datasets.

### 3 · Foundation models
Self-supervised pretraining via masked reconstruction on unlabeled EEG, then transfer to
a held-out subject. Compares **linear probe / full fine-tune / from-scratch**, plus
partial freezing, and plots the **label-efficiency curve** that actually justifies the
approach.

Section 7 shows the exact `strict=False` pattern for loading published checkpoints
(LaBraM, BENDR, EEGPT), and the two silent failure modes: montage and sampling-rate
mismatch.

### 4 · Multimodal fusion
Aligning EEG with a second stream at a different sampling rate, quantifying what a
timing offset costs, and comparing **early / late / attention** fusion in a two-stream
network. Includes graceful degradation when a modality drops out mid-recording.

---

## Dataset

All notebooks use **BCI Competition IV dataset 2a** (motor imagery: left hand, right
hand, feet, tongue — 9 subjects, 22 EEG channels, 250 Hz), downloaded automatically via
MOABB. First run fetches ~40 MB per subject and caches it.

Notebook 4 pairs the real EEG with a **simulated** gaze stream. This is deliberate: since
we control the ground-truth alignment, we can measure exactly what a timing offset costs
— which is impossible with real data where the true offset is unknown. The alignment and
fusion code is unchanged for real recordings.

---

## Running the notebooks

1. Open a notebook via its Colab badge above
2. `Runtime → Change runtime type → T4 GPU`
3. Run the first cell. **It installs dependencies and restarts the runtime** — the
   "crash" notice is expected and intentional
4. Continue from Section 1. Do not re-run the install cell

The restart is necessary because the install upgrades numpy, pandas, and mne past the
versions Colab preloads; without it you get stale C-extension errors.

---

## Requirements

Handled automatically by the first cell of each notebook:

```
braindecode[moabb]>=1.7.0    # the [moabb] extra is required for the datasets
torch>=2.0
mne>=1.11
```

Verified against **braindecode 1.7.0** / **moabb 1.5.0** (Python ≥3.11). Three API
details that break older tutorials, all handled here:

- The model class is `EEGNet`, **not** `EEGNetv4`
- Arguments are `n_chans` / `n_outputs` / `n_times`; the old `in_chans` / `n_classes` /
  `input_window_samples` names were **removed**, not deprecated — they raise
- MOABB session keys are `"0train"` / `"1test"`, not `"session_T"` / `"session_E"`

---

## Suggested workshop timing

Sized for a ~2 hour practicum slot. Notebook 4 is the natural cut if time runs short.

| Segment | Time |
|---|---|
| NB 1 — core pipeline | 30 min |
| NB 2 — transformers | 25 min |
| NB 3 — foundation models | 35 min |
| NB 4 — multimodal (optional) | 20 min |
| Q&A / hands-on | remainder |

Each notebook ends with **"Try it yourself"** exercises suitable for hands-on time or
for participants to take home.

---

## A note on the results

Every reported number comes from a single subject with ~280 training trials. Differences
of a few points between architectures are **noise**, and the notebooks say so where it
matters. The goal is working code and correct methodology, not benchmark claims.

The "foundation model" in Notebook 3 is pretrained in-notebook on 5 subjects, so its
gains are a **lower bound** on what published models trained on thousands of hours
achieve. This keeps the notebook self-contained and reproducible in ~10 minutes with no
external checkpoint that might move or disappear.

---

## License

Code released under the repository's LICENSE. BCI Competition IV 2a is distributed under
its own terms — see [MOABB](https://moabb.neurotechx.com/) — and published foundation
model checkpoints carry their own licenses, some research-only. Check before using them
in a paper or product.
