# Program text for the website — Day 1 afternoon practicum

Draft copy for the Cutting Gardens 2026 San Diego program page.
Prepared by Kung-Yi Chang. Covers **Part 2 (deep learning)** only — Part 1
(preprocessing) is left as-is for the instructors covering that half.

---

## Option A — drop-in replacement (short)

Closest to the current website text, with the deep-learning half made concrete.

> **14:00 — Practicum: Deep Learning EEG Methods and Practice**
>
> Preprocessing across EEGLAB, Brainstorm, MNE-Python and EEGPrep, followed by deep
> learning on EEG with PyTorch-based frameworks (Braindecode, NeuroAI) and existing EEG
> foundation models. The deep-learning half runs as four self-contained Google Colab
> notebooks — from converting preprocessed recordings into PyTorch training pipelines, to
> Transformers for EEG, to fine-tuning pretrained foundation models, with an optional
> segment on multimodal EEG + eye-gaze fusion. Theory combined with light hands-on
> practice; a laptop with a browser is all that is required.
>
> *Instructors: Kung-Yi Chang (UC San Diego / NTHU) and colleagues — full teaching team to
> be announced.*
>
> **17:00 — Wrap-up & Q&A**

---

## Option B — expanded (if the page has room)

> **14:00 — Practicum: Deep Learning EEG Methods and Practice**
>
> A single unified hands-on session covering both preprocessing and deep learning.
>
> **Part 1 — Preprocessing across platforms.** EEGLAB, Brainstorm, MNE-Python and EEGPrep.
>
> **Part 2 — Deep learning on EEG.** Four self-contained Google Colab notebooks, running on
> free cloud GPUs with no local installation:
>
> 1. **From Data to Model** — converting EEGLAB/MNE-preprocessed recordings into PyTorch
>    `Dataset` and `DataLoader` objects, then training classic models (EEGNet,
>    ShallowFBCSPNet) with Braindecode.
> 2. **Adding Attention** — patching and tokenization of continuous EEG, and a compact
>    Transformer encoder built from scratch.
> 3. **Fine-tuning EEG Foundation Models** — self-supervised pretraining and transfer
>    learning: linear probing, full fine-tuning, and partial freezing, plus the practical
>    recipe for loading published checkpoints.
> 4. **Multimodal Alignment and Fusion** *(bonus)* — aligning EEG with eye-gaze and
>    combining both in a two-stream network.
>
> Each notebook ends with a reusable template participants can apply to their own data. All
> examples use a public motor-imagery dataset that downloads automatically — no data
> preparation or local setup required.
>
> *Participants need only a laptop with a browser and a Google account.*
>
> **17:00 — Wrap-up & Q&A**

---

## Notes for the organizers

- **Instructor listing.** I've written myself in as *Kung-Yi Chang (UC San Diego / NTHU)*.
  Please correct the affiliation to whatever is accurate for September — I wasn't sure
  whether to list NTHU, the SCCN visiting position, or both, and I'd rather you set it than
  guess.

- **Timing.** The four notebooks run about 30 / 25 / 35 / 20 minutes including discussion.
  If Part 1 needs more of the 14:00–17:00 block, **Notebook 4 is the intended cut** — it's
  marked bonus for exactly that reason. Notebooks 1–3 form the coherent core.

- **Prerequisites for the participant email.** Suggested wording:
  > For the deep-learning practicum, please bring a laptop with a web browser and sign in
  > to a Google account beforehand. All code runs in Google Colab — no installation is
  > needed. Basic Python familiarity is helpful but not required; every notebook runs
  > top to bottom without edits.

- **Notebook links.** I'd suggest circulating them a few days ahead so people can open them
  in advance — the first run downloads ~40 MB of EEG data per subject, and doing that on
  workshop wifi for a full room is slow. I'll send the final links once the repository is
  public.

- **The "Committee note — call for contributors" box** in the booklet is marked for removal
  before public circulation. I've left it in place; that's your call, not mine.
