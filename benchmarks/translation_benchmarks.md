# English-to-Hindi Translation Benchmarks


---

### Metric Definitions
* **SacreBLEU (`flores200` / `13a`)**: Corpus-level BLEU using Flores200 SentencePiece tokenization (primary Indic MT standard) and standard `13a` tokenization.
* **Raw Flores200 BLEU**: Geometric mean of 1–4 gram precisions *without* Brevity Penalty (isolates lexical translation precision from length mismatch).
* **chrF++ ($w=2$)**: Character n-gram $F$-score with word bigrams (robust to Hindi morphological inflections).
* **Brevity Penalty (BP)**: Penalizes premature truncation ($1.0$ = exact reference length match).
* **Devanagari Script Purity & Forbidden Indic Chars**: Audits Unicode blocks to detect non-target Indic script bleed (Gurmukhi, Tamil, Gujarati, Kannada, Oriya).

---


## Subtopic Discourse Benchmark Table

Apples-to-apples comparison across all `sarvamai/sarvam-1` (2.5B) stages evaluated on the **35 canonical 15-sentence validation subtopics** 

| # | Model Stage | Architecture & Objective | Training Data & Length | Reported Flores200 BLEU | Raw Flores200 BLEU | BLEU (`13a`) | chrF++ ($w=2$) | Brevity Penalty (BP) | Sys / Ref Tokens | 1-Gram Precision | Forbidden Indic Chars | Devanagari Purity | Peak VRAM |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Zero-Shot Baseline (Devanagari Anchored)** | `sarvamai/sarvam-1` (2.5B NF4) | Untrained base model *(Devanagari prompt anchor)* | 11.94 | 11.98 | 8.19 | 31.49 | **0.996** | 22,319 / 22,398 *(45-subtopic run)* | 39.28% | Frequent Gujarati/Kannada drift | 80.20% | 1,805.3 MB |
| **2** | **Fine-tuned QLora** | **`sarvam-1` + LoRA ($r=16, \alpha=32$)** | **6,000 diversified subtopics, avg 256.4 words (200–320w)**| **`11.52`** | **`15.02`** | **`6.95`** | **`34.55`** | **`0.767`** | **10,736 / 13,588** | **`49.97%`** | **`8 chars` (0.0238%, -99.8%)** | **`93.69%`** | **`1,740.0 MB`** |

### Key Gains of Step 200 Script-Constrained Champion
* **vs. Zero-Shot Sarvam-1 Baseline**: **+3.04** Raw Flores200 BLEU (+25.4% rel), **+3.06** chrF++ (+9.7% rel), **+10.69%** 1-gram precision, **+13.49%** Devanagari purity, eliminating Gujarati/Kannada script drift and repetition loops.

---


## Production Quantization & Latency Benchmark

Evaluated end-to-end on a 15m 43s NPTEL lecture (`Deep Learning CS7015 Lec 3.2`, RTX 4050 6 GB):

| Deployment Engine | Model Artifact | VRAM Footprint | Peak System RAM | Decode Speed | Phase 4 Latency (Full Lecture Notes) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PyTorch 4-bit NF4 + Unmerged PEFT LoRA** | `sarvamai/sarvam-1` + `best_checkpoint/` | ~1.74 GB | 4,225.9 MB | ~15–25 tok/s | 395.23 s *(6.6 min)* |
| **Merged GGUF `Q4_K_M` via Ollama (Production)** | `models/sarvam1_hi_step200_q4_k_m.gguf` (`sarvam1-hi-step200`) | **1.80 GB** *(100% GPU)* | **1,559.4 MB** *(-63%)* | **~109 tok/s** | **120.06 s** *(3.29× faster)* |

---

