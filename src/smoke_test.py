"""Smoke test: runs the core summarization + translation pipeline on an existing transcript.
Mirrors src/app.py process_video (standard mode) without the Streamlit/UI/video I/O.
Run from repo root:  .venv\Scripts\python.exe src\smoke_test.py
"""
import os, sys, time, traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

import torch
from inference import load_model, generate_summary
from translate_stage import load_translation_model, translate_texts, translate_one
from glossary import glossary_from_text

TRANSCRIPT = "Should_YOU_Become_An_AI_Engineer.txt"
TGT = "hin_Deva"

def main():
    t0 = time.time()
    txt_path = REPO / "texts" / TRANSCRIPT
    transcript = txt_path.read_text(encoding="utf-8").strip()
    print(f"[load] transcript chars={len(transcript)}  ({time.time()-t0:.1f}s)")

    # --- 1. Summarization (LED + hierarchical adapter) ---
    print("[summ] loading LED hier model...")
    model, tokenizer = load_model(adapter="hier")
    print(f"[summ] model loaded ({time.time()-t0:.1f}s); generating summary...")
    summary = generate_summary(transcript, model=model, tokenizer=tokenizer, adapter="hier")
    print("===== ENGLISH SUMMARY =====")
    print(summary)
    print("============================")
    del model, tokenizer
    torch.cuda.empty_cache()

    # --- 2. Translation (IndicTrans2, 4-bit) ---
    print(f"[tran] loading IndicTrans2 4-bit ({TGT})...")
    tmodel, ttokenizer, tip = load_translation_model(
        model_name="ai4bharat/indictrans2-en-indic-1B", device="cuda", load_4bit=True
    )
    print(f"[tran] model loaded ({time.time()-t0:.1f}s)")

    translate_fn = lambda t: translate_one(tmodel, ttokenizer, tip, t, TGT)
    glossary_pairs = glossary_from_text(summary, translate_fn=translate_fn, backend="ner")
    print(f"[tran] glossary pairs: {glossary_pairs}")

    translations = translate_texts(
        tmodel, ttokenizer, tip, [summary], TGT, context=summary, glossary_pairs=glossary_pairs
    )
    out = translations[0] if translations else "(empty)"
    print("===== HINDI TRANSLATION =====")
    print(out)
    print("=============================")
    print(f"[done] total time {time.time()-t0:.1f}s")

    # Persist UTF-8 results (console may mangle Indic scripts).
    res_path = REPO / "summaries" / "smoke_test_result.txt"
    res_path.write_text(
        f"TRANSCRIPT: {TRANSCRIPT}\n\nENGLISH SUMMARY:\n{summary}\n\n"
        f"GLOSSARY: {glossary_pairs}\n\nHINDI TRANSLATION ({TGT}):\n{out}\n",
        encoding="utf-8",
    )
    print(f"[result] saved -> {res_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
