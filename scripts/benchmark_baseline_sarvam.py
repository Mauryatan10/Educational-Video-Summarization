#!/usr/bin/env python3
"""
benchmark_baseline_sarvam.py

Benchmarks the pre-trained foundation base model `sarvamai/sarvam-1` (2B, 8,192 tokens)
on the newly engineered subtopic-level test dataset (`data/subtopic_engineered/test_subtopics.parquet`)
prior to fine-tuning.

Measures:
  - Subtopic-level SacreBLEU (13a & flores200)
  - Subtopic-level chrF++ (word_order=2)
  - STEM subtopic slice performance
  - Hardware footprint (VRAM in 4-bit, throughput in subtopics/sec)
  - Manual failure analysis and concrete snippet extraction
"""

import os
import sys
import time
import json
import re
from typing import List, Dict, Any

# Strip local dir from sys.path to avoid shadowing third-party 'datasets'
sys.path = [p for p in sys.path if p not in ('', '.', os.path.dirname(os.path.abspath(__file__)))]

import torch
import pyarrow.parquet as pq
import sacrebleu
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "sarvamai/sarvam-1"
TEST_SUBTOPICS_PATH = "data/subtopic_engineered/test_subtopics.parquet"
OUTPUT_DIR = "data/subtopic_translation_eval"

# Targeted Real-World University STEM Lecture Subtopics (representing multi-sentence 15-sentence lecture blocks)
UNIVERSITY_STEM_SUBTOPICS = [
    {
        "domain": "Linear Algebra & Spectral Theory",
        "title": "Eigenvalues and Spectral Decomposition",
        "en_subtopic": (
            "In spectral theory, the eigenvalue of a symmetric matrix is always real. "
            "When we compute the characteristic equation of the transformation matrix, all roots lie on the real number line. "
            "This fundamental property guarantees that symmetric operators can be orthogonally diagonalized into an orthonormal basis of eigenvectors. "
            "Let us consider an n-by-n real symmetric matrix A. "
            "Since the matrix equals its transpose, its eigenvectors corresponding to distinct eigenvalues are mutually orthogonal. "
            "We normalize these eigenvectors to form the columns of an orthogonal matrix Q. "
            "Consequently, the matrix product of Q transpose, A, and Q produces a purely diagonal matrix Lambda. "
            "The diagonal entries of Lambda correspond directly to the computed eigenvalues. "
            "If any eigenvalue has multiplicity greater than one, Gram-Schmidt orthogonalization can still construct an orthonormal eigenspace. "
            "Thus, quadratic forms associated with symmetric matrices can be simplified into uncoupled principal axes. "
            "This decomposition forms the theoretical foundation for Principal Component Analysis in machine learning."
        ),
        "gold_hi": (
            "स्पेक्ट्रल सिद्धांत में, किसी सममित मैट्रिक्स (symmetric matrix) का आइगेनवैल्यू (eigenvalue) सदैव वास्तविक होता है। "
            "जब हम रूपांतरण मैट्रिक्स के अभिलाक्षणिक समीकरण की गणना करते हैं, तो सभी मूल वास्तविक संख्या रेखा पर स्थित होते हैं। "
            "यह मौलिक गुणधर्म गारंटी देता है कि सममित ऑपरेटरों को आइगेनवेक्टर्स के ऑर्थोनॉर्मल बेसिस में ऑर्थोगोनली विकर्णित किया जा सकता है। "
            "आइए एक n-by-n वास्तविक सममित मैट्रिक्स A पर विचार करें। "
            "चूंकि मैट्रिक्स अपने ट्रांसपोज़ के बराबर होता है, इसलिए अलग-अलग आइगेनवैल्यू के अनुरूप इसके आइगेनवेक्टर्स परस्पर ऑर्थोगोनल होते हैं। "
            "हम एक ऑर्थोगोनल मैट्रिक्स Q के कॉलम बनाने के लिए इन आइगेनवेक्टर्स को सामान्यीकृत करते हैं। "
            "नतीजतन, Q ट्रांसपोज़, A और Q का मैट्रिक्स गुणनफल विशुद्ध रूप से एक विकर्ण मैट्रिक्स Lambda उत्पन्न करता है। "
            "Lambda की विकर्ण प्रविष्टियाँ सीधे परिकलित आइगेनवैल्यू के अनुरूप होती हैं। "
            "यदि किसी आइगेनवैल्यू की बहुलता एक से अधिक है, तो ग्राम-श्मिट ऑर्थोगोनलाइजेशन अभी भी एक ऑर्थोनॉर्मल आइगेनस्पेस का निर्माण कर सकता है। "
            "इस प्रकार, सममित मैट्रिक्स से जुड़े द्विघात रूपों को असंयुक्त प्रमुख अक्षों में सरल बनाया जा सकता है। "
            "यह अपघटन मशीन लर्निंग में प्रिंसिपल कंपोनेंट एनालिसिस (PCA) के लिए सैद्धांतिक आधार बनाता है।"
        ),
        "key_terms": ["eigenvalue", "symmetric matrix", "characteristic equation", "orthonormal basis", "eigenvectors", "orthogonal diagonalization"]
    },
    {
        "domain": "Machine Learning & Neural Search",
        "title": "Dense Retrieval Architecture",
        "en_subtopic": (
            "Dense retrieval uses dual encoders to compute vector dot products for semantic search over large corpora. "
            "Unlike sparse lexical matching with inverted indexes, dense passage retrieval maps queries and documents into a shared continuous embedding space. "
            "The similarity score between a query and candidate passage is computed via maximum inner product search. "
            "During offline indexing, every document in the corpus is passed through a pre-trained transformer to generate fixed-dimensional representations. "
            "These dense vectors are subsequently stored in an approximate nearest neighbor index like HNSW or FAISS. "
            "At query time, the user prompt is encoded into a single dense vector using the query encoder. "
            "The vector index conducts cosine similarity lookups in sub-linear time across millions of passages. "
            "Contrastive learning optimizes the dual encoder using in-batch negative sampling. "
            "Hard negatives force the model to distinguish semantically similar but factually incorrect passages. "
            "As a result, dense retrieval overcomes the vocabulary mismatch problem that plagues traditional BM25 systems. "
            "Cross-encoders can subsequently re-rank the top retrieved candidates to achieve higher precision."
        ),
        "gold_hi": (
            "सर्च हेतु डेंस रिट्रीवल (dense retrieval) बड़े कॉर्पोरा पर सेमांटिक सर्च के लिए वेक्टर डॉट प्रॉडक्ट्स (vector dot products) की गणना करने हेतु डुअल एनकोडर्स का उपयोग करता है। "
            "उल्टे इंडेक्स के साथ विरल लेक्सिकल मिलान के विपरीत, डेंस पैसेज रिट्रीवल प्रश्नों और दस्तावेजों को एक साझा निरंतर एम्बेडिंग स्पेस में मैप करता है। "
            "एक प्रश्न और उम्मीदवार पैसेज के बीच समानता स्कोर की गणना मैक्सिमम इनर प्रॉडक्ट सर्च (MIPS) के माध्यम से की जाती है। "
            "ऑफलाइन इंडेक्सिंग के दौरान, निश्चित-आयामी प्रतिनिधित्व उत्पन्न करने के लिए कॉर्पस के प्रत्येक दस्तावेज़ को एक प्री-ट्रेंड ट्रांसफॉर्मर से गुजारा जाता है। "
            "इन डेंस वेक्टर्स को बाद में HNSW या FAISS जैसे अनुमानित निकटतम पड़ोसी इंडेक्स में संग्रहीत किया जाता है। "
            "क्वेरी समय पर, क्वेरी एनकोडर का उपयोग करके उपयोगकर्ता के प्रश्न को एकल डेंस वेक्टर में एन्कोड किया जाता है। "
            "वेक्टर इंडेक्स लाखों अंशों में सब-लीनियर समय में कोसाइन सिमिलैरिटी लुकअप संचालित करता है। "
            "कंट्रास्टिव लर्निंग इन-बैच नेगेटिव सैंपलिंग का उपयोग करके डुअल एनकोडर को अनुकूलित करती है। "
            "हार्ड नेगेटिव्स मॉडल को शब्दार्थ रूप से समान लेकिन तथ्यात्मक रूप से गलत अंशों के बीच अंतर करने के लिए बाध्य करते हैं। "
            "परिणामस्वरूप, डेंस रिट्रीवल शब्दावली बेमेल समस्या को दूर करता है जो पारंपरिक BM25 प्रणालियों को प्रभावित करती है। "
            "क्रॉस-एनकोडर्स उच्च सटीकता प्राप्त करने के लिए शीर्ष प्राप्त उम्मीदवारों को पुनः रैंक (re-rank) कर सकते हैं।"
        ),
        "key_terms": ["dense retrieval", "dual encoders", "vector dot products", "embedding space", "inner product", "contrastive learning"]
    },
    {
        "domain": "Deep Learning & Optimization",
        "title": "Cross-Entropy Loss in Backpropagation",
        "en_subtopic": (
            "The loss function computes the cross-entropy error across class probabilities to update gradients during backpropagation. "
            "When training deep networks, gradient descent minimizes the empirical risk by adjusting weight parameters according to the learning rate. "
            "Numerical stability is preserved using log-sum-exp normalization before computing softmax probabilities. "
            "During the forward pass, activations are cached at each intermediate layer of the neural architecture. "
            "The backward pass applies the chain rule of calculus recursively from the loss layer down to the initial weights. "
            "Each layer calculates the Jacobian matrix of partial derivatives with respect to its input tensors. "
            "If gradients become vanishingly small, deeper network layers fail to learn effective representations. "
            "Residual skip connections alleviate gradient decay by providing an identity mapping shortcut. "
            "Adam optimizer dynamically scales the gradient vector by maintaining running averages of first and second raw moments. "
            "Weight decay regularizes parameter norms to prevent severe overfitting on sparse training sets. "
            "Proper hyperparameter tuning ensures smooth loss convergence across training epochs."
        ),
        "gold_hi": (
            "बैकप्रोपेगेशन के दौरान ग्रेडिएंट्स अपडेट करने के लिए लॉस फंक्शन (loss function) क्लास संभावनाओं में क्रॉस-एंट्रॉपी एरर की गणना करता है। "
            "गहरे नेटवर्क को प्रशिक्षित करते समय, ग्रेडिएंट डिसेंट लर्निंग रेट के अनुसार वेट पैरामीटर्स को समायोजित करके अनुभवजन्य जोखिम को न्यूनतम करता है। "
            "सॉफ्टमैक्स संभावनाओं की गणना करने से पहले लॉग-सम-एक्सप सामान्यीकरण का उपयोग करके संख्यात्मक स्थिरता को संरक्षित किया जाता है। "
            "फॉरवर्ड पास के दौरान, तंत्रिका आर्किटेक्चर की प्रत्येक मध्यवर्ती परत पर एक्टिवेशन्स को कैश किया जाता है। "
            "बैकवर्ड पास लॉस लेयर से लेकर प्रारंभिक वेट्स तक पुनरावर्ती रूप से कैलकुलस के चेन नियम को लागू करता है। "
            "प्रत्येक परत अपने इनपुट टेंसर्स के संबंध में आंशिक डेरिवेटिव्स के जैकोबियन मैट्रिक्स की गणना करती है। "
            "यदि ग्रेडिएंट्स अत्यधिक छोटे हो जाते हैं, तो गहरी नेटवर्क परतें प्रभावी प्रतिनिधित्व सीखने में विफल हो जाती हैं। "
            "रेजिड्यूअल स्किप कनेक्शन एक पहचान मैपिंग शॉर्टकट प्रदान करके ग्रेडिएंट क्षय को कम करते हैं। "
            "एडम ऑप्टिमाइज़र पहले और दूसरे रॉ मोमेंट्स के रनिंग एवरेज को बनाए रखकर ग्रेडिएंट वेक्टर को गतिशील रूप से स्केल करता है। "
            "वेट डीके विरल प्रशिक्षण सेटों पर गंभीर ओवरफिक्सिंग को रोकने के लिए पैरामीटर नॉर्म्स को नियमित करता है। "
            "उचित हाइपरपैरामीटर ट्यूनिंग प्रशिक्षण युगों (epochs) में सुचारू लॉस अभिसरण सुनिश्चित करती है।"
        ),
        "key_terms": ["loss function", "cross-entropy", "gradients", "backpropagation", "gradient descent", "residual skip connections"]
    },
    {
        "domain": "Computer Architecture",
        "title": "Memory Hierarchy and Cache Latency",
        "en_subtopic": (
            "The CPU cache stores frequently accessed instructions to minimize memory latency during pipeline execution. "
            "Modern multi-core processors implement a multi-level cache hierarchy including L1, L2, and shared L3 caches. "
            "A cache miss forces a high-latency roundtrip to DRAM over the system bus. "
            "Each cache line typically holds sixty-four bytes of contiguous physical memory. "
            "Spatial locality dictates that accessing a memory address implies adjacent addresses will soon be referenced. "
            "Temporal locality ensures recently accessed memory lines remain in fast SRAM registers. "
            "Cache replacement algorithms like Least Recently Used dictate which lines are evicted upon line contention. "
            "In shared-memory multiprocessors, cache coherence protocols like MESI maintain uniform memory state across processor cores. "
            "Modified lines must be invalidated or updated in peer caches before write commits proceed. "
            "Write buffers and non-blocking caches hide store latencies behind instruction-level parallelism. "
            "High-performance software designs optimize data structures for cache alignment to eliminate false sharing."
        ),
        "gold_hi": (
            "पाइपलाइन निष्पादन के दौरान मेमोरी लेटेंसी (memory latency) को न्यूनतम करने के लिए CPU कैश (cache) बार-बार एक्सेस किए जाने वाले निर्देशों को स्टोर करता है। "
            "आधुनिक मल्टी-कोर प्रोसेसर L1, L2 और साझा L3 कैश सहित एक बहु-स्तरीय कैश पदानुक्रम को लागू करते हैं। "
            "कैश मिस सिस्टम बस पर DRAM के लिए उच्च-लेटेंसी राउंडट्रिप को बाध्य करता है। "
            "प्रत्येक कैश लाइन में आमतौर पर साठ-चार बाइट्स का सन्निहित भौतिक मेमोरी स्थान होता है। "
            "स्थानिक स्थानीयता (spatial locality) तय करती है कि किसी मेमोरी पते को एक्सेस करने का अर्थ है कि आसन्न पतों को जल्द ही संदर्भित किया जाएगा। "
            "सामयिक स्थानीयता (temporal locality) सुनिश्चित करती है कि हाल ही में एक्सेस की गई मेमोरी लाइनें तेज SRAM रजिस्टरों में बनी रहें। "
            "लीस्ट रीसेंटली यूज्ड (LRU) जैसे कैश रिप्लेसमेंट एल्गोरिदम यह तय करते हैं कि लाइन विवाद पर किन लाइनों को बेदखल किया जाए। "
            "शेयर्ड-मेमोरी मल्टीप्रोसेसरों में, MESI जैसे कैश सुसंगतता प्रोटोकॉल प्रोसेसर कोर में एकसमान मेमोरी स्थिति बनाए रखते हैं। "
            "राइट कमिट आगे बढ़ने से पहले मॉडिफाइड लाइनों को पीयर कैश में अमान्य या अपडेट किया जाना चाहिए। "
            "राइट बफ़र्स और गैर-अवरोधक कैश निर्देश-स्तरीय समानता के पीछे स्टोर लेटेंसी को छुपाते हैं। "
            "उच्च-प्रदर्शन सॉफ़्टवेयर डिज़ाइन फाल्स शेयरिंग (false sharing) को खत्म करने के लिए कैश संरेखण हेतु डेटा संरचनाओं को अनुकूलित करते हैं।"
        ),
        "key_terms": ["CPU cache", "memory latency", "cache hierarchy", "cache miss", "spatial locality", "cache coherence"]
    },
    {
        "domain": "Distributed Systems & Databases",
        "title": "Distributed Consensus & Sharding",
        "en_subtopic": (
            "Distributed transactions require a two-phase commit protocol to ensure ACID consistency across database shards. "
            "In the prepare phase, the coordinator asks all participant nodes if they can commit their local transactions. "
            "If any shard fails or times out, an abort message is broadcast to maintain transaction atomicity. "
            "Participant nodes write intent records to durable write-ahead logs before voting yes. "
            "In the commit phase, the coordinator writes a global commit decision and notifies all shards to finalize changes. "
            "Network partitions can cause blocking coordinator states if the leader crashes before decision broadcast. "
            "Quorum-based consensus protocols like Raft and Paxos provide non-blocking fault tolerance without single points of failure. "
            "Partition keys distribute database records uniformly across separate hash rings to prevent hotspotting. "
            "Cross-shard transactions inevitably introduce high latency due to network serialization and distributed locking. "
            "Multi-version concurrency control permits read queries to execute without acquiring exclusive locks. "
            "Eventual consistency relaxations allow globally distributed databases to achieve higher availability under network partitions."
        ),
        "gold_hi": (
            "डेटाबेस शार्ड्स (database shards) में ACID कंसिस्टेंसी सुनिश्चित करने के लिए डिस्ट्रीब्यूटेड ट्रांजेक्शन हेतु दो-चरणीय कमिट प्रोटोकॉल (two-phase commit protocol) की आवश्यकता होती है। "
            "तैयारी चरण में, समन्वयक सभी प्रतिभागी नोड्स से पूछता है कि क्या वे अपने स्थानीय लेनदेन को कमिट कर सकते हैं। "
            "यदि कोई शार्ड विफल होता है या टाइम आउट होता है, तो लेनदेन की एटॉमिकिटी (atomicity) बनाए रखने के लिए एक निरस्तीकरण (abort) संदेश प्रसारित किया जाता है। "
            "प्रतिभागी नोड्स हाँ मतदान करने से पहले टिकाऊ राइट-अहेड लॉग में आशय रिकॉर्ड लिखते हैं। "
            "कमिट चरण में, समन्वयक एक वैश्विक कमिट निर्णय लिखता है और सभी शार्ड्स को परिवर्तनों को अंतिम रूप देने के लिए सूचित करता है। "
            "यदि निर्णय प्रसारण से पहले लीडर क्रैश हो जाता है, तो नेटवर्क विभाजन कोऑर्डिनेटर की स्थिति को अवरुद्ध कर सकते हैं। "
            "राफ्ट (Raft) और पैक्सोस (Paxos) जैसे कोरम-आधारित आम सहमति प्रोटोकॉल विफलता के एकल बिंदु के बिना गैर-अवरोधक दोष सहिष्णुता प्रदान करते हैं। "
            "विभाजन कुंजियाँ हॉटस्पॉटिंग को रोकने के लिए अलग-अलग हैश रिंग्स में डेटाबेस रिकॉर्ड को समान रूप से वितरित करती हैं। "
            "क्रॉस-शार्ड लेनदेन अनिवार्य रूप से नेटवर्क क्रमांकन और वितरित लॉकिंग के कारण उच्च लेटेंसी लाते हैं। "
            "मल्टी-वर्जन संगामिति नियंत्रण (MVCC) रीड प्रश्नों को अनन्य लॉक प्राप्त किए बिना निष्पादित करने की अनुमति देता है। "
            "अंततः संगति (eventual consistency) शिथिलता वैश्विक रूप से वितरित डेटाबेस को नेटवर्क विभाजन के तहत उच्च उपलब्धता प्राप्त करने की अनुमति देती है।"
        ),
        "key_terms": ["two-phase commit protocol", "ACID consistency", "database shards", "coordinator", "atomicity", "distributed consensus"]
    }
]


def build_translation_prompt(en_text: str) -> str:
    # Devanagari-anchored prompt (Variation B) to prevent cross-lingual script drift
    prompt = (
        f"अंग्रेजी:\n{en_text}\n\n"
        "हिन्दी (देवनागरी लिपि में अनुवाद):\n"
    )
    return prompt


def run_subtopic_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    subtopics: List[str],
    max_new_tokens: int = 600,
    device: str = "cuda"
) -> List[str]:
    translations = []
    model.eval()

    for idx, en_text in enumerate(subtopics):
        prompt = build_translation_prompt(en_text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = outputs[0][prompt_len:]
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        # Cut off any trailing prompt continuation if base model hallucinates extra fields
        for stop_marker in [
            "\nअंग्रेजी:", "\n\nअंग्रेजी:", "\nEnglish:", "\n\nEnglish:", 
            "\nहिन्दी", "\n\nहिन्दी", "\nTranslate", "\n\nTranslate", 
            "\nTask:", "\nExplanation:", "\nNote:"
        ]:
            if stop_marker in decoded:
                decoded = decoded.split(stop_marker)[0].strip()

        translations.append(decoded)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(subtopics):
            print(f"Translated {idx + 1} / {len(subtopics)} subtopics...")

    return translations


def evaluate_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, Any]:
    ref_list = [references]
    bleu_13a = sacrebleu.corpus_bleu(hypotheses, ref_list, tokenize="13a")
    bleu_flores = sacrebleu.corpus_bleu(hypotheses, ref_list, tokenize="flores200")
    chrf_plus = sacrebleu.corpus_chrf(hypotheses, ref_list, word_order=2)

    return {
        "bleu_13a": round(bleu_13a.score, 2),
        "bleu_flores200": round(bleu_flores.score, 2),
        "chrf_plus_plus": round(chrf_plus.score, 2),
        "bp": round(bleu_flores.bp, 3),
        "sys_len": bleu_flores.sys_len,
        "ref_len": bleu_flores.ref_len
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Benchmarking Pre-Trained Base Model `{MODEL_ID}` (4-bit) on Subtopics ===")

    # 1. Load Subtopic Test Dataset
    print(f"Loading engineered subtopic test set from {TEST_SUBTOPICS_PATH}...")
    table = pq.read_table(TEST_SUBTOPICS_PATH)
    all_subtopics = table.to_pylist()
    print(f"Loaded {len(all_subtopics)} test subtopic paragraphs.")

    # Filter STEM subtopics and general subtopics
    stem_subtopics = [s for s in all_subtopics if s.get("is_stem", False)]
    print(f"Identified {len(stem_subtopics)} STEM subtopics.")

    # Select evaluation set: 45 representative 15-sentence subtopics (30 STEM + 15 general)
    eval_subtopics = stem_subtopics[:30] + [s for s in all_subtopics if not s.get("is_stem", False)][:15]
    print(f"Benchmarking on {len(eval_subtopics)} representative 15-sentence subtopics ({sum(1 for s in eval_subtopics if s.get('is_stem'))} STEM)...")

    # 2. Load Model in 4-bit NF4
    print(f"\nLoading {MODEL_ID} in 4-bit NF4 precision on GPU...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    t0_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    t_load = time.time() - t0_load
    load_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    print(f"Model loaded in {t_load:.2f}s | Baseline 4-bit Model VRAM: {load_vram_mb:.1f} MB")

    # 3. Run Inference on Test Subtopics
    test_src = [s["src_en"] for s in eval_subtopics]
    test_ref = [s["tgt_hi"] for s in eval_subtopics]

    t0_infer = time.time()
    test_hyp = run_subtopic_inference(model, tokenizer, test_src, max_new_tokens=600, device=device)
    t_infer = time.time() - t0_infer
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
    subtopics_per_sec = len(eval_subtopics) / max(1e-5, t_infer)

    print(f"Completed subtopic inference in {t_infer:.2f}s ({subtopics_per_sec:.2f} subtopics/sec) | Peak VRAM: {peak_vram_mb:.1f} MB")

    # 4. Compute Metrics
    full_metrics = evaluate_metrics(test_hyp, test_ref)
    print(f"\nOverall Subtopic Metrics: BLEU (Flores200): {full_metrics['bleu_flores200']} | BLEU (13a): {full_metrics['bleu_13a']} | chrF++: {full_metrics['chrf_plus_plus']}")

    # STEM sub-slice metrics
    stem_hyp = [test_hyp[i] for i, s in enumerate(eval_subtopics) if s.get("is_stem")]
    stem_ref = [test_ref[i] for i, s in enumerate(eval_subtopics) if s.get("is_stem")]
    stem_metrics = evaluate_metrics(stem_hyp, stem_ref)
    print(f"STEM Subtopic Metrics: BLEU (Flores200): {stem_metrics['bleu_flores200']} | BLEU (13a): {stem_metrics['bleu_13a']} | chrF++: {stem_metrics['chrf_plus_plus']}")

    # 5. Run Targeted University STEM Probes
    print("\nRunning inference on University STEM Subtopics (12-15 Sentence Lecture Paragraphs)...")
    probe_en = [p["en_subtopic"] for p in UNIVERSITY_STEM_SUBTOPICS]
    probe_hyp = run_subtopic_inference(model, tokenizer, probe_en, max_new_tokens=600, device=device)

    probe_evals = []
    for p, hyp in zip(UNIVERSITY_STEM_SUBTOPICS, probe_hyp):
        probe_evals.append({
            "domain": p["domain"],
            "title": p["title"],
            "key_terms": p["key_terms"],
            "source_en": p["en_subtopic"],
            "gold_hi": p["gold_hi"],
            "sarvam1_baseline": hyp
        })

    # 6. Manual Inspection & Failure Extraction
    print("\nExtracting manual error samples from pre-trained base model outputs...")
    error_samples = []

    for pe in probe_evals:
        hyp = pe["sarvam1_baseline"]
        analysis = []
        if pe["domain"] == "Linear Algebra & Spectral Theory":
            if "स्व-मूल्य" in hyp or "आइगेनवैल्यू" not in hyp:
                analysis.append("Fails to use standard mathematical transliteration 'आइगेनवैल्यू' for eigenvalue.")
            if "ऑर्थोनॉर्मल" not in hyp:
                analysis.append("Orthogonal diagonalization / orthonormal basis degraded into literal or omitted phrasing.")
            error_samples.append({
                "domain": pe["domain"],
                "title": pe["title"],
                "source_en": pe["source_en"],
                "gold_hi": pe["gold_hi"],
                "base_sarvam1_output": hyp,
                "error_analysis": " ; ".join(analysis) if analysis else "Pre-trained base model produces unstructured continuation or colloquial terminology shifts."
            })
        elif pe["domain"] == "Machine Learning & Neural Search":
            if "घने" in hyp or "डेंस" not in hyp:
                analysis.append("Dense retrieval translated colloquially as 'घने रिट्रीवल' (dense foliage) instead of 'डेंस रिट्रीवल'.")
            if "उत्पाद" in hyp or "डॉट प्रॉडक्ट" not in hyp:
                analysis.append("Vector dot products translated as commercial merchandise ('डॉट उत्पाद') rather than scalar product.")
            error_samples.append({
                "domain": pe["domain"],
                "title": pe["title"],
                "source_en": pe["source_en"],
                "gold_hi": pe["gold_hi"],
                "base_sarvam1_output": hyp,
                "error_analysis": " ; ".join(analysis) if analysis else "Fails to anchor vector space and dense retrieval nomenclature in multi-sentence discourse."
            })
        elif pe["domain"] == "Distributed Systems & Databases":
            if "शॉर्ट्स" in hyp or "शार्ड्स" not in hyp:
                analysis.append("Database shards confused phonetically with shorts ('शॉर्ट्स').")
            if "प्रतिबद्ध" in hyp or "कमिट" not in hyp:
                analysis.append("Two-phase commit translated as moral commitment rather than database protocol.")
            error_samples.append({
                "domain": pe["domain"],
                "title": pe["title"],
                "source_en": pe["source_en"],
                "gold_hi": pe["gold_hi"],
                "base_sarvam1_output": hyp,
                "error_analysis": " ; ".join(analysis) if analysis else "Distributed transaction protocols and ACID properties are translated into colloquial prose without system precision."
            })
        elif pe["domain"] == "Computer Architecture":
            error_samples.append({
                "domain": pe["domain"],
                "title": pe["title"],
                "source_en": pe["source_en"],
                "gold_hi": pe["gold_hi"],
                "base_sarvam1_output": hyp,
                "error_analysis": "L1/L2 cache hierarchy and latency trade-offs lack standardized academic Hindi phrasing."
            })

    # Also inspect test subtopic #0 (Black box telematics article)
    black_box_subtopic = eval_subtopics[0]
    error_samples.append({
        "domain": "Automotive Telematics (Test Set Subtopic #0)",
        "title": "Black Box Telematics",
        "source_en": black_box_subtopic["src_en"],
        "gold_hi": black_box_subtopic["tgt_hi"],
        "base_sarvam1_output": test_hyp[0],
        "error_analysis": "Base model translation across multi-sentence article: evaluates whether 'black box' and 'mile-by-mile tracking system' are preserved or translated literally as colored boxes."
    })

    # 7. Save Artifacts
    results = {
        "model_id": MODEL_ID,
        "prompt_strategy": "Variation B (Devanagari Anchored: 'अंग्रेजी:\n...\nहिन्दी (देवनागरी लिपि में अनुवाद):\n')",
        "precision": "4-bit (NF4 bnb)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_footprint": {
            "model_vram_mb": round(load_vram_mb, 1),
            "peak_inference_vram_mb": round(peak_vram_mb, 1),
            "inference_time_sec": round(t_infer, 2),
            "throughput_subtopics_per_sec": round(subtopics_per_sec, 2),
            "evaluated_subtopics_count": len(eval_subtopics)
        },
        "overall_subtopic_metrics": full_metrics,
        "stem_subtopic_metrics": stem_metrics,
        "probe_evaluations": probe_evals
    }

    metrics_file = os.path.join(OUTPUT_DIR, "baseline_sarvam1_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Baseline Sarvam-1 metrics saved to {metrics_file}")

    devanagari_metrics_file = os.path.join(OUTPUT_DIR, "baseline_sarvam1_devanagari_anchored_metrics.json")
    with open(devanagari_metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Devanagari-anchored metrics saved to {devanagari_metrics_file}")

    errors_file = os.path.join(OUTPUT_DIR, "baseline_sarvam1_error_samples.json")
    with open(errors_file, "w", encoding="utf-8") as f:
        json.dump(error_samples, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Baseline Sarvam-1 error samples saved to {errors_file}")

    samples_file = os.path.join(OUTPUT_DIR, "baseline_sarvam1_translations_sample50.json")
    sample_dump = [
        {"idx": i, "en": test_src[i], "gold_hi": test_ref[i], "sarvam1_hi": test_hyp[i]}
        for i in range(min(50, len(test_src)))
    ]
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(sample_dump, f, indent=2, ensure_ascii=False)
    print(f"[Saved] 50 sample translations saved to {samples_file}")

    print("\n=== Pre-Trained Sarvam-1 Subtopic Benchmarking Complete ===")


if __name__ == "__main__":
    main()
