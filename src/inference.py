import torch
from transformers import LEDForConditionalGeneration, LEDTokenizerFast
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training
import os
from peft import PeftModel
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
import math
import re

MAX_INPUT = 16384
MAX_TARGET = 516

class AttentionPooling(nn.Module):
    def __init__(self, d_hidden: int, temperature: float = 1.0):
        super().__init__()
        self.scorer = nn.Linear(d_hidden, 1)
        nn.init.normal_(self.scorer.weight, std=0.02)
        if self.scorer.bias is not None:
            nn.init.zeros_(self.scorer.bias)
        self.temperature = float(temperature)

    def forward(self, tokens: torch.Tensor):
        # tokens: (T, d)
        if tokens.numel() == 0:
            return tokens.new_zeros(tokens.size(-1))
        scores = self.scorer(tokens).squeeze(-1)  # (T,)
        scores = scores / max(self.temperature, 1e-6)
        weights = F.softmax(scores, dim=0)       # (T,)
        pooled = (weights.unsqueeze(-1) * tokens).sum(dim=0)
        return pooled


class HierarchicalAdapter(nn.Module):

    def __init__(self, d_model:int, d_bottleneck:int=256, pooling:str="attn", tau:float=0.1, use_proj=True):
        super().__init__()
        assert pooling in ("attn","mean")
        self.d_model = d_model
        self.d_bottleneck = d_bottleneck
        self.tau = tau
        self.pooling = pooling

        self.W_down = nn.Linear(d_model, d_bottleneck)
        self.layernorm = nn.LayerNorm(d_bottleneck)
        self.activation = F.relu

        if pooling == "attn":
            self.attn_pool = AttentionPooling(d_bottleneck, temperature=1.0)
        else:
            self.attn_pool = None

        if use_proj:
            self.W_sim_q = nn.Linear(d_bottleneck, d_bottleneck, bias=True)
            self.W_sim_k = nn.Linear(d_bottleneck, d_bottleneck, bias=True)
        else:
            self.W_sim_q = None
            self.W_sim_k = None

        self.W_sim_v = nn.Linear(d_bottleneck, d_bottleneck, bias=True)
        self.W_up = nn.Linear(d_bottleneck, d_model)
        if self.W_up.bias is not None:
            nn.init.zeros_(self.W_up.bias)

    def _pool_span(self, span_tokens: torch.Tensor):
        # span_tokens: (T, d_b)
        if span_tokens.size(0) == 0:
            return span_tokens.new_zeros(self.d_bottleneck)
        if self.pooling == "attn":
            return self.attn_pool(span_tokens)
        return span_tokens.mean(dim=0)

    def forward(self, hidden_states: torch.Tensor, utterance_boundaries: Optional[List[List[Tuple[int,int]]]] = None):

        B, L, _ = hidden_states.shape
        device = hidden_states.device
        if utterance_boundaries is None:
            utterance_boundaries = [[(0, L)] for _ in range(B)]

        # 1. down project
        h_down = self.W_down(hidden_states)  # (B, L, d_b)
        h_down = self.layernorm(h_down)
        h_down = self.activation(h_down)

        # 2. build utt vectors
        utt_vectors = []
        mask_list = []
        for b in range(B):
            spans = utterance_boundaries[b] if len(utterance_boundaries[b])>0 else [(0, L)]
            per_utt = []
            for (s,e) in spans:
                s_c = max(0, min(s, L))
                e_c = max(s_c, min(e, L))
                if e_c <= s_c:
                    per_utt.append(torch.zeros(self.d_bottleneck, device=device))
                else:
                    span_tokens = h_down[b, s_c:e_c]  # (T, d_b)
                    pooled = self._pool_span(span_tokens)
                    per_utt.append(pooled)
            if len(per_utt) == 0:
                uv = torch.zeros(1, self.d_bottleneck, device=device)
                mask = torch.tensor([0], device=device)
            else:
                uv = torch.stack(per_utt, dim=0)  # (U_b, d_b)
                mask = torch.ones(uv.size(0), device=device)
            utt_vectors.append(uv)
            mask_list.append(mask)

        # 3. pad to max_U
        max_U = max(uv.size(0) for uv in utt_vectors)
        padded_uv = []
        masks = []
        for uv, mask in zip(utt_vectors, mask_list):
            U = uv.size(0)
            if U < max_U:
                pad = torch.zeros(max_U-U, self.d_bottleneck, device=device)
                padded = torch.cat([uv, pad], dim=0)
                m = torch.cat([mask, torch.zeros(max_U-U, device=device)], dim=0)
            else:
                padded = uv
                m = mask
            padded_uv.append(padded)
            masks.append(m)
        utterance_tensor = torch.stack(padded_uv, dim=0)  # (B, max_U, d_b)
        mask_tensor = torch.stack(masks, dim=0).to(utterance_tensor.device)  # (B, max_U)

        # 4. similarity / attention
        Q = self.W_sim_q(utterance_tensor) if self.W_sim_q is not None else utterance_tensor
        K = self.W_sim_k(utterance_tensor) if self.W_sim_k is not None else utterance_tensor
        V = self.W_sim_v(utterance_tensor)

        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d_bottleneck)  # (B, max_U, max_U)
        # avoid fp16 overflow: use safe min for dtype
        min_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask_tensor.unsqueeze(1)==0, min_val)
        attn = F.softmax(scores / max(self.tau, 1e-6), dim=-1)
        context = torch.matmul(attn, V)  # (B, max_U, d_b)

        # 5. scatter back to tokens (additive)
        h_up_tokens = h_down.clone()
        for b in range(B):
            spans = utterance_boundaries[b]
            U_b = min(len(spans), context.size(1))
            for u in range(U_b):
                s,e = spans[u]
                s_c = max(0, min(s, L))
                e_c = max(s_c, min(e, L))
                if e_c <= s_c:
                    continue
                ctx = context[b, u]  # (d_b,)
                h_up_tokens[b, s_c:e_c] = h_up_tokens[b, s_c:e_c] + ctx.unsqueeze(0)

        # 6. up-project & resid
        h_out = self.W_up(h_up_tokens)  # (B, L, d_model)
        return hidden_states + h_out



def attach_hierarchical_adapters_to_led(model: LEDForConditionalGeneration, d_bottleneck=256, pooling='attn', tau=0.1):

    if hasattr(model, "led") and hasattr(model.led, "encoder"):
        encoder = model.led.encoder
    elif hasattr(model, "model") and hasattr(model.model, "led") and hasattr(model.model.led, "encoder"):
        encoder = model.model.led.encoder
    elif hasattr(model, "model") and hasattr(model.model, "encoder"):
        encoder = model.model.encoder
    else:
        raise RuntimeError("Cannot locate LED encoder")

    for i, layer in enumerate(encoder.layers):
        # attach module named hierarchical_adapter
        adapter = HierarchicalAdapter(d_model=model.config.d_model, d_bottleneck=d_bottleneck, pooling=pooling, tau=tau)
        # register as submodule
        layer.add_module("hierarchical_adapter", adapter)



def apply_lora_peft_to_led(model: LEDForConditionalGeneration, r=8, alpha=32, dropout=0.0, target_modules=None):

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "k_proj_global", "v_proj_global"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    peft_model = get_peft_model(model, lora_config)
    return peft_model

def load_peft_and_hier(base_model: LEDForConditionalGeneration, checkpoint_dir: str):

    model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=False)

    hier_path = os.path.join(checkpoint_dir, "hier_adapter.pth")
    if os.path.exists(hier_path):
        hier_state = torch.load(hier_path, map_location="cpu")
        missing = []
        unexpected = []
        model_state = model.state_dict()
        to_load = {k: v for k, v in hier_state.items() if k in model_state}

    return model



def build_utterance_boundaries_from_segments(segments, tokenizer, max_length=MAX_INPUT):
    tokenized = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in segments]
    input_ids = []
    boundaries = []
    start = 0
    for ids in tokenized:
        if len(ids) == 0:
            continue
        end = start + len(ids)
        if end > max_length:
            remaining = max_length - start
            if remaining <= 0:
                break
            ids = ids[:remaining]
            end = start + len(ids)
            input_ids.extend(ids)
            boundaries.append((start, end))
            break
        input_ids.extend(ids)
        boundaries.append((start, end))
        start = end
    return input_ids, boundaries


def tokenize_example_with_segments_for_map(example, tokenizer, max_input=MAX_INPUT, max_target=MAX_TARGET):

    segments = []
    if "transcript_segments" in example and isinstance(example["transcript_segments"], dict):
        segments = example["transcript_segments"].get("text", [])
    if not segments and "transcript" in example:
        segments = [example["transcript"]]

    input_ids, boundaries = build_utterance_boundaries_from_segments(segments, tokenizer, max_length=max_input)
    if len(input_ids) == 0 and "transcript" in example:
        enc = tokenizer(example["transcript"], add_special_tokens=False, truncation=True, max_length=max_input)
        input_ids = enc["input_ids"]
        boundaries = [(0, len(input_ids))] if len(input_ids) > 0 else []

    target_text = example.get("summary") or example.get("abstract") or example.get("summary_text") or ""
    label_enc = tokenizer(target_text, truncation=True, max_length=max_target, padding=False)
    labels = label_enc["input_ids"]
    return {"input_ids": input_ids, "labels": labels, "utterance_boundaries": boundaries}

import globals_store
class CollatorWithUtteranceBoundaries:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):

        globals_store.GLOBAL_UTTERANCE_BOUNDARIES = [ex.get("utterance_boundaries", [(0, len(ex.get("input_ids", [])))]) for ex in batch]

        input_ids_list = [ex.get("input_ids", []) for ex in batch]
        labels_list = [ex.get("labels", []) for ex in batch]

        max_len = max((len(x) for x in input_ids_list), default=0)
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        padded_inputs = [ids + [self.tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids_list]
        attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]

        max_label_len = max((len(l) for l in labels_list), default=0)
        padded_labels = [l + [-100] * (max_label_len - len(l)) for l in labels_list]

        batch_output = {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        return batch_output




def load_model(
    base_model_name="allenai/led-base-16384",
    adapter="hier",
    peft_dir=r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\models\led-tib-lora\checkpoint-9000",
    hier_dir=r"B:\stacks\Machine_learning_projects\Video_Summarization_Translation\models\led-tib-hier\led_hier_adapter_run",
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LEDTokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    base = LEDForConditionalGeneration.from_pretrained(base_model_name)
    
    if(adapter=="lora"):
        model = PeftModel.from_pretrained(base, peft_dir)
    else:
        attach_hierarchical_adapters_to_led(base, d_bottleneck=256, pooling="attn", tau=0.1)
        model = load_peft_and_hier(base, hier_dir)

    model.to(device)
    model.eval()
    return model, tokenizer



def split_into_segments(text: str, tokenizer, target_seg_tokens: int = 200):

    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    segments = []
    cur_seg = ""
    cur_len = 0
    for sent in sentences:
        enc = tokenizer(sent, add_special_tokens=False)
        sent_len = len(enc["input_ids"])
        if cur_len + sent_len > target_seg_tokens and cur_seg:
            segments.append(cur_seg.strip())
            cur_seg = sent
            cur_len = sent_len
        else:
            if cur_seg:
                cur_seg = cur_seg + " " + sent
            else:
                cur_seg = sent
            cur_len += sent_len
    if cur_seg:
        segments.append(cur_seg.strip())
    return segments


def _prepare_encoder_inputs_from_segments(segments: List[str], tokenizer, device, max_input=MAX_INPUT):

    input_ids, boundaries = build_utterance_boundaries_from_segments(segments, tokenizer, max_length=max_input)
    if len(input_ids) == 0:
        return None

    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)  # (1, L)
    attention_mask = torch.tensor([[1] * len(input_ids)], dtype=torch.long, device=device)  # (1, L)
    return {"input_ids": input_ids_tensor, "attention_mask": attention_mask, "boundaries": [boundaries]}


def generate_summary(
    transcript,
    model=None,
    tokenizer=None,
    adapter="hier",
    batch_size=4,
    device=None,
    min_length=50,
    max_target=MAX_TARGET,
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if tokenizer is None or model is None:
        raise ValueError("Please pass model and tokenizer (use load_model(...))")

    single_input = False
    if isinstance(transcript, str):
        transcripts = [transcript]
        single_input = True
    elif isinstance(transcript, (list, tuple)):
        transcripts = list(transcript)
    else:
        raise ValueError("transcript must be str or list of str")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    summaries = []

    for start in range(0, len(transcripts), batch_size):
        batch_texts = transcripts[start : start + batch_size]

        if adapter == "hier":
            enc_inputs = []
            boundaries_batch = []
            for txt in batch_texts:
                segments = split_into_segments(txt, tokenizer, target_seg_tokens=200)
                if not segments:
                    segments = [txt]
                prepared = _prepare_encoder_inputs_from_segments(segments, tokenizer, device, max_input=MAX_INPUT)
                if prepared is None:
                    enc_inputs.append(None)
                    boundaries_batch.append([])
                else:
                    enc_inputs.append(prepared)
                    boundaries_batch.append(prepared["boundaries"][0])

            lens = [inp["input_ids"].size(1) if inp is not None else 0 for inp in enc_inputs]
            max_len = max(lens) if lens else 0
            if max_len == 0:
                summaries.extend([""] * len(batch_texts))
                continue

            input_id_tensors = []
            attention_masks = []
            for inp in enc_inputs:
                if inp is None:
                    pad_ids = torch.full((1, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
                    pad_mask = torch.zeros((1, max_len), dtype=torch.long, device=device)
                    input_id_tensors.append(pad_ids)
                    attention_masks.append(pad_mask)
                else:
                    L = inp["input_ids"].size(1)
                    if L < max_len:
                        pad_len = max_len - L
                        pad_ids = torch.cat([inp["input_ids"], torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long, device=device)], dim=1)
                        pad_mask = torch.cat([inp["attention_mask"], torch.zeros((1, pad_len), dtype=torch.long, device=device)], dim=1)
                    else:
                        pad_ids = inp["input_ids"]
                        pad_mask = inp["attention_mask"]
                    input_id_tensors.append(pad_ids)
                    attention_masks.append(pad_mask)

            batch_input_ids = torch.cat(input_id_tensors, dim=0)        
            batch_attention_mask = torch.cat(attention_masks, dim=0)    

            globals_store.GLOBAL_UTTERANCE_BOUNDARIES = boundaries_batch

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_length=max_target,
                    num_beams=4,
                    length_penalty=1.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.5,
                    encoder_no_repeat_ngram_size=3,
                    min_length=min_length,
                )

            decoded = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            summaries.extend(decoded)

        elif adapter == "lora":
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=MAX_INPUT,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                generated_ids = model.generate(
                    **enc,
                    max_length=max_target,
                    num_beams=5,
                    length_penalty=1.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=2.5,
                    encoder_no_repeat_ngram_size=3,
                    min_length=min_length,
                )

            decoded = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            summaries.extend(decoded)

        else:
            raise ValueError(f"Unknown adapter type: {adapter!r}. Use 'hier' or 'lora'.")

    if single_input:
        return summaries[0] if summaries else ""
    return summaries


# model, tokenizer = load_model()

# summary = generate_summary(
#     "All right, so when you walked in, you saw this. That's me, that's Justin Searles, but you can see how it's written by hand on an index card. There's a story behind that. This is not my talk, which is part of why I'm so nervous, but really, please don't leave, you're in the right place, at least. So you just stay where you are and I'm going to do my best despite this. This is actually not acceptable. I am being trolled so hard. Okay, so this is not a bait and switch. I've spoken at RailsCon two times before and I intentionally wrote abstracts to get into the CFP and then I talked about what I wanted to talk about. So this is not a bait and switch like those two. It wasn't intentionally that, it is now. This was supposed to be Sam Thippen's talk. Everyone go follow Sam, he's great. Sadly, Sam is in the hospital, so he wasn't able to give this talk and that's why I'm here instead, but it's a British hospital, so he's just in hospital. So send your good wishes to Sam. Why me, why am I here? Well, Sam likes to give conference presentations wearing my company's branded t-shirt, Testdouble, and so people are often mistaking him for one of our employees such that he actually now has intro slides like, I do not work for Testdouble, but I love them and also Surls, which I heartily appreciate. We love you too, Sam. That's why we're here. You're a great member of the community. So this talk's going to be fipping great. Only problem is I finally understand imposter syndrome. So I've got a little bit of imposter syndrome because I am a literal imposter today. In three main categories, one, I am not British, and as we all know as Americans, of those of us in the room here are American, everything out of British people's mouths sounds a lot more intelligent, so I have that shortcoming. And therefore today I resolved to use fewer contractions to speak with authority and to drop the rotic r. So let's practice the sentence together. Mini-test is not better than R-spec. All right. I feel better already. Two, I lack Sam's glorious mane. I don't have a big bushy beard. Sam of course drives his R-spec powers from his beard. This is obvious because why else would he have it? So I have not shaved since I agreed to do this at 7 a.m. Friday morning. Some straggles. So I now know a few things based on the R-spec beard powers. One, beards are itchy. Two, R-spec. And three, what beard oil is. So if anyone, I forgot my razor, true story, if anyone has some beard oil on them, hook a brother up. Third thing, third way in which I am an imposter today, I am not on R-spec core. Here's a little like organizational chart of where I fit in to R-spec. That's R-spec core and that's me not being in it. But you know what, apparently it's just not a RailsConf without a talk from an R-spec committer about R-spec. So far to date, the only R-spec thing I've committed to is this talk. So I decided to become an R-spec committer. It sounds like a good idea. So let's get started. I'm going to make my first R-spec commit right here. I am so committed right now to R-spec. All right, so I'm just going to push it up. Access denied. So I tried everything earlier in the hotel. So let's try it one more time. It always works. You know what, you get this error message also when you get hubs down. So it's probably just to get hubs down. So as this talk's resident R-spec committer, I have some startling announcements to make. I'm here ready to announce the future of R-spec for you today. Current version of R-spec is 3.4.0. I'm here to announce the next major release of R-spec, R-spec 5. R-spec 5 is going to be revolutionary because we have some really awesome headline features that are very convenient to me and my purposes. The first, Turbo-spec. Let me tell you about Turbo-spec. Yep. Turbo-spec dumps the object space into the cache, into memory after running every single one of your before hooks. It does this so that it can cache each nested example group setup code so that you don't have to run it across all your tests. And then if you run the R-spec CLI with tack tack turbo button, it speeds up your tests. Turbo-spec is going to make all of our slow R-spec suites way faster. Warning, it doesn't work if your application has side effects. But for the rest of us, it's going to be just awesome. I have another feature for R-spec 5 that I think is going to really just make true believers of R-spec happy. Specspecs. You just create a spec of type spec and then you can say things like, hey, this model order, I expect it to have five specs. I expect order to finish within about two hours to have 95% code coverage, to limit the nesting and indentation to just three contexts, to usually pass, and to be good code. I don't know why they didn't have this in R-spec 3, it's in R-spec 5. Remember, it's important to spec spec your spec specs, people. It's not get lazy. Obama's saying things. Audio doesn't work anymore because of their shenanigans. Let's try one more time. All right, what he said was just in just give it a rest. Damn it. I'm going to be, now I'm not going to sleep tonight. So thanks, audio. All right. So I'm still anyway, regardless. I'm not sure if I'm cured or if I'm still impostering. I am not Sam. If you don't know me, this is what I look like on Twitter when I'm getting retweeted for saying terrible things, that's me, Searle's. I'd love if you became my Twitter friend and got me some feedback about how things are going. I know it's not great so far. This is the Justin Searle's marriage simulator. Basically it's just you sitting across the table with me looking at my phone and making slanted faces. So we can all empathize with Becky Joy a little better. This is me on brand. I help run a software agency called TestDouble. Our mission is audacious. We're just trying to make the world software less awful. And I'd love if you got us any feedback. Hello at TestDouble.com. All right. So again, talk title, back to basics. RSpec, Rails 5. What's there to know? By the way, sidebar, did you know Sam rejected my Rails talk? I just thought I should mention that because I am supposedly honor bound to cover all this Rails 5 stuff because it's important to cover for the purpose of the program, which I took with just nothing but grace. So Rails 5 stuff. My first question to Sam via text message on Friday morning was, will RSpec just work with Rails 5? No. And he was saying it as an implementer. He's thinking about all the work they needed to do because obviously if you've ever maintained a gem, news slash major Rails releases break gems in surprising and myriad ways. I went and searched for just open GitHub issues that are demanding Rails 5 support. Just search for it and you get a whole lot of salty randos saying, hey, Rails 5 is not supported. No description. Give me Rails 5. You owe me. Come on. Gems. Work. Work. Give me. Rails 5 is not even out yet, people. So if you know a maintainer, go give a maintainer a hug because seriously, Rails major release upgrades are big work. RSpec considers this to be feature work. They don't want to make any breaking changes. They want you to be able to upgrade very gracefully. That's why they respect Semver as much as I don't. They're at 3.40 now. It's going to be 3.5.0, which means that they have to keep it running for older versions of Rails but also new versions of Rails. So I hope that you take a moment to thank the RSpec team for their thankless work because everything that they're doing here is behind the scenes. But there is one change that we all have to know about, which is, is it true that functional tests and controller specs are really deprecated? Well, yes, it actually is true. They're going away with Rails 5. They're deprecated, at least soft deprecated. To which I say finally. If you don't write controller specs, by the way, feel free to just play with your phone for this portion of the talk. If you do, it all started when DHH opened this issue saying, you know, the mechanism for making, verifying that you assigned a particular instance variable and a controller, making sure that a particular template was going to get rendered, those are testing implementation, those aren't really valuable. Let's deprecate functional tests. And I feel like he was absolutely right. That was a really good point. And of course, if you disagree, you might disagree just because you write controller specs, but here's my beef with controller specs. This is the testing pyramid here. At the top of the testing pyramid, it's just a way to illustrate these are full stack tests that call through everything in reality. And stuff at the bottom, these are just unit tests. Stuff in the middle are difficult to explain tests. And that's what controller specs are. So the problem, right? The opportunity. Oh, my gosh. All right. I'm so glad to be one of those like just chill, go with the flow kind of guys. All right. So the problem with controller specs at this level is that above that point in the pyramid, there are untestable things that can break. So there are only of limited value. And everything below it, the messages that you get are going to be unclear reasons why things are going to fail because it might be something way, way deep below you. That is actually the root cause of the failure and the error messages aren't going to be very helpful. So it helps you in that very skinny way, but I don't know how much value that really adds. Another thing about controller specs that sucks is that they were a lie to begin with. Their API implies that a request is being made. So if you've got a controller, you do like get index like you're actually making an HTTP request and then you have these assertions like you render this template or you redirect it or you have this HTTP status. Oh, look, I'm making a request. Wrong. It's just like that's just really silly sugar of a facade and it's just invoking your controller methods, which means all this other stuff is not happening like middleware is not getting invoked. So your controller specs might be passing when your controller is totally busted, but they're faster and that's why they exist. And they might be faster at runtime, but in my experience they're much slower at fixed time. They're just a maintenance nightmare for all that no value that they provide. So but you know, despite the criticism controller specs, it's semver, right? So our spec is promising not to break our tests with Rails 5. The way that we are doing that, the way that you do that, all that you have to do is add this gem to your gem file called Rails controller testing, which will reintroduce the functional testing bits that our spec Rails needs. And then meanwhile, the RSpec team is doing the hard work to make it seamless. It's my understanding Sam Pippin's doing a lot of that work. And I hope that's not what put him in hospital. So thanks to Sam and the RSpec Core team. If you already have a lot of controller specs, stop writing those now. There's stuff that you can do instead of controller specs in the future. Here's some alternatives. One, you could write full stack feature tests that test that everything's fully working when everything's really integrated. You could also do nothing. I do nothing. I have not written a controller spec for seven years. And you could also do request specs, which are very similar. We'll talk about those in a second. Because request specs are like honest versions of controller specs. They bind to, they map to many test integration tests in Rails. And the reason that they're honest is that the API looks the same and the assertions look the same except it actually exercises the routing, the middlewares and the views. So if something blows up, you know, it's a good blow up. Another cool thing is because it's using RackTest, you have access to the response body and you can make assertions on the actual response that's generated instead of all this weird implementation stuff. When to use request specs instead of controller specs or nothing. Specs that assert like a complete API response, like if you've got like a JSON API and you can assert everything that it does. Cool. Request specs are probably the right layer to test at. Specs that just assert, you're assigning certain IVars or rendering certain templates. Just needlessly coupled to the implementation, probably don't need a request spec. Specs that assert HTML that comes out of the response body, probably not a good idea unless your app has absolutely no JavaScript, which is probably unlikely. So that's a bit about request specs controller specs. Third bit. It was in the abstract, right, that we're going to learn how to test action cables. So does our spec help us test action cable? No. It turns out that action cable testing isn't built into Rails yet. There's an open pull request and I assume that when that ships, our spec will have a wrapper for it or something. So just test through the browser for now and make sure your website works. All right. There we go. You're now ready to our spec with Rails 5. Thanks very much, Sam, for trusting me with your talk. There's nothing more for you to see here. You can close Skype, Sarah. There's nothing, I think he's actually like maybe here. I think I see him waving actually. Hi, Sam. Yeah. He just looks excited. Yep. All right. Bye, Sam. So one time, Aaron Patterson's up in the front row. One time I texted Aaron something and he tweeted it and got a million retweets and I felt really salty about that because I was like, no, that was my random internet meme that I copied and pasted. And he sent me this in response. It's not fundamental attribution error. It's internet attribution error. So this is my talk. Our spec in Rails 5. Why are you here? Really? Like shout it out. Somebody tell me why you're here. No. Next question. Somebody sent. So why'd you come to this talk, especially if you didn't know who I was? Okay. Something R spec related, anyone? What's that? Okay, thanks. All right. Thank you. Action cable. Thank you. R spec cable. Well, I had two theories because I couldn't make the slides after asking you. One, how the hell do I test action cable? Sorry for those people because I don't know. Two, I'm not happy with my test suite. And now I have a third theory too. You know, I'm new here and what the hell is all this about because it's just like a lot of forensics and who are these people? I'll focus on the one that I can actually address which is what happens when we're still not happy with our test suites. Well, if you have this motivation and that's part of why you came to this talk, maybe you were thinking like, well, R spec might have a new feature that'll help me hate my test less or maybe Rails has some new thing or removes a new thing that will help make the pain stop, make my test suite more sane. I think that's a natural thing to do, especially when you're in a conference, we're here to learn about technology. We're searching for tools and tools are easy because we can grab them off a shelf and use them but they're way easier than like critical introspection, asking ourselves hard questions like maybe it's our fault that we have terrible tests. There's two keys to happiness with testing or anything in software. One, the tools that we use. Two, how we use those tools. And it's not a two-step recipe. There's, it's like not a false, it is a false dichotomy to like blame one side or the other. Some people will say like, oh well, clearly we just need better tools whenever we have a problem and some people have a disposition that says, well no, we just have to think differently. We have to design harder. Like if the tools failing us, we're not using it hard enough. And that's not a good mental model either. I like to think of it as like first there were people thinking and they were doing stuff and then that they wrote tools to help them do their job and then the tools are actually a usage of them informs how we think about the problem and it's this hopefully virtuous cycle, this feedback loop. So I do believe that tools matter. Tools aren't everything but tools are important and we're going to talk about how tools prompt behavior. Some tools guide us in a healthy direction to build good stuff. Some tools enable our bad habits and some tools just are written to be relatively low opinion, not very opinionated. First, I want to talk about a tool that enables a lot of bad habits. It's a, you might have heard of it, it's called RSpec Rails. And I feel like whoever invented RSpec Rails was like, here's our marching orders. We're going to just do whatever Rails does and then wrap it with RCLI and DSL as uncritically as possible. So you got controllers? Yeah, we can spec them. Great. Without thinking whether that was a good idea. You got a testing pyramid? We got a testing pyramid. You want model specs and controller specs and helper specs and view specs and routing specs and request specs? Sure. And feature tests too. Why not have all these layers? And honestly, as somebody who's, especially when I was a novice coming in, I was like, well, clearly our tools are built for good reason. They have a good reason for having all these different tests. Test all the fucking time. That's great. Okay. So I thought, like, I looked at that and I was like, man, I got my work cut out for me to like live up to this seven layer nacho of testing. And what I came to realize over through a lot of usages is like, well, all those tests are very integrated. Every single one of them will call through to the database. And additionally, they're very redundant. When I have a new model that I'm writing here and I make a change there, I have this incidental coverage in all the tests above it. So all those tests now need to be updated as well. That creates a lot of like low value work, just cleaning up all my tests. So here's pro tip. Here's how I use RSpec Rails. This is a secret. My secret to using RSpec Rails is I have this whole thing and then I blow away all of them except for sometimes feature specs and sometimes model specs. And then if I have any sort of Ruby that's like at all interesting, I'll write it in Ruby code and then I'll test it with plain old RSpec. And that's the only way I've been able to find sanity with RSpec Rails. But it's not the tools fault per se, but I had to fight that tool to get to this point. I had to fight all the documentation and all the blog posts and all of the arguments with people about why I was having problems. And that was not an example of a great tool experience. Let me tell you about an experience with the tool that I thought was really, really helpful and great. Its name is RSpec. RSpec itself is actually really awesome, but I think that a lot of people have a hard time with RSpec Rails and then they turn around and they blame RSpec too. And I think that's kind of unfair. It's worth it to like look at them separately. So let's talk about what makes RSpec kind of cool. First of all, I don't believe that RSpec is a test framework per se. I think it's better to think of RSpec as a framework for helping you write better tests. RSpec influences our design. It was designed to do that. It was a response to XUnit with lots of repetitive methods that were all set up, like tons of tests set up in action and assert. But what was cool about nested example groups is we can see the same symmetry and have very terse tests that aren't redundant, but we don't lose any clarity through drying it up. That's one of my favorite things about RSpec style testing. Additionally, I love that the assertions guide the naming for our methods. If I write this test and the thing doesn't exist yet, by using this matcher BSilent, it's going to assume that there's an instance method called silent question mark on that class, which is a really handy way to inform that the usage is like sensible. Like, that's a natural name now. Additionally, years ago when I learned about Let, I was pairing with Corey Haynes. Corey is a really smart developer. He looked up to him and said Let is great because it lets you call out your setup stuff, create a new user and assign it to this method, and even better, it's lazily evaluated. I was like, I don't know, Corey, I worship you, so lazily evaluated sounds sweet. That's great. I'm going to use Let for everything, so I've used Let a lot. Another feature, LetBang, which will eagerly invoke that block, it has this interesting thing because people generally find LetBang by being like, well, I want this to run in exactly this order. I want to make sure that it invokes. Jim Weirich and I paired and he looked at my code base and he's like, dude, you're doing this totally wrong. Don't just use LetBang for absolutely everything. It's like there to draw out your attention to side effects in your code. It should be minimal. It should have them very, very sparingly. If you need to have a side effect in order for your code to work, that means that you have this coupling of state not just to the arguments but to other stuff happening in the system. That's why there's a bang. It means don't do it. That was an interesting conversation that I never would have had if it wasn't for RSpec. Additionally, RSpec reduces friction. The CLI is great because it's really convenient, easy to use, pretty obvious, helps you focus on just what you want to run, has a good output. It's all work that I'd have to do if I was building my own rake tasks and my own testing CLI stuff on every project. I love RSpec's built-in reporters. Oh, my God, we're at 30 minutes because of all the AV stuff. Please don't leave. All the reporters you need. You have all the CI stuff that you need. There's so many RSpec plug-ins. I love that I get to focus on just my tests and not the stuff around my tests. Additionally, RSpec fosters empathy. The API is designed to let you have a place to write in what the heck you're doing, describe the slide and how it complements RSpec. You have this opportunity in there to tell a little bit of your story in a way that's congruent with your tests. Another thing I love is that it shifts your perspective. RSpec has a domain-specific language. It does not look like normal Ruby. That is a level of indirection. However, it forces me to think of my methods not just as methods, but outside in what's it like to use them. What's it like from the perspective of a stakeholder? What's it like under a different context? I really like the DSL for forcing me out of just thinking of just methods and classes. Another tool talking about tools prompting behavior. It's possible to write tools that just don't have a whole lot of opinions. Minitest is a good example of one such tool. It has a different priority than RSpec. An analogy I picked up from Aaron this week is you could think of Minitest as a race car. That's why DHH uses Minitest, by the way, if you don't know. It's lean, mean, it's essential. It's only what you need to get your tests written. It's all pure Ruby, except it has these hard bucket seats. Versus RSpec, a luxury sedan with a lot of knobs and dials, but it's mostly full-featured and quite comfortable to ride in. If you want a comfortable seat, RSpec offers you this rich Corinthian leather experience that you can just sit in and feel comfortable. The Zeitgeist right now, and by the way, if you don't know the word Zeitgeist, it's a German word for time snapchat. The Zeitgeist right now is saying that Minitest is really hot. When I talked to all my friends, a lot of them have dropped to RSpec, started using Minitest. I think it's just really popular right now. I think that one of the reasons is people generally spread fear and uncertainty and doubt about RSpec that is too verbose, is bloated, is slow, is too much in direction, is better to just write pure Ruby. You ain't going to need it. I am here too. I use Minitest on a lot of my projects. I like Minitest just fine. I like that it doesn't have very many opinions and it gets out of my way and I can just write just the tests I want. But of course, I carry with me the fact that I actually have very finely, after years and years, I have my own testing opinions that I know work very well for me and I can write tests without getting myself into too much trouble usually. But if you're not a testing expert and you don't want to be a testing expert or if you're on a team with novices, what I would suggest is like, remember, I learned a lot discussing RSpec and grappling with its API and its features with past teammates. I think that you might benefit from that too if you haven't had that experience yet. So yeah, on one hand, RSpec takes longer to learn, but when you learn how to use RSpec, you're also learning stuff about design and testing. And so maybe that's not so much a bug as a feature in some cases. So if you're still not happy with your test sweeps, I suspect that you might be looking for a tool to solve your problems when instead we can use our brains and use thinking instead and change our approach. Oddly enough, at RubyConf last year, I gave a talk on exactly that. You can find it called is.gd slash stop hate. It's called how to stop hating your tests and it's not about tools, it's just about thinking. All right, so in the time remaining, I'm going to get a little bit more meta. Why are we here really? The fact that anyone came to this talk worries me. I would not have come to this talk. Let me explain. Let me back up. First of all, giving somebody else's talk is a lot like testing their code. Because I've had to open up all of Sam's work and his notes and stuff and try to understand what he was going to say here today. So if you see something confusing when you're looking at somebody else's code and you're trying to write tests for it or trying to review it, it's easy to think there are obviously a moron. So it's important to assume that the author is smart and intelligent and had reasons. Meanwhile, if you see something that's obviously awesome, great, it's still your job to put on a critical hat and investigate it anyway and ask the hard questions about why we're here. So let's critique this talk. Not the stuff that I said, just the Sam stuff. The stuff that I said, it's fine. This is the abstract. I assume you've read it. I won't reread it or anything. This is abstract. This is the first thing I read when he texted me to see if I could give this talk. This is my opinion of the abstract. People like peanut butter, people like chocolate, slam them together, RSpec Rails. This talk, I felt like I read the abstract. I'm like, this could be a six paragraph blog post. And so the next thing I did was I googled RSpec Rails 5 and found Sam's six paragraph blog post. And I was just thinking, I was mad, I was like, why was this talk selected here? How did this talk fly through the CFP process without any criticality whatsoever? Like, that just doesn't seem right. Now granted, my talk was rejected and I'm a little bit biased. I might be a little salty. But when I thought about it, I think that the reason was that this was a safe talk. This is a comfortable talk. This is well within everyone here's comfort zones. I use RSpec, I use Rails, let's find out what's new. Great. But I feel like that comfort should scare us because when we're in a group like this that's maturing and we're getting up to major version numbers like five, you know, comfort can breed complacency. So RSpec, if we're just content with where things are and we're pretty happy with RSpec and we're just happy to see, you know, like little tiny tweaks here and there and make sure it continues to support stuff in the future, you're not writing blog posts about this new RSpec thing, you're not writing new tools, you're talking about RSpec less. Even if RSpec does everything you want it to do. Minitests, meanwhile, lately, like the zeitgeist, I've seen a lot of people talking up Minitests, writing more plugins, educating people a little bit more with blog posts and as a result, it's getting a little bit more attention. So as a new person walks into the room, they're going to see people talking about Minitests more than RSpec, they're going to tend to go towards Minitests, not RSpec. So this reminds me a little bit of a similar dichotomy. Rails. Rails is pretty mature now. It's over 10 years old, it solves the problems that it solves really well and it's pretty well known what it's good at and what it's not. So people talk about Rails a little bit less. Especially all of us busy getting stuff done and building things. We're not out there advocating Rails anymore because we get to use Rails at work, which is itself fantastic. However, when you look at jobs, Rails jobs are on the decline. They're not just slowing down, it's negative growth. This is another thing, the technology that shall not be named. Everyone's talking about Node.js. Like it or not, 900% year over year growth in jobs on Indeed. There's a lot of activity there and it's not about, this is not a contest of who's the better technology or who solves stuff better, it's what's the front page of Hacker News. So my challenge is thinking about this talk and why the hell we're giving this talk and why we're here. That was ironic. Because that's one of our options. The other option, if we're not willing to be uncomfortable, is we're going to see Ruby jobs start to dry up. There might be fewer people at Rails count 2018 than this year. Another way to think about this is if you're not familiar with Ruby Together is a non-profit that pays people to work on Ruby open source. Another way to think about this is ask, what were the conditions necessary in order for Ruby Together to seem like a necessary and good idea? Well, when an ecosystem is popular, everything's easy because there's just wave after wave of person on the Internet who's going to write open source for free just for the ego, just for the fame to be attributed to the new popular thing. Also easy, sponsored stuff like Oracle backs Java. Java's not going to go anywhere because Oracle's incentivized for Java to be successful. Google is not going to drop go unless they feel like it. I already dropped the mic, but it's done. JavaScript cannot die because multiple vendors have staked their businesses on it. Every single browser, JavaScript is not going to go anywhere, so it's a really safe bet. We're talking about RSpec, it's mature at this point. I don't mean mature as like a four letter word. Mature means mostly done. Bundler is mature. Rails is mature. Ruby is mature. They mostly do what they need to do to do their job well. That means as a result that when you maintain a popular gem like RSpec, it no longer makes you rich and famous necessarily. The ecosystem, the stuff that they had to do just to make RSpec continue working with Rails 5 is almost all stuff that you don't actually see. It's all internals, legacy code refactoring. No one really wants to do that. The reason Ruby together needs to exist is because the energy and the funding to keep Ruby competitive isn't there otherwise. That is disconcerting because Ruby together isn't going to ever be big enough to solve that fundamental systemic problem. Let's talk about my real job, sales. I spent a lot of time talking to business people about software solutions and building software apps and stuff. Entrepreneurs that I talk to are always talking about certain technologies that they hear about that are advice to them. The mean stack like Mongo Express, Angular, and by the way, when people, I've talked to multiple business people this year who are like, yeah, we're going to build a new application, we're going to do it all in Angular 1.x. People are teaching business people, oh, you don't want Angular 2, just stay on one forever. I don't get it. We're just going to wait. Wait it out. And Node.js, the so-called mean stack. A lot of entrepreneurs are pushing this kind of stuff. Another one, a lot of people are just assuming based on trendiness. Node and React are just the way to go. You know who's talking about Ruby and Rails nowadays out in the marketplace? Like has the ear of CTOs and directors of engineering? People spreading fear and certainty and doubt because they have their preferred upstart technology that's faster or whatever. And what those businesses are hearing is that there aren't enough Rubyists out there, that the Rubyists that do exist cost too much, that Ruby is slow, and that our spec doesn't scale either at runtime or operationally. Now if you're in the room, you're like, no, no, no, Ruby's fine, this is okay. But I think that this is like a real important bit of anecdote from the life of Justin Searles we all need to deal with to help solve my consulting sales problem. So, because I don't like sales. But that's why it's so frustrating is Rails is still the best choice for entire classes of applications. But because we stopped saying it a few years ago, businesses stopped hearing it. People only share new stuff that excites them. That's novel. If you were to discover immortality today, it would drop off the front page of Hacker News after a week or two. People wouldn't be talking about it. They'd find some new shiny thing. They'd be talking about React Native 1.0. And not that you just, you know, defeated death. Even though that thing is way more objectively better, it's not novel after a certain bit of time. So the dilemma, right? Ruby is no longer new. Ruby is still good. We got to do something so Ruby can remain relevant and we can keep working on Ruby at work. What's the, we do something part? Remember Ruby's mature. It does its job mostly well and one thing that I think our community, the technologists need to get comfortable with is that it is okay for tools to be mostly finished. It is okay for software to just mostly do its job and be good at what it does. In any other industry, it would be ridiculous for us to say otherwise. Like, oh, that's clearly obsolete now because it's not, you know, super active and they're not adding new features. At a certain point, it just does what it needs to do. Remember I said the key is to happiness. We're our tools. We're not like Ruby. We like Rails. That's why you're here. And how we use them. So maybe it's time for us as a community to de-emphasize the tools and start talking more about how we use those tools to accomplish really cool stuff. Because there's all these evergreen problems in software. There's all these problems we're never going to solve. We're never going to solve testing. We're going to just get asymptotically better each time. We're never going to solve design because we're always going to find new ways to design code. Like human issues are never going to be solved either, right? How our code communicates its intent to its reader is never going to be solved. I swear, I get like five bonus minutes. Sarah, can I have a minute? She's nodding very tepidly. So we got to tell stories that help people solve problems in ways that are more than just look at this new shiny bobble. And if you love Ruby, tell your story in Ruby and associate it back with Ruby so that Ruby remains known as a community of people who really get object oriented design right, who get testing right, who get community and inclusiveness right. Being known for those things and having people talk about those things are enough to keep us relevant. And when you think about whose job this is, remember that most of the people who made Ruby so famous in the first place don't write Ruby anymore. Their chapter is complete. Most of them have moved on to other ecosystems. Some of them are no longer even with us. And that means that keeping Ruby relevant is not somebody else's job. I hate to break it to you, but the fact that you show up to a conference called RailsConf in a room that holds just a couple hundred people means that you're one of the top couple hundred people whose job this is now to keep Ruby relevant, if you care. So my message is make Ruby great again. And tell your story. We don't have the time to talk about it today. Use this hashtag and tell me something that you could do to tell a story that might change something, that might have an impact on others, and I convinced them that Ruby is a better solution than the technology that shall not be named for whatever it is that you're doing. Again, my name is Soros. I'd love to be your friend. I'm going to be here for the rest of the week. If you want to help us in our mission to fix how the world writes software, consider joining test double. We're always hiring great developers. If your company is looking for senior developers and you're struggling to find people to add to your team, our developer consultants are great senior developers who would love to work on your team with you and build stuff alongside you. You don't want either of those things, but you want a sticker. I've got stickers too. And most importantly, thank you all so much for your time. I really, really appreciate it."
#     , model=model, tokenizer=tokenizer
# )

# print(summary)