import os
import json
import numpy as np
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from tqdm import tqdm
from datetime import datetime
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import random
import torch
#SEED_LIST = [10777140, 10, 19970624, 666666, 20250421]
SEED_LIST = [10777140]
SAMPLE_NUMBERS = 50000
#EXTRA_ID_0 = 256299
EXTRA_ID_0 = 250099

BATCH_SIZE = 200
SUFIX_RATIO = 0.1
MAX_TOKEN = 150
PREFIX = str(int(SUFIX_RATIO*MAX_TOKEN))
MODEL_NAME = "google/mt5-large"
print(PREFIX)

languages = ['af', 'am', 'be', 'bg', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'et', 'eu', 'fi', 'fil', 'fr',
 'ga', 'gd', 'gl', 'gu', 'ha', 'hi', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn',
 'ko', 'ky', 'lb', 'lo', 'lt', 'lv', 'mi', 'mk', 'ml', 'mr', 'mt', 'my', 'ne', 'nl', 'ny', 'pa', 'pl', 'pt',
 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th',
 'tr', 'uk', 'ur', 'vi', 'xh', 'yo', 'zu', 'ar', 'zh', 'fa', 'no', 'es', 'ht', 'ms', 'sq', 'ku', 'yi', 'uz',
 'ps', 'mg', 'az', 'bn', 'iw']

model_id = MODEL_NAME.split("/")[-1]
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "dataset", "longone")
result_dir = os.path.join(base_dir, "results")
memorization_path = os.path.join(result_dir, "memorization_sample", model_id)
cache_path = os.path.join(base_dir, "model")


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_path)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=cache_path).to("cuda")


model.eval()


def extract_spans_from_tokens_precisely(ids, max_extra_id=4, eos_token_id=1):
    spans = []
    current_span = []
    capturing = False
    current_expected_extra_id = EXTRA_ID_0  # <extra_id_0>

    for token in ids:
        if token == current_expected_extra_id:
            if capturing:
                spans.append(current_span)
                current_span = []
            capturing = True
            current_expected_extra_id -= 1

            if current_expected_extra_id < EXTRA_ID_0 - max_extra_id:
                capturing = False
        elif token in range(EXTRA_ID_0 - max_extra_id - 10, EXTRA_ID_0 + 1) or token == eos_token_id:
            if capturing:
                spans.append(current_span)
            capturing = False
        elif capturing:
            current_span.append(token)

    if capturing and current_span:
        spans.append(current_span)

    return spans



def random_sample_and_corrupt_batch(text_list, tokenizer, num_tokens=100, corruption_rate=SUFIX_RATIO, mean_span_length=3, seed=None):
    # 一次性batch encode
    batch = tokenizer(text_list, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    input_ids_batch = batch['input_ids'].to("cuda")  # shape: (batch_size, seq_len)

    corrupted_inputs = []
    target_outputs = []
    selected_texts = []

    batch_size, max_seq_len = input_ids_batch.shape

    for i in range(batch_size):
        input_ids = input_ids_batch[i]
        seq_len = (input_ids != tokenizer.pad_token_id).sum().item()
        input_ids = input_ids[:seq_len]  # remove paddings

        if seq_len < num_tokens:
            continue  #

        if seed is not None:
            random.seed(seed + i)

        start_idx = random.randint(0, seq_len - num_tokens)
        selected_ids = input_ids[start_idx: start_idx + num_tokens]

        selected_text = tokenizer.decode(selected_ids, skip_special_tokens=False)
        seq_len = selected_ids.shape[0]
        num_tokens_to_mask = int(seq_len * corruption_rate)
        num_spans = int(num_tokens_to_mask / mean_span_length)

        if num_spans == 0:
            continue  #

        lengths = [1] * num_spans
        remaining = num_tokens_to_mask - num_spans

        while remaining > 0:
            idx = random.randint(0, num_spans - 1)
            lengths[idx] += 1
            remaining -= 1

        assert sum(lengths) == num_tokens_to_mask, "wrong all mask"

        available_positions = list(range(seq_len))
        spans = []

        for span_len in lengths:
            possible_starts = []
            for pos in available_positions:
                if all((pos + offset) in available_positions for offset in range(span_len)):
                    possible_starts.append(pos)

            if not possible_starts:
                raise ValueError(f"error, no enough space")

            start_idx = random.choice(possible_starts)
            end_idx = start_idx + span_len

            spans.append((start_idx, end_idx))

            for idx in range(start_idx - 1, end_idx + 1):
                if idx in available_positions:
                    available_positions.remove(idx)

        spans = sorted(spans, key=lambda x: x[0])

        corrupted_tokens = []
        target_tokens = []
        current_idx = 0
        extra_id = 0

        for start, end in spans:
            if start > current_idx:
                corrupted_tokens.extend(selected_ids[current_idx:start].tolist())
            corrupted_tokens.append(EXTRA_ID_0 - extra_id)
            target_tokens.append(EXTRA_ID_0 - extra_id)
            target_tokens.extend(selected_ids[start:end].tolist())
            extra_id += 1
            current_idx = end

        if current_idx < seq_len:
            corrupted_tokens.extend(selected_ids[current_idx:].tolist())

        corrupted_input_ids = torch.tensor(corrupted_tokens, device=selected_ids.device)
        target_ids = torch.tensor(target_tokens + [tokenizer.eos_token_id], device=selected_ids.device)

        corrupted_inputs.append(corrupted_input_ids)
        target_outputs.append(target_ids)
        selected_texts.append(selected_text)

    if len(corrupted_inputs) == 0:
        raise ValueError("skip,no enough sample！")

    corrupted_inputs_batch = torch.stack(corrupted_inputs, dim=0)
    target_outputs_batch = torch.stack(target_outputs, dim=0)

    return corrupted_inputs_batch, target_outputs_batch, selected_texts


def compute_mean_and_confidence_interval(acc_list, confidence_level=0.95):

    arr = np.array(acc_list)
    mean = arr.mean()
    std = arr.std(ddof=1)
    n = len(arr)
    stderr = std / np.sqrt(n)

    from scipy import stats
    ci = stats.t.interval(confidence_level, n-1, loc=mean, scale=stderr)

    return mean, ci

def summarize_likelihood_metrics(log_liks, avg_nlls, perplexities):

    total_samples = len(log_liks)

    loglik_mean, loglik_ci = compute_mean_and_confidence_interval(log_liks)
    nll_mean, nll_ci = compute_mean_and_confidence_interval(avg_nlls)
    ppl_mean, ppl_ci = compute_mean_and_confidence_interval(perplexities)

    print(f"==== Log-Likelihood Evaluation on {total_samples} Samples ====")
    print(f"Log-Likelihood:       {loglik_mean:.4f} (95% CI: {loglik_ci[0]:.4f} ~ {loglik_ci[1]:.4f})")
    print(f"Average NLL (Loss):   {nll_mean:.4f} (95% CI: {nll_ci[0]:.4f} ~ {nll_ci[1]:.4f})")
    print(f"Perplexity:           {ppl_mean:.4f} (95% CI: {ppl_ci[0]:.4f} ~ {ppl_ci[1]:.4f})")
    print("=========================================================\n")

    return {
        "total_samples": total_samples,
        "loglik_mean": loglik_mean,
        "loglik_ci": loglik_ci,
        "nll_mean": nll_mean,
        "nll_ci": nll_ci,
        "ppl_mean": ppl_mean,
        "ppl_ci": ppl_ci
    }

import torch
import torch.nn.functional as F

def compute_encoder_decoder_ppl_verbose_batch(model, tokenizer, input_ids, labels, verbose=3):

    batch_size, seq_len = labels.shape
    decoder_input_ids = model._shift_right(labels)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]

    eos_token_id = tokenizer.eos_token_id
    extra_id_min, extra_id_max = 250001, 250099

    valid_mask = labels != -100
    special_mask = (labels >= extra_id_min) & (labels <= extra_id_max)
    if eos_token_id is not None:
        special_mask |= (labels == eos_token_id)

    decode_mask = torch.ones_like(valid_mask, dtype=torch.bool)
    try:
        flat_labels = labels.view(-1)
        decoded = tokenizer.batch_decode(flat_labels, skip_special_tokens=False)
        decode_mask = torch.tensor(
            [s.strip() != "" for s in decoded], dtype=torch.bool, device=labels.device
        ).view_as(labels)
    except:
        pass

    final_mask = valid_mask & (~special_mask) & decode_mask
    filtered_log_probs = token_log_probs.masked_fill(~final_mask, float("nan"))

    log_prob_sums = torch.nan_to_num(filtered_log_probs, nan=0.0).sum(dim=1)
    token_counts = final_mask.sum(dim=1)
    avg_neg_log_likelihood = -log_prob_sums / token_counts.clamp(min=1)
    ppl = torch.exp(avg_neg_log_likelihood)

    
    ppl_list = []
    loglik_list = []
    count_list = []

    for i in range(batch_size):
        if token_counts[i] == 0:
            if isinstance(verbose, int) and i < verbose:
                print(f"\n🟡 Sample {i} skipped: no valid tokens.")
            continue

        ppl_list.append(ppl[i].item())
        loglik_list.append(log_prob_sums[i].item())
        count_list.append(token_counts[i].item())

        if isinstance(verbose, int) and i < verbose:
            print(f"\n🟢 Sample {i}:")
            print(f"   Token count     = {token_counts[i].item()}")
            print(f"   Log prob sum    = {log_prob_sums[i].item():.4f}")
            print(f"   Perplexity (PPL)= {ppl[i].item():.4f}")

    return {
        "perplexities": ppl_list,
        "logliks": loglik_list,
        "token_counts": count_list
    }

os.makedirs(result_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

final_results = {}
result_output_path = os.path.join(result_dir, f"LOG_PREPLIXITY_{model_id}_{MAX_TOKEN}_{PREFIX}_{timestamp}.json")


for lang in languages:
    print(f"\n🌍 Processing Language: {lang}")
    file_path = os.path.join(data_dir, f"{lang}.jsonl")

    with open(file_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f]
        samples = samples[:SAMPLE_NUMBERS]
    
    all_log_liks = []
    all_avg_nlls = []
    all_perplexities = []


    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"🔍 {lang} (greedy decode)"):
        batch_samples = samples[i:i + BATCH_SIZE]
        text_list = [item["text"] for item in batch_samples]

        for repeat_idx, seed in enumerate(SEED_LIST):
            try:
                input_ids, labels, selected_texts = random_sample_and_corrupt_batch(
                    text_list,
                    tokenizer,
                    num_tokens=MAX_TOKEN,
                    corruption_rate=SUFIX_RATIO,
                    mean_span_length=3,
                    seed=seed  #
                )
                
                with torch.no_grad():                    
                    metrics = compute_encoder_decoder_ppl_verbose_batch(
                        model=model,
                        input_ids=input_ids,
                        labels=labels,
                        tokenizer=tokenizer,
                        verbose=0  #
                    )
                    all_log_liks.extend(metrics["logliks"])
                    all_perplexities.extend(metrics["perplexities"])
                    all_avg_nlls.extend([
                        -loglik / count for loglik, count in zip(metrics["logliks"], metrics["token_counts"])
                        ])

            except Exception as e:
                print(f"⚠️ Skip batch repeat {repeat_idx} (seed={seed}) due to error: {e}")
                continue
            


    lang_summary = summarize_likelihood_metrics(all_log_liks, all_avg_nlls, all_perplexities)



    final_results[lang] = lang_summary

    with open(result_output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)


with open(result_output_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"✅ Final summary for ALL languages saved to {result_output_path}")
