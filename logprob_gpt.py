import torch
import numpy as np
import random
import scipy.stats as st
from transformers import AutoTokenizer, GPT2LMHeadModel,MT5Tokenizer
from tqdm import tqdm
import json
import os
from datetime import datetime
import os
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"

# ==== Config ====
SAMPLE_NUMBERS = 50000
SEED_LIST = [667766]

# ==== Language for mgpt1.3b & 13b ====
languages = [
    "af", "ar", "az", "be", "bg", "bn", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fil", 
    "fr", "hi", "hu", "hy", "id", "it", "iw", "ja", "jv", "ka", "kk", "ko", "ky", "lt", "lv", "ml", 
    "mr", "ms", "my", "pl", "pt", "ro", "ru", "sv", "sw", "ta", "te", "tg", "th", "tr", "uk", "ur", 
    "uz", "vi", "yo"
]

# ==== Language for mgpt101 ====
languages = ['af', 'am', 'be', 'bg', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'et', 'eu', 'fi', 'fil', 'fr',
 'ga', 'gd', 'gl', 'gu', 'ha', 'hi', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn',
 'ko', 'ky', 'lb', 'lo', 'lt', 'lv', 'mi', 'mk', 'ml', 'mr', 'mt', 'my', 'ne', 'nl', 'ny', 'pa', 'pl', 'pt',
 'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th',
 'tr', 'uk', 'ur', 'vi', 'xh', 'yo', 'zu', 'ar', 'zh', 'fa', 'no', 'es', 'ht', 'ms', 'sq', 'ku', 'yi', 'uz',
 'ps', 'mg', 'az', 'bn', 'iw']




BATCH_SIZE =100
MAX_TOKEN = 150
SUFFIX_RATIO = 0.1
PREFIX_NUM = int((1-SUFFIX_RATIO) * MAX_TOKEN)

#MODEL_NAME = "THUMT/mGPT"

MODEL_NAME = "ai-forever/mGPT-13B"

model_id = MODEL_NAME.split("/")[-1]
MGPT_NEW = True

# ==== Paths ====
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "dataset", "longone")
result_dir = os.path.join(base_dir, "results")
memorization_path = os.path.join(result_dir, "memorization_sample", model_id)
cache_path = os.path.join(base_dir, "model")

for path in [data_dir, result_dir, memorization_path, cache_path]:
    os.makedirs(path, exist_ok=True)

# ==== Model Load ====
model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME,
    cache_dir=cache_path,
    torch_dtype=torch.float16
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_path, use_fast=False)
#tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_path)

model.eval()

# ==== Output Path ====
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
result_output_path = os.path.join(result_dir, f"logprob_misstartfrommr_{model_id}_{MAX_TOKEN}_{int(SUFFIX_RATIO*100)}_{timestamp}.json")
final_results = {}

# ==== Utility Functions ====

import math

def visualize_token_logprobs(token_logprobs):
    max_token_len = max(len(token) for token, _ in token_logprobs)

    for token, logprob in token_logprobs:
        token_display = token.replace("\n", "\\n").replace("\t", "\\t")
        print(f"{token_display:<{max_token_len}}  |  log-prob: {logprob:.4f}")

    total_log_prob = sum(lp for _, lp in token_logprobs)
    prob = math.exp(total_log_prob)

    print(f"\n all log-prob: {total_log_prob:.4f}")
    print(f"Total log-prob: {prob:.8f}")


def compute_token_logprobs(model, tokenizer, full_input_ids, device="cuda"):

    with torch.no_grad():
        outputs = model(input_ids=full_input_ids.unsqueeze(0))
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    
    suffix_token_ids = full_input_ids[PREFIX_NUM:]

    token_logprobs = []
    for i, token_id in enumerate(suffix_token_ids):
        log_prob = log_probs[0, PREFIX_NUM + i - 1, token_id].item()
        token_str = tokenizer.decode([token_id])
        token_logprobs.append((token_str, log_prob))

    total_logprob = sum(lp for _, lp in token_logprobs)
    return token_logprobs, total_logprob

def compute_token_logprobs_batch(model, tokenizer, full_input_ids_batch, device="cuda"):
    """
    full_input_ids_batch: Tensor of shape [B, L]
    Returns:
        log_probs_per_token: Tensor [B, L_suffix]
        total_logprobs: Tensor [B]
    """
    
    B, L = full_input_ids_batch.shape
    suffix_token_ids = full_input_ids_batch[:, PREFIX_NUM:]  # [B, L_suffix]
    L_suffix = suffix_token_ids.size(1)

    with torch.no_grad():
        outputs = model(input_ids=full_input_ids_batch)
        logits = outputs.logits  # [B, L, V]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    aligned_logits = log_probs[:, PREFIX_NUM - 1:L - 1, :]  # [B, L_suffix, V]

    batch_indices = torch.arange(B).unsqueeze(1).to(device)
    time_indices = torch.arange(L_suffix).unsqueeze(0).to(device)

    log_probs_per_token = aligned_logits[batch_indices, time_indices, suffix_token_ids]  # [B, L_suffix]
    total_logprobs = log_probs_per_token.sum(dim=1)  # [B]

    return log_probs_per_token, total_logprobs


    
def compute_mean_and_confidence_interval(data, confidence=0.95):
    clean_data = [x for x in data if x is not None and not np.isnan(x)]
    if len(clean_data) == 0:
        return 0.0, (0.0, 0.0)
    clean_data = np.array(clean_data)
    mean = np.mean(clean_data)
    if len(clean_data) == 1:
        return mean, (mean, mean)
    sem = st.sem(clean_data)
    ci_range = sem * st.t.ppf((1 + confidence) / 2., len(clean_data) - 1)
    return mean, (mean - ci_range, mean + ci_range)


def summarize_all_metrics(results):
    logprobs = [r["total_logprob"] for r in results]
    mean, ci = compute_mean_and_confidence_interval(logprobs)
    print("==== Log Probability Summary ====")
    print(f"Avg LogProb : {mean:.4f} (95% CI: {ci[0]:.4f} ~ {ci[1]:.4f})")
    print("=================================\n")
    return {
        "avg_total_logprob": mean,
        "logprob_ci": ci,
        "total_samples": len(results)
    }


def random_sample_prefix_suffix_batch(
    text_list, tokenizer, num_tokens=100, suffix_ratio=0.15, seed=None
):
    batch = tokenizer(text_list, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=2048)
    input_ids_batch = batch['input_ids'].to("cuda")

    prefix_inputs = []
    target_outputs = []
    full_tokens_inputs = []
    batch_size, max_seq_len = input_ids_batch.shape

    for i in range(batch_size):
        input_ids = input_ids_batch[i]
        input_ids = input_ids[input_ids != tokenizer.pad_token_id]
        seq_len = len(input_ids)

        if seq_len < num_tokens:
            continue

        if seed is not None:
            random.seed(seed + i)

        start_idx = random.randint(0, seq_len - num_tokens)
        selected_ids = input_ids[start_idx: start_idx + num_tokens]

        #suffix_len = int(num_tokens * suffix_ratio)
        #prefix_len = num_tokens - suffix_len

        #prefix_ids = selected_ids[:prefix_len]
        #suffix_ids = selected_ids[prefix_len:]

        #prefix_inputs.append(prefix_ids)
        #target_outputs.append(suffix_ids)
        full_tokens_inputs.append(selected_ids)


    #prefix_inputs_padded = torch.nn.utils.rnn.pad_sequence(prefix_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    #target_outputs_padded = torch.nn.utils.rnn.pad_sequence(target_outputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    full_tokens_inputs_padded = torch.nn.utils.rnn.pad_sequence(full_tokens_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    return full_tokens_inputs_padded 
    #return prefix_inputs_padded, target_outputs_padded


def save_logprob_samples_as_txt(lang, results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"logprob_samples_{lang}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Total LogProb: {r['total_logprob']:.4f}\n")
            f.write("Token LogProbs:\n")
            for token, logp in r["token_logprobs"]:
                f.write(f"  {token}: {logp:.4f}\n")
            f.write("="*60 + "\n")

# ==== MAIN LOOP ====

for lang in languages:
    print(f"\nProcessing Language: {lang}")
    file_path = os.path.join(data_dir, f"{lang}.jsonl")

    if not os.path.exists(file_path):
        print(f"File not found for language: {lang}, skipping.")
        continue
        
    with open(file_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f][:SAMPLE_NUMBERS]

    results_per_lang = []

    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"🔍 {lang} logprob"):
        batch_samples = samples[i:i + BATCH_SIZE]
        text_list = [item["text"] for item in batch_samples]

        for repeat_idx, seed in enumerate(SEED_LIST):

            full_input_batch = random_sample_prefix_suffix_batch(
                    text_list,
                    tokenizer,
                    num_tokens=MAX_TOKEN,
                    suffix_ratio=SUFFIX_RATIO,
                    seed=seed
                )

            batch_size = full_input_batch.shape[0]

            token_logprobs_bat, total_logprob_bat = compute_token_logprobs_batch(
                        model, tokenizer, full_input_batch, device="cuda"
                    )
            for token_log_probs, total_log_prob in zip(token_logprobs_bat, total_logprob_bat):
                results_per_lang.append({
                    "token_logprobs": token_log_probs.tolist(),
                    "total_logprob": total_log_prob.item()
                 })



              

    lang_summary = summarize_all_metrics(results_per_lang)
    final_results[lang] = lang_summary


    with open(result_output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"\nFinal summary for ALL languages saved to {result_output_path}")
