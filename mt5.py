import os
import json
import numpy as np
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from tqdm import tqdm
from datetime import datetime
import random
import torch

# Configure CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Experiment seeds and parameters
SEED_LIST = [10777140]
SAMPLE_NUMBERS = 50000
EXTRA_ID_0 = 250099

BATCH_SIZE = 1000
SUFIX_RATIO = 0.3
MAX_TOKEN = 50
PREFIX = str(int(SUFIX_RATIO * MAX_TOKEN))
MODEL_NAME = "google/mt5-base"
print(PREFIX)

# List of languages to process
languages = [
    'af', 'am', 'be', 'bg', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'et', 'eu', 'fi', 'fil', 'fr',
    'ga', 'gd', 'gl', 'gu', 'ha', 'hi', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn',
    'ko', 'ky', 'lb', 'lo', 'lt', 'lv', 'mi', 'mk', 'ml', 'mr', 'mt', 'my', 'ne', 'nl', 'ny', 'pa', 'pl', 'pt',
    'ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th',
    'tr', 'uk', 'ur', 'vi', 'xh', 'yo', 'zu', 'ar', 'zh', 'fa', 'no', 'es', 'ht', 'ms', 'sq', 'ku', 'yi', 'uz',
    'ps', 'mg', 'az', 'bn', 'iw'
]

model_id = MODEL_NAME.split("/")[-1]
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "dataset", "longone")
result_dir = os.path.join(base_dir, "results")
memorization_path = os.path.join(result_dir, "memorization_sample", model_id)
cache_path = os.path.join(base_dir, "model")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_path)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=cache_path).to("cuda")
model.eval()


def extract_spans_from_tokens_precisely(ids, max_extra_id=4, eos_token_id=1):
    """
    Extract masked spans from token IDs as per T5-style corruption.

    Args:
        ids (List[int]): Token ID sequence.
        max_extra_id (int): Maximum number of extra IDs expected.
        eos_token_id (int): EOS token ID.

    Returns:
        spans (List[List[int]]): List of token spans.
    """
    spans = []
    current_span = []
    capturing = False
    current_expected_extra_id = EXTRA_ID_0  # Start with <extra_id_0>

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


def save_exact_match_samples_as_txt(
    lang,
    input_texts,
    predicted_batch,
    target_batch,
    exact_match_indices,
    tokenizer,
    save_dir="emnlp25/results/memorization_sample",
    verbe=False
):
    """
    Save exact match samples to a text file for inspection.

    Args:
        lang (str): Language identifier.
        input_texts (torch.Tensor): Batch of corrupted input IDs.
        predicted_batch (List[List[int]]): Predicted token IDs.
        target_batch (List[List[int]]): Reference token IDs.
        exact_match_indices (List[int]): Indices of exact matches.
        tokenizer: Tokenizer to decode tokens.
        save_dir (str): Output directory.
        verbe (bool): Verbose flag (unused).
    """
    os.makedirs(save_dir, exist_ok=True)
    assert len(predicted_batch) == len(target_batch) == len(exact_match_indices), \
        "Mismatch in exact match subset lengths!"
    save_path = os.path.join(save_dir, f"exact_match_samples_{lang}.txt")

    with open(save_path, "a", encoding="utf-8") as f_out:
        for i, idx in enumerate(exact_match_indices):
            sample_input = tokenizer.decode(input_texts[idx], skip_special_tokens=True)
            reference = tokenizer.decode(target_batch[i], skip_special_tokens=True)
            prediction = tokenizer.decode(predicted_batch[i], skip_special_tokens=True)

            f_out.write(f"Input: {sample_input}\n")
            f_out.write(f"Reference: {reference}\n")
            f_out.write(f"Prediction: {prediction}\n")
            f_out.write("=" * 50 + "\n")


def evaluate_predictions_token_level_precisely(
    lang,
    pred_ids_batch,
    target_ids_batch,
    corrupted_inputs_batch,
    max_extra_id=4,
    tokenizer=None,
    eos_token_id=1,
    verbose=True
):
    """
    Evaluate token, span, and sample-level accuracy for predicted sequences.

    Args:
        lang (str): Language identifier.
        pred_ids_batch (torch.Tensor): Generated output IDs.
        target_ids_batch (torch.Tensor): Reference output IDs.
        corrupted_inputs_batch (torch.Tensor): Corrupted input IDs.
        max_extra_id (int): Max number of extra IDs.
        tokenizer: Tokenizer to decode tokens.
        eos_token_id (int): EOS token ID.
        verbose (bool): Whether to print per-sample details.

    Returns:
        dict: Accuracies per level.
    """
    batch_token_correct = []
    batch_span_correct = []
    batch_sample_correct = []
    exact_match_inputs_index = []
    exact_match_preds = []
    exact_match_targets = []

    for i in range(pred_ids_batch.shape[0]):
        pred_ids = pred_ids_batch[i].tolist()
        target_ids = target_ids_batch[i].tolist()

        pred_spans = extract_spans_from_tokens_precisely(pred_ids, max_extra_id, eos_token_id)
        target_spans = extract_spans_from_tokens_precisely(target_ids, max_extra_id, eos_token_id)

        token_correct = 0
        token_total = 0
        span_correct = 0
        span_total = 0
        all_spans_correct = True

        if verbose:
            print(f"==== Sample {i} ====")

        for idx in range(min(len(pred_spans), len(target_spans))):
            pred_span = pred_spans[idx]
            target_span = target_spans[idx]

            pred_set = set(pred_span)
            target_set = set(target_span)

            token_hits = len(pred_set & target_set)
            token_correct += token_hits
            token_total += len(target_set)

            is_span_correct = (pred_span == target_span)
            if not is_span_correct:
                all_spans_correct = False

            if verbose:
                print(f"[Mask {idx}]")
                if tokenizer:
                    print("Predicted:", tokenizer.decode(pred_span))
                    print("Target   :", tokenizer.decode(target_span))
                print(f"Tokens hit: {token_hits}/{len(target_set)}")
                print(f"Span correct: {is_span_correct}\n")

            if is_span_correct:
                span_correct += 1

            span_total += 1

        token_acc = token_correct / token_total if token_total > 0 else 0
        span_acc = span_correct / span_total if span_total > 0 else 0
        sample_acc = 1.0 if all_spans_correct else 0.0

        if sample_acc == 1.0:
            exact_match_inputs_index.append(i)
            exact_match_preds.append(pred_ids)
            exact_match_targets.append(target_ids)

        batch_token_correct.append(token_acc)
        batch_span_correct.append(span_acc)
        batch_sample_correct.append(sample_acc)

        if verbose:
            print(f"Token-level Accuracy: {token_acc*100:.2f}%")
            print(f"Span-level Accuracy : {span_acc*100:.2f}%")
            print(f"Sample-level Accuracy: {sample_acc*100:.2f}%")
            print("==============================\n")

    if exact_match_inputs_index:
        save_exact_match_samples_as_txt(
            lang,
            corrupted_inputs_batch,
            exact_match_preds,
            exact_match_targets,
            exact_match_inputs_index,
            tokenizer,
            memorization_path
        )

    return {
        "token_level": batch_token_correct,
        "span_level": batch_span_correct,
        "sample_level": batch_sample_correct
    }


def random_sample_and_corrupt_batch(
    text_list,
    tokenizer,
    num_tokens=100,
    corruption_rate=0.15,
    mean_span_length=3,
    seed=None
):
    """
    Randomly sample text chunks and create masked corruption for T5 denoising.

    Args:
        text_list (List[str]): Raw text samples.
        tokenizer: Tokenizer to encode text.
        num_tokens (int): Length of sampled sequences.
        corruption_rate (float): Fraction of tokens to mask.
        mean_span_length (int): Average length of mask spans.
        seed (int): Random seed.

    Returns:
        corrupted_inputs_batch (torch.Tensor): Inputs with <extra_id> placeholders.
        target_outputs_batch (torch.Tensor): Targets containing masked spans.
        selected_texts (List[str]): Raw decoded selected texts.
    """
    batch = tokenizer(text_list, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    input_ids_batch = batch['input_ids'].to("cuda")

    corrupted_inputs = []
    target_outputs = []
    selected_texts = []

    batch_size, max_seq_len = input_ids_batch.shape

    for i in range(batch_size):
        input_ids = input_ids_batch[i]
        seq_len = (input_ids != tokenizer.pad_token_id).sum().item()
        input_ids = input_ids[:seq_len]

        if seq_len < num_tokens:
            continue

        if seed is not None:
            random.seed(seed + i)

        start_idx = random.randint(0, seq_len - num_tokens)
        selected_ids = input_ids[start_idx:start_idx + num_tokens]

        selected_text = tokenizer.decode(selected_ids, skip_special_tokens=False)
        seq_len = selected_ids.shape[0]
        num_tokens_to_mask = int(seq_len * corruption_rate)
        num_spans = max(1, int(num_tokens_to_mask / mean_span_length))

        lengths = [1] * num_spans
        remaining = num_tokens_to_mask - num_spans

        while remaining > 0:
            idx = random.randint(0, num_spans - 1)
            lengths[idx] += 1
            remaining -= 1

        available_positions = list(range(seq_len))
        spans = []

        for span_len in lengths:
            possible_starts = [
                pos for pos in available_positions
                if all((pos + offset) in available_positions for offset in range(span_len))
            ]

            if not possible_starts:
                raise ValueError(f"Cannot insert span of length {span_len}, no space available!")

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
        raise ValueError("No samples to process, all skipped!")

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

def summarize_global_metrics(token_accs, span_accs, sample_accs):

    token_mean, token_ci = compute_mean_and_confidence_interval(token_accs)
    span_mean, span_ci = compute_mean_and_confidence_interval(span_accs)
    sample_mean, _ = compute_mean_and_confidence_interval(sample_accs)

    total_samples = len(sample_accs)
    sample_full_correct = sum(sample_accs)
    print(f"[DEBUG] len(sample_accs): {len(lang_sample_accs)}")

    print(f"==== Final Evaluation on {total_samples} Samples ====")
    print(f"Token-Level Accuracy: {token_mean*100:.2f}% (95% CI: {token_ci[0]*100:.2f}% ~ {token_ci[1]*100:.2f}%)")
    print(f"Span-Level Accuracy : {span_mean*100:.2f}% (95% CI: {span_ci[0]*100:.2f}% ~ {span_ci[1]*100:.2f}%)")
    print(f"Sample-Level Accuracy: {sample_mean*100:.2f}%")
    print(f"Total Fully Correct Samples: {sample_full_correct}/{total_samples} ({(sample_full_correct/total_samples)*100:.2f}%)")
    print("========================================\n")

    return {
        "token_mean": token_mean,
        "token_ci": token_ci,
        "span_mean": span_mean,
        "span_ci": span_ci,
        "sample_mean": sample_mean,
        "total_samples": total_samples,
        "fully_correct_samples": sample_full_correct
    }



os.makedirs(result_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

final_results = {}
result_output_path = os.path.join(result_dir, f"T5style_memorization_{model_id}_{MAX_TOKEN}_{PREFIX}_{timestamp}.json")

for lang in languages:
    print(f"\n🌍 Processing Language: {lang}")
    file_path = os.path.join(data_dir, f"{lang}.jsonl")

    with open(file_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f]
        samples = samples[:SAMPLE_NUMBERS]
    lang_token_accs = []
    lang_span_accs = []
    lang_sample_accs = []


    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"🔍 {lang} (greedy decode)"):
        batch_samples = samples[i:i + BATCH_SIZE]
        text_list = [item["text"] for item in batch_samples]

        for repeat_idx, seed in enumerate(SEED_LIST):  # 5 seed
            try:
                corrupted_inputs_batch, target_outputs_batch, selected_texts = random_sample_and_corrupt_batch(
                    text_list,
                    tokenizer,
                    num_tokens=MAX_TOKEN,
                    corruption_rate=SUFIX_RATIO,
                    mean_span_length=3,
                    seed=seed
                )
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids=corrupted_inputs_batch,
                        do_sample=False,
                        #min_length=15,
                        num_beams=1,
                        repetition_penalty=1.2,
                    )

                batch_result = evaluate_predictions_token_level_precisely(
                    lang,output_ids, target_outputs_batch,corrupted_inputs_batch,
                    max_extra_id=5,
                    tokenizer=tokenizer,
                    verbose=False
                )

                lang_token_accs += batch_result["token_level"]
                lang_span_accs += batch_result["span_level"]
                lang_sample_accs += batch_result["sample_level"]

            except Exception as e:
                print(f"⚠️ Skip batch repeat {repeat_idx} (seed={seed}) due to error: {e}")
                continue


    lang_summary = summarize_global_metrics(lang_token_accs, lang_span_accs, lang_sample_accs)


    final_results[lang] = lang_summary

    with open(result_output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)


with open(result_output_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"✅ Final summary for ALL languages saved to {result_output_path}")
