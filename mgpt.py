import torch
import numpy as np
import random
import re
import scipy.stats as st
from transformers import MT5Tokenizer, GPT2LMHeadModel, TextGenerationPipeline,AutoTokenizer,AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from evaluate import load as load_metric
#from bert_score import score as bert_score_fn
from tqdm import tqdm
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
from datetime import datetime
SAMPLE_NUMBERS = 50000
SEED_LIST = [10777140]
languages =['af', 'am', 'be', 'bg', 'ca', 'ceb', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'et', 'eu', 'fi', 'fil', 'fr','ga', 'gd', 'gl', 'gu', 'ha', 'hi', 'hu', 'hy', 'id', 'ig', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn','ko', 'ky', 'lb', 'lo', 'lt', 'lv', 'mi', 'mk', 'ml', 'mr', 'mt', 'my', 'ne', 'nl', 'ny', 'pa', 'pl', 'pt','ro', 'ru', 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th','tr', 'uk', 'ur', 'vi', 'xh', 'yo', 'zu', 'ar', 'zh', 'fa', 'no', 'es', 'ht', 'ms', 'sq', 'ku', 'yi', 'uz','ps', 'mg', 'az', 'bn', 'iw']


BATCH_SIZE = 100
MAX_TOKEN = 100
SUFFIX_RATIO = 0.15
#MODEL_NAME = "THUMT/mGPT"
MODEL_NAME = "ai-forever/mGPT"
model_id = MODEL_NAME.split("/")[-1]
MGPT_NEW = True

base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "dataset", "longone")
result_dir = os.path.join(base_dir, "results")
memorization_path = os.path.join(result_dir, "memorization_sample", model_id)
#memorization_path = os.path.join(result_dir, "memorization_sample", "mgpt61")
cache_path = os.path.join(base_dir, "model")

for path in [data_dir, result_dir, memorization_path, cache_path]:
    os.makedirs(path, exist_ok=True)
    
model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME,
    cache_dir=cache_path,
    torch_dtype=torch.float16
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_path, use_fast=False)

#tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_path)
#model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, cache_dir=cache_path).to("cuda")
model.eval()
# ==== PATH ====
os.makedirs(result_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
result_output_path = os.path.join(result_dir, f"MISSEDazfatguz_memorization_{MODEL_NAME.split('/')[-1]}_{MAX_TOKEN}_{int(SUFFIX_RATIO*100)}_{timestamp}.json")
final_results = {}


def evaluate_generation_metrics_per_sample(predicted_batch, target_batch, tokenizer, verbose=True):
    """
    Evaluate text generation quality per sample using BLEU, ROUGE, and token-level accuracy.

    Args:
        predicted_batch (List[torch.Tensor]): List of predicted token ID sequences.
        target_batch (List[torch.Tensor]): List of reference token ID sequences.
        tokenizer: Tokenizer for decoding tokens.
        verbose (bool): Whether to print per-sample details.

    Returns:
        results (List[dict]): List of dictionaries with metrics for each sample.
        exact_match_indices (List[int]): Indices where prediction exactly matches reference.
    """
    rouge_metric = load_metric("rouge")
    smoothing = SmoothingFunction().method1

    results = []
    exact_match_indices = []
    predictions_text = []
    references_text = []

    for i in range(len(predicted_batch)):
        pred_ids = predicted_batch[i].tolist()
        tgt_ids = target_batch[i].tolist()
        tgt_ids = [t for t in tgt_ids if t != tokenizer.pad_token_id]

        #pred_ids = pred_ids[:len(tgt_ids)]

        token_acc = len(set(pred_ids) & set(tgt_ids)) / len(set(tgt_ids)) if tgt_ids else 0
        exact = 1.0 if pred_ids == tgt_ids else 0.0
        if exact == 1.0:
            exact_match_indices.append(i)

        ref_tokens = tokenizer.convert_ids_to_tokens(tgt_ids)
        hyp_tokens = tokenizer.convert_ids_to_tokens(pred_ids)

        ref_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)
        hyp_text = tokenizer.decode(pred_ids, skip_special_tokens=True)

        ref_id_text = " ".join(str(t) for t in pred_ids)
        hyp_id_text = " ".join(str(t) for t in tgt_ids)

        predictions_text.append(hyp_id_text)
        references_text.append(ref_id_text)

        bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)

        if verbose:
            print(f"==== Sample {i} ====")
            print("Ref:", ref_text)
            print("Hyp:", hyp_text)
            print(f"BLEU: {bleu:.4f} | Token Acc: {token_acc:.4f} | Exact Match: {exact}")
            print("------------------------------")

        results.append({
            "bleu": bleu,
            "token_accuracy": token_acc,
            "exact_match": exact,

        })

   
    rouge_result = rouge_metric.compute(
        predictions=predictions_text,
        references=references_text,
        use_stemmer=True,
        use_aggregator=False
    )

    #P, R, F1 = bert_score_fn(predictions_text, references_text, lang="en", verbose=False)

    for i in range(len(results)):
        results[i]["rouge1"] = rouge_result["rouge1"][i]
        results[i]["rouge2"] = rouge_result["rouge2"][i]
        results[i]["rougeL"] = rouge_result["rougeL"][i]
        #results[i]["bertscore_P"] = P[i].item()
        #results[i]["bertscore_R"] = R[i].item()
        #results[i]["bertscore_F1"] = F1[i].item()
        results[i]["bertscore_P"] = None
        results[i]["bertscore_R"] = None
        results[i]["bertscore_F1"] = None

    return results, exact_match_indices


def compute_mean_and_confidence_interval(data, confidence=0.95):
    """
    Compute the mean and confidence interval of a list of numerical values.

    Args:
        data (List[float]): List of metric values.
        confidence (float): Confidence level for the interval.

    Returns:
        mean (float): Mean value.
        ci (Tuple[float, float]): Lower and upper bounds of the confidence interval.
    """
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
    """
    Aggregate and print summary statistics (mean and confidence intervals) for all metrics.

    Args:
        results (List[dict]): Per-sample evaluation metrics.

    Returns:
        summary (dict): Summary statistics and counts of exact matches.
    """
    def extract(key): return [r[key] for r in results]

    metrics_to_summarize = [
        "token_accuracy", "exact_match", "bleu",
        "rouge1", "rouge2", "rougeL",
        "bertscore_P", "bertscore_R", "bertscore_F1"
    ]

    summary = {}

    print("==== Global Evaluation Summary ====")
    for key in metrics_to_summarize:
        values = extract(key)
        mean, ci = compute_mean_and_confidence_interval(values)
        summary[key] = {
            "mean": mean,
            "ci": ci
        }
        print(f"{key:<15}: {mean*100:.2f}% (95% CI: {ci[0]*100:.2f}% ~ {ci[1]*100:.2f}%)")

 
    total_samples = len(results)
    full_match = sum(r["exact_match"] == 1.0 for r in results)
    print(f"Exact Match Samples : {full_match}/{total_samples} ({(full_match/total_samples)*100:.2f}%)")
    print("===================================\n")

    summary["total_samples"] = total_samples
    summary["fully_correct_samples"] = full_match

    return summary


def save_exact_match_samples_as_txt(
    lang,
    input_texts,
    predicted_batch,
    target_batch,
    exact_match_indices,
    tokenizer,
    save_dir="emnlp25/results/memorization_sample"
):
    """
    Save examples where the model's output exactly matches the reference to a text file.

    Args:
        lang (str): Language identifier.
        input_texts (torch.Tensor): Original input prefix IDs.
        predicted_batch (List[torch.Tensor]): Generated output IDs.
        target_batch (torch.Tensor): Reference output IDs.
        exact_match_indices (List[int]): Indices of exact matches.
        tokenizer: Tokenizer for decoding.
        save_dir (str): Directory to save text files.
    """
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"exact_match_samples_{lang}.txt")

    with open(save_path, "a", encoding="utf-8") as f_out:
        for idx in exact_match_indices:
            sample_input = tokenizer.decode(input_texts[idx], skip_special_tokens=True)
            reference = tokenizer.decode(target_batch[idx], skip_special_tokens=True)
            prediction = tokenizer.decode(predicted_batch[idx], skip_special_tokens=True)

            #print("📌 Exact Match Sample:")
            #print("Input:", sample_input)
            #print("Ref:", reference)
            #print("Pred:", prediction)
            #print("=" * 50)

            f_out.write(f"Input: {sample_input}\n")
            f_out.write(f"Reference: {reference}\n")
            f_out.write(f"Prediction: {prediction}\n")
            f_out.write("=" * 50 + "\n")



def random_sample_prefix_suffix_batch(
    text_list, tokenizer, num_tokens=100, suffix_ratio=0.15, seed=None
):
    """
    Randomly sample fixed-length sequences from text as prefix and suffix pairs.

    Args:
        text_list (List[str]): Raw text samples.
        tokenizer: Tokenizer to encode text.
        num_tokens (int): Total tokens to sample per example.
        suffix_ratio (float): Fraction of tokens used as suffix target.
        seed (int): Random seed for reproducibility.

    Returns:
        prefix_inputs_padded (torch.Tensor): Batch of prefix token IDs.
        target_outputs_padded (torch.Tensor): Batch of suffix token IDs.
        selected_texts (List[str]): Decoded sampled sequences.
    """

   
    batch = tokenizer(text_list, return_tensors="pt", padding=True, add_special_tokens=False,truncation=True, max_length=2048)
    input_ids_batch = batch['input_ids'].to("cuda")

    prefix_inputs = []
    target_outputs = []
    selected_texts = []

    batch_size, max_seq_len = input_ids_batch.shape

    for i in range(batch_size):
        input_ids = input_ids_batch[i]
        if MGPT_NEW:
            input_ids = input_ids[input_ids != tokenizer.pad_token_id]
            seq_len = len(input_ids)
        else:
            seq_len = (input_ids != tokenizer.pad_token_id).sum().item()
            input_ids = input_ids[:seq_len]

        if seq_len < num_tokens:
            continue  # skip

        if seed is not None:
            random.seed(seed + i)

        start_idx = random.randint(0, seq_len - num_tokens)
        selected_ids = input_ids[start_idx: start_idx + num_tokens]

        suffix_len = int(num_tokens * suffix_ratio)
        prefix_len = num_tokens - suffix_len

        prefix_ids = selected_ids[:prefix_len]
        suffix_ids = selected_ids[prefix_len:]

        prefix_inputs.append(prefix_ids)
        target_outputs.append(suffix_ids)
        selected_texts.append(tokenizer.decode(selected_ids, skip_special_tokens=False))

    if len(prefix_inputs) == 0:
        raise ValueError("no sample, all skip")

    
    prefix_inputs_padded = torch.nn.utils.rnn.pad_sequence(prefix_inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_outputs_padded = torch.nn.utils.rnn.pad_sequence(target_outputs, batch_first=True, padding_value=tokenizer.pad_token_id)

    return prefix_inputs_padded, target_outputs_padded, selected_texts


def batch_strip_prefix_and_cleanup(
    output_ids_batch, prefix_inputs_batch, tokenizer, remove_token_ids=None
):
    """
    Remove the prefix tokens and cleanup unwanted tokens from generated outputs.

    Args:
        output_ids_batch (torch.Tensor): Model-generated token IDs (prefix + suffix).
        prefix_inputs_batch (torch.Tensor): Input prefix IDs.
        tokenizer: Tokenizer for token IDs.
        remove_token_ids (List[int], optional): Token IDs to filter out from outputs.

    Returns:
        batch_cleaned_outputs (List[torch.Tensor]): Cleaned suffix token sequences.
    """
    batch_cleaned_outputs = []

    if remove_token_ids is None:
        remove_token_ids = [250099]  

    batch_size = output_ids_batch.shape[0]

    for i in range(batch_size):

        prefix_len = (prefix_inputs_batch[i] != tokenizer.pad_token_id).sum().item()


        generated_suffix = output_ids_batch[i][prefix_len:]


        cleaned_suffix = [token_id for token_id in generated_suffix if token_id not in remove_token_ids]

        batch_cleaned_outputs.append(torch.tensor(cleaned_suffix, device=output_ids_batch.device))

    return batch_cleaned_outputs





# ==== MAIN LOOP ====
for lang in languages:
    print(f"\n🌍 Processing Language: {lang}")
    file_path = os.path.join(data_dir, f"{lang}.jsonl")

    if not os.path.exists(file_path):
        print(f"⚠️ File not found for language: {lang}, skipping.")
        continue
            
    with open(file_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f]
        samples = samples[:SAMPLE_NUMBERS]
    results_per_lang = []

    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"🔍 {lang} (gpt decode)"):
        batch_samples = samples[i:i + BATCH_SIZE]
        text_list = [item["text"] for item in batch_samples]

        for repeat_idx, seed in enumerate(SEED_LIST):
            try:

                prefix_batch, target_suffix_batch, _ = random_sample_prefix_suffix_batch(
                    text_list,
                    tokenizer,
                    num_tokens=MAX_TOKEN,
                    suffix_ratio=SUFFIX_RATIO,
                    seed=seed
                )
                with torch.no_grad():

                    output_ids = model.generate(
                        input_ids=prefix_batch,
                        max_new_tokens=target_suffix_batch.shape[1],
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                    )


                predicted_suffix_batch = batch_strip_prefix_and_cleanup(
                    output_ids_batch=output_ids,
                    prefix_inputs_batch=prefix_batch,
                    tokenizer=tokenizer,
                    remove_token_ids= None
                )


                batch_results, exact_match_indices = evaluate_generation_metrics_per_sample(
                    predicted_suffix_batch,
                    target_suffix_batch,
                    tokenizer=tokenizer,
                    verbose=False
                )

                results_per_lang.extend(batch_results)

                save_exact_match_samples_as_txt(
                    lang=lang,
                    input_texts=prefix_batch,
                    predicted_batch=predicted_suffix_batch,
                    target_batch=target_suffix_batch,
                    exact_match_indices=exact_match_indices,
                    tokenizer=tokenizer,
                    save_dir=memorization_path
                )


            except Exception as e:
                print(f"⚠Skipping batch repeat {repeat_idx} (seed={seed}) due to error: {e}")
                continue


    lang_summary = summarize_all_metrics(results_per_lang)
    final_results[lang] = lang_summary


    with open(result_output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"\nFinal summary for ALL languages saved to {result_output_path}")



