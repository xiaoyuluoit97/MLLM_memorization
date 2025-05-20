import os
import re
import json
import hashlib
import unicodedata
from datasets import load_dataset
from tqdm import tqdm
import cld3


OUTPUT_BASE_PATH = "dataset/longone/"
SAMPLES_PER_LANGUAGE = 50000

os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)



def get_hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()


def is_too_numeric(text, threshold=0.3):
    tokens = re.findall(r"\w+", text)
    if not tokens:
        return False
    num_digits = sum(1 for token in tokens if token.isdigit())
    return num_digits / len(tokens) > threshold


def has_long_number_sequence(text, min_len=30):
    pattern = r"(?:(?:\d{2,})[\s\|,]{1,}){%d,}" % min_len
    return bool(re.search(pattern, text))


def check_clean(text: str, expected_lang: str = "en", min_lang_ratio: float = 0.9):
    text = text.strip()
    if not text or len(text) < 600:
        return False, "too_short"
    if any(bad in text for bad in ["#", "<", ">", "©", "→", "}", "{", "null"]):
        return False, "bad_char"
    if re.search(r"[�]{2,}", text):
        return False, "garbled"
    if re.search(r"(.)\1{10,}", text):
        return False, "repeat_char"
    if is_too_numeric(text):
        return False, "too_numeric"
    if has_long_number_sequence(text):
        return False, "long_number_seq"
    if re.search(r"https?://", text):
        return False, "contains_url"

    lang_results = cld3.get_frequent_languages(text, 2)
    if not lang_results:
        return False, "lang_detect_fail"

    top_lang = lang_results[0]
    if top_lang.language != expected_lang:
        return False, "lang_mismatch"
    if top_lang.probability < 0.9:
        return False, "lang_low_confidence"
    if top_lang.proportion < min_lang_ratio:
        return False, "lang_low_proportion"

    return True, None




def process_language(lang_code, lang_name, samples_per_language):
    print(f"\n🌍 Processing language: {lang_name} ({lang_code})")

    try:
        dataset = load_dataset("allenai/c4", lang_code, streaming=True, split="train")
        dataset.shuffle(buffer_size=5000000)
    except Exception as e:
        print(f"❌ Failed to load language '{lang_code}': {e}")
        return

    seen_hashes = set()
    selected_samples = []
    filter_reasons = {
        "too_short": 0,
        "bad_char": 0,
        "garbled": 0,
        "repeat_char": 0,
        "too_numeric": 0,
        "long_number_seq": 0,
        "contains_url": 0,
        "lang_detect_fail": 0,
        "lang_mismatch": 0,
        "lang_low_confidence": 0,
        "lang_low_proportion": 0,
    }

    for sample in tqdm(dataset, desc=f"🔍 {lang_code}", total=samples_per_language):
        text = sample.get("text", "")
        h = get_hash(text)

        if h in seen_hashes:
            continue

        is_ok, reason = check_clean(text, expected_lang=lang_code)
        if not is_ok:
            if reason in filter_reasons:
                filter_reasons[reason] += 1
            continue

        seen_hashes.add(h)
        selected_samples.append({
            "id": f"{lang_code}_{len(selected_samples):04}",
            "lang": lang_code,
            "text": text.strip(),
        })

        if len(selected_samples) >= samples_per_language:
            break


    print(f"\n✅ Collected {len(selected_samples)} clean samples for {lang_code}")
    print(f"🧼 Filtered samples:")
    for reason, count in filter_reasons.items():
        print(f"   - {reason:18s}: {count}")


    save_path = os.path.join(OUTPUT_BASE_PATH, f"{lang_code}.jsonl")
    with open(save_path, "w", encoding="utf-8") as f:
        for record in selected_samples:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"📁 Saved to: {save_path}")




def main(languages: dict, samples_per_language: int = SAMPLES_PER_LANGUAGE):
    for lang_code, lang_name in languages.items():
        process_language(lang_code, lang_name, samples_per_language)




if __name__ == "__main__":
    languages = {
    "ny": "Nyanja",
    "ps": "Pashto",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sl": "Slovenian",
    "sm": "Samoan",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    }
    main(languages)

