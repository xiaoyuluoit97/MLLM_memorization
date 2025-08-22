# MLLM memorization

## Files

- **dataset_clean.py**  
  Cleans the dataset. Use this file first to prepare the data.

- **mgpt.py**  
  Runs the mGPT model and calculates:
  - EM (Exact Match)
  - RM (BLEU score)
  - RM (ROUGE-L score)

- **mt5.py**  
  Runs the mT5 model and calculates:
  - EM (Exact Match)

- **logprob_gpt.py**  
  Calculates PM (probability score, or log probability) using GPT.

- **logprob_mt5.py**  
  Calculates PM (probability score, or log probability) using mT5.

## How to Use

1. Run `dataset_clean.py` to clean & filterd mC4 data.
2. Run `mgpt.py` or `mt5.py` to generate results and get EM, BLEU, and ROUGE-L scores.
3. Run `logprob_gpt.py` or `logprob_mt5.py` to get PM scores.

## Citation

This work has been accepted to **EMNLP 2025 (Main Conference, Suzhou, China)**.  
If you find this code useful, please cite:  

```bibtex
@inproceedings{luo2025shared,
  title        = {Shared Path: Unraveling Memorization in Multilingual LLMs through Language Similarities},
  author       = {Luo, Xiaoyu and Chen, Yiyi and Bjerva, Johannes and Li, Qiongxiu},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year         = {2025},
  address      = {Suzhou, China},
}
