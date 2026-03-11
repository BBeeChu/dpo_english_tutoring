# DPO-Based English Tutoring Feedback Generation

A research pipeline for fine-tuning small language models (SLLMs) with **Direct Preference Optimization (DPO)** to generate higher-quality tutoring feedback in English reading comprehension lessons.

---

## Overview

In an intelligent tutoring system, when a student provides an incorrect answer to a reading comprehension question, the model must generate a **one-sentence hint** that guides the student toward the correct answer — without directly revealing it. This project uses DPO to train models to prefer high-quality feedback (chosen) over low-quality feedback (rejected), where preference labels are derived from GPT-based quality scoring across five pedagogical criteria.

---

## Pipeline

```
1. SLLM Feedback Generation        small_llm_feedback_generation.py
         ↓
2. Data Aggregation                data_analysis.ipynb
         ↓
3. GPT-Based Quality Scoring       feedback_evaluation.py
         ↓
4. DPO Dataset Construction        criteria_dpo_{train,dev,test}_feedback_data.json
         ↓
5. DPO Fine-Tuning & Inference     dpo_llm_feedback_generation.py
         ↓
6. Evaluation                      result_evaluation.ipynb
                                   faithfulness_calculation.py
                                   i_calculation.py
```

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `transformers` | 4.37.2 | Model loading & tokenization |
| `torch` | ≥ 2.0.0 | Deep learning framework |
| `peft` | — | LoRA fine-tuning |
| `bitsandbytes` | — | LLM quantization |
| `accelerate` | — | Multi-GPU training |
| `datasets` | — | HuggingFace dataset handling |
| `wandb` | — | Experiment tracking |

---

## Project Structure

```
dpo_english_tutoring/
├── data/
│   ├── article-id_mapping.json               # Article ID → passage text
│   ├── original_direct_train_data.json        # Raw tutoring dialogues (train)
│   ├── original_direct_dev_data.json          # Raw tutoring dialogues (dev)
│   ├── sllm_augmented_train_data.json         # Merged SLLM-generated feedback
│   ├── criteria_dpo_train_feedback_data.json  # DPO training pairs (chosen/rejected)
│   ├── criteria_dpo_dev_feedback_data.json    # DPO dev pairs
│   └── criteria_dpo_test_feedback_data.json   # DPO test pairs
├── sllm_generated_data/
│   └── *.json                                 # Raw feedback from each SLLM
├── results/
│   └── {model}/dpo/{mode}/seed_{seed}/        # Predictions and evaluation results
├── small_llm_feedback_generation.py
├── dpo_llm_feedback_generation.py
├── feedback_evaluation.py
├── faithfulness_calculation.py
├── i_calculation.py
├── gpt_model.py
├── data_analysis.ipynb
├── result_evaluation.ipynb
├── deepspeed_config.json
└── requirements.txt
```

---

## Step-by-Step Usage

### Step 1. Generate Feedback with Small LLMs

```bash
python small_llm_feedback_generation.py --model llama_8 --seed 42
```

**Supported models:**

| Argument | HuggingFace Model ID |
|----------|----------------------|
| `llama_1` | `meta-llama/Llama-3.2-1B-Instruct` |
| `llama_3` | `meta-llama/Llama-3.2-3B-Instruct` |
| `llama_8` | `meta-llama/Llama-3.1-8B-Instruct` |
| `qwen_1` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen_3` | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen_7` | `Qwen/Qwen2.5-7B-Instruct` |

- Input: `data/original_direct_dev_data.json`, `data/article-id_mapping.json`
- Output: `dev_{model}_{seed}_generated_data.json`
- Falls back to `gpt-4o` automatically if local inference fails 3 times.

---

### Step 2. Aggregate SLLM Feedback

Run `data_analysis.ipynb` to merge all SLLM-generated feedback into a single unified file.

- Output: `data/sllm_augmented_train_data.json`
- Feedback index mapping per dialogue turn:
  `[0]=gpt4, [1]=gpt3, [2]=llama_70, [3]=llama_8, [4]=qwen_72, [5]=gemma, [6]=llama_1, [7]=llama_3, [8]=qwen_1, [9]=qwen_3, [10]=qwen_7`

---

### Step 3. GPT-Based Quality Scoring

Score each feedback candidate with GPT across five pedagogical criteria:

```bash
python feedback_evaluation.py --model llama_8
```

**Evaluation criteria (0 or 1 each):**

| Criterion | Description |
|-----------|-------------|
| **Correct** | Feedback should be factually accurate  to support the student. |
| **Revealing** | Feedback should avoid explicitly providing the correct answer. |
| **Guidance** | Feedback should offer direction to help the student find the right answer |
| **Diagnostic** | Feedback should address misconceptions or errors made by the student. |
| **Encouragement** | Feedback should convey a positive and supportive tone to motivate the student. |

- Output: `data/test_evaluated_{model}_feedback.json`
- Requires OpenAI API key set in `gpt_model.py`.

---

### Step 4. DPO Fine-Tuning

Fine-tune LLaMA 3.1 8B with DPO using LoRA and DeepSpeed:

```bash
accelerate launch --config_file deepspeed_config.json \
    dpo_llm_feedback_generation.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --seed 42 \
    --epochs 1 \
    --learning_rate 1e-4 \
    --train_batch_size 32 \
    --eval_steps 20 \
    --patience 2
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `meta-llama/Llama-3.1-8B-Instruct` | Base model |
| `--seed` | `42` | Random seed (loops 42–46) |
| `--epochs` | `1` | Training epochs |
| `--learning_rate` | `1e-4` | Learning rate |
| `--train_batch_size` | `32` | Train batch size |
| `--eval_steps` | `20` | Evaluation frequency (steps) |
| `--patience` | `2` | Early stopping patience |
| `--mode` | `criteria` | Dataset/experiment mode |
| `--test` | — | Skip training, run inference only |
| `--debug` | — | Quick debug run (10 samples, 1 epoch) |

**LoRA Configuration:**
- Rank `r=16`, alpha `32`
- Target modules: `q_proj`, `v_proj`
- Dropout: `0.05`

**DPO Configuration:**
- `beta=0.5`, `bf16=True`, `max_grad_norm=1.0`, `weight_decay=0.05`

**Outputs** (per seed):
```
results/{model}/dpo/criteria/seed_{seed}/
├── feedback_predictions.pkl
├── feedback_true_labels.pkl
└── best_checkpoint_path.txt
```

---

### Step 5. Evaluation

#### 5a. Automatic Metrics (ROUGE, BLEU, BERTScore)

Run `result_evaluation.ipynb`.

Metrics computed: **ROUGE-L**, **METEOR**, **BLEU**, **BERTScore** (with baseline rescaling).

#### 5b. Faithfulness Score

Measures how faithful generated feedback is to the source passage using T5-NLI:

```bash
python faithfulness_calculation.py --version dpo --seed 42
```

Score formula: `(1 + min(p_entailment, 0.5) - p_contradiction) / 1.5`

Output: `results/.../dpo_faithfulness_results.json`

#### 5c. I-Score (Alignment with Human Feedback)

Measures how closely predicted feedback aligns with human feedback patterns using T5-NLI:

```bash
python i_calculation.py --version dpo --seed 42
```

Score formula: `1 - |p_pred_entailment - p_human_entailment|` (higher = more similar to human)

Output: `results/.../dpo_i_results.json`

---

## Models Used

| Model | Role |
|-------|------|
| `meta-llama/Llama-3.1-8B-Instruct` | Primary DPO fine-tuning target |
| `meta-llama/Llama-3.2-{1,3}B-Instruct` | Small baseline models |
| `Qwen/Qwen2.5-{1.5,3,7}B-Instruct` | Small baseline models |
| GPT-4 / GPT-3.5 | Reference feedback & fallback generation |
| GPT-4o | Fallback when local inference fails |
| GPT-5.1 | LLM judge for quality scoring |
| `t5-base` | NLI-based faithfulness & I-score evaluation |

---

## Datasets from Prior Studies

The following data files are sourced from prior published datasets and must be obtained separately:

| File | Source | Description |
|------|--------|-------------|
| `data/original_direct_train_data.json` | [DIRECTDataset](https://github.com/DIRECTDataset) | Training split of the DIRECT tutoring dialogue dataset |
| `data/original_direct_dev_data.json` | [DIRECTDataset](https://github.com/DIRECTDataset) | Validation split of the DIRECT tutoring dialogue dataset |
| `data/article-id_mapping.json` | [DIRECTDataset/DIRECTFeedback](https://github.com/DIRECTDataset/DIRECTFeedback) | Mapping from article IDs to reading passage texts (derived from RACE) |

### DIRECT Dataset

The tutoring dialogue data (`original_direct_*.json`) originates from the **DIRECT** (Dialogue-based Reading Comprehension Tutoring) dataset:

> Hyein Seo, Taewook Hwang, Yohan Lee, and Sangken Jung. 2025. FEAT: A Preference Feedback Dataset through a Cost‑Effective Auto‑Generation and Labeling Framework for English AI Tutoring. In Proceedings of the 2025 Association for Computational Linguistics (ACL 2025).


### DIRECT-F Dataset

The article mapping file (`article-id_mapping.json`) is provided as part of the **DIRECT-F** dataset extension, released alongside:

> Wencke Liermann, Jin-Xia Huang, Yohan Lee, and Kong Joo Lee. 2024. *More Insightful Feedback for Tutoring: Enhancing Generation Mechanisms and Automatic Evaluation.* In Proceedings of EMNLP 2024, pages 10838–10851, Miami, Florida, USA.

This file is derived from the **RACE** dataset ([Lai et al., 2017](https://aclanthology.org/D17-1082/)) and is provided solely for non-commercial research purposes, subject to the terms of use of both RACE and DIRECT-F (CC BY-NC-SA 4.0).

> Please refer to [DIRECTDataset/DIRECTFeedback](https://github.com/DIRECTDataset/DIRECTFeedback) for the full dataset description, license terms, and download instructions.

---

## Configuration

**`deepspeed_config.json`** — DeepSpeed ZeRO Stage 2 configuration for multi-GPU DPO training:
- `bf16` precision enabled
- `allgather_partitions`, `reduce_scatter`, `contiguous_gradients` for memory efficiency

**`gpt_model.py`** — Set your OpenAI API key before running evaluation scripts:
```python
# gpt_model.py
openai.api_key = "YOUR_API_KEY"
```

---

## Citation

To be shared upon accepted.
