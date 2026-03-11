import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import torch
import pandas as pd
import argparse

def main(args):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
    model.eval()
    soft_max = torch.nn.Softmax(dim=2)

    pred_feedback = pd.read_pickle(f"./results/Llama-3.1-8B-Instruct/dpo/criteria/seed_{args.seed}/feedback_predictions.pkl")
    reference = pd.read_pickle(f"./results/Llama-3.1-8B-Instruct/dpo/criteria/seed_{args.seed}/feedback_true_references.pkl")
    dev_feedback = json.load(open(f"./data/original_direct_dev_data.json"))

    human_feedback = []

    for dev_f in dev_feedback:
        feedback_info = dev_f[1]
        for feedback_data in feedback_info:
            human_feedback.append(feedback_data["reply_candidates"][0][-1])

    results = []
    for i in tqdm(range(len(human_feedback))):
        if args.version == "dpo":
            pred_input_format = f"Sentence 1: {reference[i]}\nDoes Sentence 2 support Sentence 1? Sentence 2: {pred_feedback[i]}\nAnswer with True or False."
            target_input_format = f"Sentence 1: {reference[i]}\nDoes Sentence 2 support Sentence 1? Sentence 2: {human_feedback[i]}\nAnswer with True or False."
        else:
            pred_input_format = f"Sentence 1: {reference[i]}\nDoes Sentence 2 support Sentence 1? Sentence 2: {llama_3_baseline_feedback[i]}\nAnswer with True or False."
            target_input_format = f"Sentence 1: {reference[i]}\nDoes Sentence 2 support Sentence 1? Sentence 2: {human_feedback[i]}\nAnswer with True or False."

        pred_onset_encoding = tokenizer(
                                    [
                                        pred_input_format
                                    ],
                                    padding="longest",
                                    return_tensors="pt"
                                )
        pred_onset_input_ids = pred_onset_encoding.input_ids.cuda()
        pred_onset_attention_mask = pred_onset_encoding.attention_mask.cuda()

        target_onset_encoding = tokenizer(
                                    [
                                        target_input_format
                                    ],
                                    padding="longest",
                                    return_tensors="pt"
                                )
        target_onset_input_ids = target_onset_encoding.input_ids.cuda()
        target_onset_attention_mask = target_onset_encoding.attention_mask.cuda()

        expected_bos_token = tokenizer(
                                ["True", "False"],
                                padding=True,
                                return_tensors="pt"
                            ).input_ids[:, 0]
        with torch.no_grad():
            pred_out = model(
                    input_ids=pred_onset_input_ids,
                    attention_mask=pred_onset_attention_mask,
                    labels=torch.ones(
                        (len(pred_onset_input_ids), 1),
                        device=pred_onset_input_ids.device,
                        dtype=pred_onset_input_ids.dtype,
                    ),
                    return_dict=True,
                )
            pred_true_prob = soft_max(
                pred_out.logits
            )[:, 0, expected_bos_token].flatten().tolist()

            target_out = model(
                    input_ids=target_onset_input_ids,
                    attention_mask=target_onset_attention_mask,
                    labels=torch.ones(
                        (len(target_onset_input_ids), 1),
                        device=target_onset_input_ids.device,
                        dtype=target_onset_input_ids.dtype,
                    ),
                    return_dict=True,
                )
            target_true_prob = soft_max(
                target_out.logits
            )[:, 0, expected_bos_token].flatten().tolist()

        final_i = (1-np.abs(pred_true_prob[0]-target_true_prob[0]))
        results.append(final_i)
    if args.version == "dpo":
        with open(f"./results/Llama-3.1-8B-Instruct/dpo/criteria/seed_{args.seed}/dpo_i_results.json", "w") as file:
            json.dump(results, file)
   

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, default="dpo", help="model version")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    
    print(f"Processing seed {args.seed}...")
    main(args)
