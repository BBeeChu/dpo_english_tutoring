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

    ds = json.load(open("./data/article-id_mapping.json"))
    pred_feedback = pd.read_pickle(f"./results/Llama-3.1-8B-Instruct/dpo/criteria/seed_{args.seed}/feedback_predictions.pkl")
    
    dev_feedback = json.load(open(f"./data/original_direct_dev_data.json"))

    results = []
    pred_feedback_id = 0
    for feedback_info in tqdm(dev_feedback):
        ds_id = feedback_info[0]
        for feedback_data in feedback_info[1]:
            llama_3_baseline_feedback = feedback_data["llama_8_feedback"]
            llama_3_pred_feedback = pred_feedback[pred_feedback_id]
            if args.version == "dpo":
                input_format = f"Judge whether the hypothesis is entailed by the premise. Answer with 'entailment' or 'contradiction'.\n\nPremise: {ds[ds_id]}\nHypothesis: {llama_3_pred_feedback}"
            else:
                input_format = f"Judge whether the hypothesis is entailed by the premise. Answer with 'entailment' or 'contradiction'.\n\nPremise: {ds[ds_id]}\nHypothesis: {llama_3_baseline_feedback}"

            pred_onset_encoding = tokenizer(
                                        [
                                            input_format
                                        ],
                                        padding="longest",
                                        return_tensors="pt"
                                    )
            pred_onset_input_ids = pred_onset_encoding.input_ids.cuda()
            pred_onset_attention_mask = pred_onset_encoding.attention_mask.cuda()


            expected_bos_token = tokenizer(
                                    ["entailment", "contradiction"],
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

                

            final_faithfulness = (1+np.min([pred_true_prob[0], 0.5])-pred_true_prob[1])/1.5
            results.append(final_faithfulness)

            pred_feedback_id += 1

            if args.version == "dpo":
                with open(f"./results/Llama-3.1-8B-Instruct/dpo/criteria/seed_{args.seed}/dpo_faithfulness_results.json", "w") as file:
                    json.dump(results, file)
        

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, default="dpo", help="model version")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    
    print(f"Processing seed {args.seed}...")
    main(args)
