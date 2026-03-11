from deepinfra_model import LlamaModel
from rmts_gpt_model import GPTModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BartForConditionalGeneration, PegasusForConditionalGeneration
import pickle
import pandas as pd
import json
from tqdm import tqdm
import argparse
import ast
import re
import os

 
def construct_prompt(passage, correct_answer, dialogue):
    system_message = """
    You are a proficient tutoring assistant who provides just a few clues to an user in the correct direction. 
    The user should understand the following passage and then answer your question.
    """
    user_message = f"""
    Passage: {passage}
    
    The correct answer is “{correct_answer}”, but the user don ́ t answer correctly as the following tutoring dialogues. 
    Generate an indirect one sentence feedback or hint to guide the user to find the answer on him/her own. 
    
    {dialogue}
    """
    
    prompt_template = [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_message.strip()}
    ]
    return prompt_template


def main(args):
    
    ds = json.load(open("./data/article-id_mapping.json"))
    data = json.load(open("./data/original_direct_train_data.json"))

    if args.model == "llama_3":
        # response = llama_model.ask_chatgpt(prompt, temperature=0.0, model="meta-llama/Llama-3.2-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    elif args.model == "llama_8":
        # response = llama_model.ask_chatgpt(prompt, temperature=0.0, model="meta-llama/Llama-3.2-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    
    elif args.model == "llama_1":
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        
    elif args.model == "qwen_3":
        # response = llama_model.ask_chatgpt(prompt, temperature=0.0, model="Qwen/Qwen2.5-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    
    elif args.model == "qwen_7":
        # response = llama_model.ask_chatgpt(prompt, temperature=0.0, model="Qwen/Qwen2.5-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    elif args.model == "qwen_1":
        # response = llama_model.ask_chatgpt(prompt, temperature=0.0, model="Qwen/Qwen2.5-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    
    

    processed_data_list = []
    for d in tqdm(data):
        passage = ds[d[0]]
        for tutor_info in d[1]:
            conversatoin_history = tutor_info["history"]
            conversation = ""
            for speaker, utterance in conversatoin_history:
                conversation += f"{speaker}: {utterance}\n"
            processed_data_list.append({
                "passage": passage,
                "dialogue": conversation.strip(),
                "correct_answer": tutor_info["reference"]
            })
    
    
            prompt = construct_prompt(passage, tutor_info["reference"], conversation.strip())
        
            llama_model = LlamaModel()
            gpt_model = GPTModel()
            
            patience = 0
            while patience < 3:
                try: 
                    
                    
                    text = tokenizer.apply_chat_template(
                            prompt,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                    generated_ids = model.generate(
                                        **model_inputs,
                                        max_new_tokens=512,
                                        temperature=0.7,
                                        top_p=0.9,
                                        do_sample=True,
                                    )
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                

                    tutor_info[f"{args.model}_feedback"] = response
                    
                    with open(f"./dev_{args.model}_{args.seed}_generated_data.json", "w") as f:
                        json.dump(data, f, indent=4)
                        
                    break
                except Exception as e:
                    print(f"Error processing: {e}")
                    patience += 1
                    
                    if patience >= 3:
                        print(f"Failed to process after multiple attempts, and switching the LLM model to gpt-4o.")
                        new_patience = 0
                        while new_patience < 3:
                            try: 
                                print("Switching the LLM model to gpt-4o for retry...")
                                response = gpt_model.ask_chatgpt(prompt, seed=42, temperature=0.0, model="gpt-4o")
                                tutor_info[f"{args.model}_feedback"] = response[0]

                                with open(f"./{args.model}_{args.seed}_generated_data.json", "w") as f:
                                    json.dump(data, f, indent=4)
                                    
                                break
                            except Exception as e:
                                print(f"Still not working : {e}")
                                new_patience += 1

                                if new_patience >= 3:
                                    print(f"Failed to process after multiple attempts with gpt-4o. Recording 'None' as result.")
                                    tutor_info[f"{args.model}_feedback"] = "None"
                                with open(f"./{args.model}_{args.seed}_generated_data.json", "w") as f:
                                    json.dump(data, f, indent=4)
    
                        
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract snippets from essays based on analysis.")
    parser.add_argument('--model', type=str, default="llama_70", help="Model name or path.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    
    print(f"Running with {args.model} and seed {args.seed}...")

    results = main(args)