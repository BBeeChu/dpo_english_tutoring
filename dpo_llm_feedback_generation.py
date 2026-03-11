import os
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback, Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback, AutoModelWithLMHead, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import pickle
import numpy as np
import json
import argparse
import random
import torch as th
from tqdm import tqdm
import re
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def set_seed(args):
    """
    Ensure reproducibility by setting the seed for random number generation.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    if th.cuda.is_available():
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)  # if use multi-GPU
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def prepare_dataset(examples, tokenizer, args, test=False):

    system_prompt = """
    You are an English teacher tasked with providing feedback to students. Your goal is to provide feedback that guides the student from an incorrect answer to the correct one. The feedback must be limited to one sentence.
    """
    
    if not test:
        chosen_msgs = []
        rejected_msgs = []
        for input_text, chosen, rejected, reference_text in zip(examples["input"], examples["chosen"], examples["rejected"], examples["reference"]):
            chosen_msgs.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{input_text}\nFeedback:"},
                {"role": "assistant", "content": chosen}
            ]   
            )
            rejected_msgs.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{input_text}\nFeedback:"},
                {"role": "assistant", "content": rejected}
            ]
            )
        dataset = {
            "chosen": chosen_msgs,     # messages 리스트
            "rejected": rejected_msgs, # messages 리스트
        }
    
    else:
        tokenizing_prompts = []
        for input_text, chosen, rejected, reference_text in zip(examples["input"], examples["chosen"], examples["rejected"], examples["reference"]):
            tokenizing_prompts.append(
                    tokenizer.apply_chat_template([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{input_text}\nFeedback:"},
                    ], tokenize=False, add_generation_prompt=True)
                )
        tokenized = tokenizer(tokenizing_prompts, padding="max_length", truncation=True, max_length=700)
        chosen_responses = examples[f"chosen"]
        reference_responses = examples[f"reference"]
        chosen_list = []
        for chosen in chosen_responses:
            chosen_list.append(
                chosen
            )
        reference_list = []
        for reference in reference_responses:
            reference_list.append(
                reference
            )
            
        dataset = {
                "chosen": chosen_list,
                "reference": reference_list,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }

    return dataset

def llama_train(args, model, lora, train_dataset, dev_dataset, tokenizer): 
    eval_steps = args.eval_steps
    if args.debug:
        eval_steps = 2
    print("Evaluation steps:", eval_steps)
    """
    Train the model using the provided training dataset and evaluation dataset.
    """
    
    training_args = DPOConfig(
        output_dir=args.checkpoint_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        # max_steps=1000,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=eval_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # fp16=True,
        bf16=True,
        max_grad_norm=1.0,
        weight_decay=0.05,
        beta=0.5,
        deepspeed="deepspeed_config.json"
    )


    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()

    best_path = trainer.state.best_model_checkpoint
    with open(f"{args.checkpoint_dir}/best_checkpoint_path.txt", "w") as f:
        f.write(best_path)
    
    return model

def llama_test(model, tokenizer, test_dataset, args):
    
    pred_list = []
    true_label_list = []
    true_reference_list = []
    model.eval()
    batch_size = args.test_batch_size
    
    with th.no_grad():
        for i in tqdm(range(0, len(test_dataset), args.test_batch_size)):
            test = test_dataset[i:i+batch_size]

            outputs = model.generate(
                input_ids=th.tensor(test["input_ids"]).to(model.device),
                attention_mask=th.tensor(test["attention_mask"]).to(model.device),
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1)

            generated_texts = tokenizer.batch_decode(outputs[:, th.tensor(test["input_ids"]).shape[1]:], skip_special_tokens=True)
            for i, (pred, true_label, true_reference) in enumerate(zip(generated_texts, test["chosen"], test["reference"])):
                pred = pred.replace("assistant", "").strip()
                pred_list.append(pred)
                true_label_list.append(true_label)
                true_reference_list.append(true_reference)

                with open(f"{args.model_saving_dir}/tmp_feedback_predictions.json", "w") as f:
                    json.dump(pred_list, f, indent=4)
                with open(f"{args.model_saving_dir}/tmp_feedback_true_labels.json", "w") as f:
                    json.dump(true_label_list, f, indent=4)
                with open(f"{args.model_saving_dir}/tmp_feedback_true_references.json", "w") as f:
                    json.dump(true_reference_list, f, indent=4)
                

    return pred_list, true_label_list, true_reference_list

def main(args):
    set_seed(args)
    
    args.device = "cuda" if th.cuda.is_available() else "cpu"
    
    if not os.path.exists("results"):
        os.makedirs("results")
    model_saving_dir = os.path.join("results", args.model_name.split("/")[-1])
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
    
    model_saving_dir = os.path.join(model_saving_dir, "dpo") 
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)

    model_saving_dir = os.path.join(model_saving_dir, f"{args.mode}")   

    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
    
    model_saving_dir = os.path.join(model_saving_dir, f"seed_{args.seed}")
    if not os.path.exists(model_saving_dir):
        os.makedirs(model_saving_dir)
    args.model_saving_dir = model_saving_dir
    
    
    
    args.checkpoint_dir = f"./{args.model_name.split('/')[-1]}"
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, "dpo")
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"{args.mode}")
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"seed_{args.seed}")
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        


    if not args.test:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                cache_dir=".", 
                                                torch_dtype=th.bfloat16, 
                                                pad_token_id=tokenizer.pad_token_id,
                                                )
        lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # for LLaMA
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", use_fast=False)
        
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                cache_dir=".", 
                                                torch_dtype=th.bfloat16, 
                                                pad_token_id=tokenizer.pad_token_id,
                                                )
        
        with open(f"{args.checkpoint_dir}/best_checkpoint_path.txt", "r") as f:
            best_model_checkpoint = f.read().strip()
        # best_model_checkpoint = th.load(f"{args.checkpoint_dir}/best_checkpoint_path.txt" )
        print("Best model checkpoint:", best_model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(best_model_checkpoint, padding_side="left", use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = PeftModel.from_pretrained(model, best_model_checkpoint, is_trainable=False)
        model.print_trainable_parameters()
    

    train_data = json.load(open(f"./data/criteria_dpo_train_feedback_data.json"))
    dev_data = json.load(open(f"./data/criteria_dpo_dev_feedback_data.json"))
    test_data = json.load(open(f"./data/criteria_dpo_test_feedback_data.json"))
    train_df = pd.DataFrame(train_data)
    dev_df = pd.DataFrame(dev_data)
    test_df = pd.DataFrame(test_data)
    

    if args.debug:
        train_df = train_df.sample(10).reset_index(drop=True)
        dev_df = dev_df.sample(10).reset_index(drop=True)
        test_df = test_df[:10]

        args.epochs = 1
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    test_dataset = Dataset.from_pandas(test_df)

    
    train_dataset = train_dataset.map(lambda x: prepare_dataset(x, tokenizer, args), batched=True)
    dev_dataset = dev_dataset.map(lambda x: prepare_dataset(x, tokenizer, args), batched=True)
    test_dataset = test_dataset.map(lambda x: prepare_dataset(x, tokenizer, args, test=True), batched=True)

    

    if not args.test:
        model = llama_train(args, model, lora_config, train_dataset, dev_dataset, tokenizer)

        
        pred_list, true_label_list, true_reference_list = llama_test(model, tokenizer, test_dataset, args)
    
    else:
        
        pred_list, true_label_list, true_reference_list = llama_test(model, tokenizer, test_dataset, args)

    
    with open(f"{args.model_saving_dir}/feedback_predictions.pkl", "wb") as f:
        pickle.dump(pred_list, f)
    with open(f"{args.model_saving_dir}/feedback_true_labels.pkl", "wb") as f:
        pickle.dump(true_label_list, f)
    with open(f"{args.model_saving_dir}/feedback_true_references.pkl", "wb") as f:
        pickle.dump(true_reference_list, f)


            
    return pred_list, true_label_list, true_reference_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA model for essay scoring")
    # parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Model name or path")
    # parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name or path")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Batch size for training (8B)")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation (8B)")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Batch size for testing (8B)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=20, help="Evaluation steps during training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with reduced dataset and epochs")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--test", action="store_true", help="Run in test mode without training")
    parser.add_argument("--mode", type=str, default="criteria")
    args = parser.parse_args()
    
    
    print(f"Running for model {args.model_name} with seed {args.seed}")
    pred_list, true_label_list, true_reference_list = main(args)

