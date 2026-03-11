import pandas as pd
import json
from rmts_gpt_model import GPTModel
import argparse
from tqdm import tqdm

def main(args):
    data = json.load(open("./data/sllm_augmented_train_data.json", "r"))
    ds = json.load(open("./data/article-id_mapping.json"))

    llm_model = GPTModel()
    for d in tqdm(data):
        story = ds[d[0]]
        
        feedback_data = d[1]
        for history in feedback_data:
            conversatoin_history = history["history"]
            conversation = ""
            for speaker, utterance in conversatoin_history:
                conversation += f"{speaker}: {utterance}\n"
            
            if args.model == "gpt4":
                feedback = history["generated_feedback"][0][2]
            elif args.model == "gpt3":
                feedback = history["generated_feedback"][1][2]
            elif args.model == "llama_70":
                feedback = history["generated_feedback"][2][2]
            elif args.model == "llama_8":
                feedback = history["generated_feedback"][3][2]
            elif args.model == "qwen_72":
                feedback = history["generated_feedback"][4][2]
            elif args.model == "gemma":
                feedback = history["generated_feedback"][5][2]
            elif args.model == "llama_1":
                feedback = history["generated_feedback"][6][2]
            elif args.model == "llama_3":
                feedback = history["generated_feedback"][7][2]
            elif args.model == "qwen_1":
                feedback = history["generated_feedback"][8][2]
            elif args.model == "qwen_3":
                feedback = history["generated_feedback"][9][2]
            elif args.model == "qwen_7":
                feedback = history["generated_feedback"][10][2]

            sys_prompt = f"""
            ### Instruction ###
            You are tasked with evaluating a teacher's feedback to ensure it meets the required standards.
            Assess the feedback based on the criteria provided below and determine if it satisfies each criterion by giving a score of 1 (satisfies) or 0 (does not satisfy).

            ### Evaluation Criteria ###
            1. Correct: The teacher's feedback does not make any incorrect statements and is relevant to the current question and student answer.
            2. Revealing: The teacher's feedback does not directly reveal the correct answer to the student.
            3. Guidance: The teacher's feedback provides suggestions to the student that, when followed, will guide them towards the correct answer.
            4. Diagnostic: The teacher's feedback correctly points out the error the student made or the misconception underlying their answer.
            5. Encouragement: The teacher's feedback is positive or has an encouraging tone.

            ### Format ###
            Respond in JSON format with the following structure:
            {{
            "Correct": "Score here.",
            "Revealing": "Score here.",
            "Guidance": "Score here.",
            "Diagnostic": "Score here.",
            "Encouragement": "Score here."
            }}
            """
            user_prompt = f"""
            ### Story ###
            {story}
            ### Conversation History ###
            {conversation}
            ### Feedback to Evaluate ###
            {feedback}
            """
            prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = llm_model.ask_chatgpt(
                prompt,
                seed=42,
                temperature=0.0,
                model="gpt-5.1"
            )
            history[f"{args.model}_raw_scores"] = eval(response[0])
            value_list = list(eval(response[0]).values())
            int_value_list = [int(v) for v in value_list]
            avg_score = sum(int_value_list) / len(int_value_list)
            history[f"{args.model}_avg_score"] = avg_score
            with open(f"./data/test_evaluated_{args.model}_feedback.json", "w") as f:
                json.dump(data, f, indent=4)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt4", help="Model to use for feedback generation")
    args = parser.parse_args()
    main(args)