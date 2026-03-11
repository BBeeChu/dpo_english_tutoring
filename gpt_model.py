import re

from openai import OpenAI


key = 'API_KEY'

client = OpenAI(api_key = key)

class GPTModel():
    
    def ask_chatgpt(self, prompt, seed, model="gpt-5.1", temperature=0.0, n=1):
        response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    n=n,
                    seed=seed,
                    temperature=temperature
                )

        return [choice.message.content for choice in response.choices]
    
    def post_process(self, answer):
        answer = answer.replace('\n', ' ').replace('sql','').replace('```','')
        answer = re.sub('[ ]+', ' ', answer)
        answer = answer.strip()
        return answer