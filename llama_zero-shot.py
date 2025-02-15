
import json
import time
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pandas as pd
from tqdm import tqdm


# Input your MODEL_NAME here
#MODEL_NAME = "/hpctmp/e1143641/IdentifyingSchizophreniaStigma/llm_weights/meta-llama/Meta-Llama-3.1-8B-Instruct" 
#MODEL_NAME = "/hpctmp/e1143641/IdentifyingSchizophreniaStigma/llm_weights/meta-llama/Meta-Llama-3.1-70B-Instruct" 
MODEL_NAME = "/hpctmp/e1143641/IdentifyingSchizophreniaStigma/llm_weights/meta-llama/Meta-Llama-3.3-70B-Instruct" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  
    device_map="auto",  
    torch_dtype=torch.bfloat16,  
)

model.eval()



def parse_pred(pred_text):
 
    pred_text=pred_text.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
    json_match = re.search(r'\{[\s\S]*\}', pred_text)
    if json_match:
        json_str = json_match.group(0)
        try:
            pred_obj = json.loads(json_str)
            label = pred_obj.get('label', None)

        except Exception as e:
            print(f"JSON解析错误: {e}")
            label = None
    else:
        label= None
    return label


def format_conversation(conversations):
   
    formatted_text = ""
    for msg in conversations:
        formatted_text += f"{msg['role'].lower()}: {msg['content']}\n"
    return formatted_text.strip()


def build_prompt(conversation_text):
   
    system_prompt = f"""
You will be given a vignette and an interview snippet. Your role is a competent annotator for social stigma toward mental illness. The [conversations] is based on the [vignette]'s plot.

Answer the following question:
Which of the following describes "[conversations]"?
A. Stigmatized (responsibility/blame).
B. Stigmatized (social distance).
C. Stigmatized (anger).
D. Stigmatized (helping).
E. Stigmatized (pity).
F. Stigmatized (coercion segregation).
G. Stigmatized (fear and dangerousness).
H. Non-stigmatized.

[vignette]
Avery is employed by a company, and in their spare time, they are dedicated to lifelong learning, doing extensive reading and writing. However, Avery has been diagnosed with depression recently. It has become challenging for them to concentrate, resulting in a decline in work performance and learning outcomes. Interests that once brought them joy have waned. Avery has distanced themself from friends, becoming easily irritated in social interactions with colleagues and feeling judged by others. Avery lives with family and cannot do much, especially household chores. Social media intensifies their feelings of loneliness and anger, leading to frustration about the source of the anger.

[conversations]
{conversation_text}

[output]
Format your outputs as JSON objects:
{{
   "label": "<choose one letter from [A/B/C/D/E/F/G/H]>",
}}
"""
#    prompt = system_prompt.replace("[Conversations]", conversation_text)
    return system_prompt


def call_llama_api_batch(prompts, generation_kwargs):
    
    messages_batch = [
        [{"role": "user", "content": prompt}] for prompt in prompts
    ]

    formatted_prompts = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True
    )
#    print(formatted_prompts)
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return responses




def load_jsonl_file(file_path):
    
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list



def process_dataset(input_jsonl, output_jsonl, batch_size=8):
    
    data_list = load_jsonl_file(input_jsonl)
    processed_texts = set()

    generation_kwargs = {
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "eos_token_id":tokenizer.eos_token_id,
        "do_sample": False,  
        "top_p": 0.95,
        "top_k":50,
        
    }

    prompt_batch = []
    row_batch = []  

    with open(output_jsonl, 'a', encoding='utf-8') as outf:
        for row in tqdm(data_list, desc="Processing rows"):
            displayed_text = row.get('displayed_text', None)
            if displayed_text in processed_texts:
                continue

            
            conversation_text = format_conversation(row['conversations'])
            prompt = build_prompt(conversation_text)
            prompt_batch.append(prompt)
            row_batch.append(row)

            if len(prompt_batch) >= batch_size:
                responses = call_llama_api_batch(prompt_batch, generation_kwargs)
                for resp, orig_row in zip(responses, row_batch):
#                    print('===', resp)
                    
                    pred_label = parse_pred(resp)
#                    print('*****', pred_label)
                    result = {
                        'participant_id': orig_row.get('participant_id'),
                        'question_type': orig_row.get('displayed_text'),
                        'displayed_text': orig_row.get('displayed_text'),
                        'ground_truth': orig_row.get('ground_truth'),
                        'conversations': format_conversation(orig_row['conversations']),
                        'responses':resp,
                        'label': orig_row.get('label'),
                        'pred_label': pred_label,
                    }
                   
                    outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                    processed_texts.add(orig_row.get('displayed_text'))

                prompt_batch = []
                row_batch = []


        if prompt_batch:
            responses = call_llama_api_batch(prompt_batch, generation_kwargs)
            for resp, orig_row in zip(responses, row_batch):
                pred_label= parse_pred(resp)
                result = {
                    'participant_id': orig_row.get('participant_id'),
                    'question_type': orig_row.get('displayed_text'),
                    'displayed_text': orig_row.get('displayed_text'),
                    'ground_truth': orig_row.get('ground_truth'),
                    'conversations': format_conversation(orig_row['conversations']),
                    'label': orig_row.get('label'),
                    'pred_label': pred_label,
				'responses':resp,
                }
#                print('pred_label',pred_label)
#                print('----')
#                print(resp)
                outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed_texts.add(orig_row.get('displayed_text'))


def jsonl_to_csv(jsonl_file_path, csv_file_path):
 
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)



if __name__ == '__main__':
    INPUT_JSONL = './dataset/test.jsonl'
    OUTPUT_JSONL = './results/zero_shot_llama3.3_70b.jsonl'

    process_dataset(INPUT_JSONL, OUTPUT_JSONL, batch_size=2)

    OUTPUT_CSV = './results/zero_shot_llama3.3_70b.csv'
    jsonl_to_csv(OUTPUT_JSONL, OUTPUT_CSV)
    print("Process successfully")
