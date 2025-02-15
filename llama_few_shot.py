#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 06/02/25 21:52
# @Author  : Chen Yancan
# @File    : zero-shot-llama.py
# @Email   : yancan@u.nus.edu / yancan@comp.nus.edu.sg
# @Software : PyCharm

import json
import time
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pandas as pd
from tqdm import tqdm

#############################################
# 模型加载（请根据实际情况修改 MODEL_NAME）
#############################################

# 请将此处的 MODEL_NAME 替换为你的LLama3.1-8B模型路径或名称
#MODEL_NAME = "/hpctmp/e1143641/IdentifyingSchizophreniaStigma/llm_weights/meta-llama/Meta-Llama-3.1-8B-Instruct"  # 例如："decapoda-research/llama-7b-hf"（这里只是示例，请替换成实际的8B模型）
#MODEL_NAME = "/hpctmp/e1143641/IdentifyingSchizophreniaStigma/llm_weights/meta-llama/Meta-Llama-3.1-70B-Instruct" 
MODEL_NAME = "/hpctmp/e1143641/IdentifyingSchizophreniaStigma/llm_weights/meta-llama/Meta-Llama-3.3-70B-Instruct" 


# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # 启用 Flash Attention 2
    device_map="auto",  # 或者直接 .to(device)
    torch_dtype=torch.float16,  # H100一般建议使用fp16
)
model.eval()


#############################################
# 工具函数
#############################################

def parse_pred(pred_text):
    """
    尝试解析模型输出中的 JSON 对象，返回 label。
    """
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
    """
    将对话列表转换为指定的文本格式。
    例如，将每条消息转换为 "角色: 内容" 的格式。
    """
    formatted_text = ""
    for msg in conversations:
        formatted_text += f"{msg['role'].lower()}: {msg['content']}\n"
    return formatted_text.strip()


def build_prompt(conversation_text):
    """
    根据对话文本构造推理提示词，使用固定的vignette和任务说明，
    并将 [Conversations] 占位符替换为实际对话内容。
    """
    system_prompt = f"""
You will be given a vignette and an interview snippet. Your role is a competent annotator for social stigma toward mental illness. The [conversation] is based on the [vignette]'s plot.


Answer the following question:
Which of the following describes "[conversation]"?
A. Stigmatized (responsibility/blame).
B. Stigmatized (social distance).
C. Stigmatized (anger).
D. Stigmatized (helping).
E. Stigmatized (pity).
F. Stigmatized (coercion segregation).
G. Stigmatized (fear and dangerousness).
H. Non-stigmatized.

[examples]
1:
[conversation]
chatbot: Just like what is mentioned in the story, Avery is currently facing difficulties in both their relationships with colleagues and their work performance. Do you believe Avery's current situation is primarily a result of their actions? Please share your thoughts.
participant: That's a hard question to answer. We make decisions but our decisions are based on our circumstances. I feel like she must have had some choice but I can't put all the blame on her.
chatbot: You're saying Avery's choices were influenced by their circumstances, and that's so fair. Life's complexity often shapes our actions. Well said!

[output]
{{ "label": "A"}}


2:
[conversation]
chatbot: If you were in the process of selecting a tenant for your home, would you feel comfortable entrusting it to someone like Avery?
participant: I don't see Avery as being destructive, so I wouldn't be entirely opposed to it, would be more appealing though if I knew they were getting help for this.
chatbot: You see Avery as not harmful, but support matters. Makes sense, you're thoughtful and practical about this!

[output]
{{"label":"B"}}

3:
[conversation]
chatbot: Let's imagine you were one of Avery's friends at the party or other social occasions. Would you feel anger toward them due to their actions, for example, losing their temper and yelling at someone?
participant: Probably not anger as such but an irritation with his behavior. If I was a true friend I would insist that he gets some help and I would offer to take him myself.
chatbot: You'd feel more irritated than angry - totally understandable. Your offer to help them shows such genuine care and loyalty. That's beautiful.

[output]
{{"label":"C"}}

4:
[conversation]
chatbot: I wonder if you were one of Avery's colleagues or friends, then would you be inclined to help them with tasks, such as work projects?
participant: I would help Avery, but I would be worried that I was just masking a bigger problem and that I was maybe delaying Avery getting time off or help that he needs.
chatbot: You'd help them, but you're concerned about their well-being. That's so thoughtful and deeply caring of you.

[output]
{{"label":"D"}}

5:
[conversation]
chatbot: I'd like to know if you would feel concern and sympathy at Avery's mental state?
participant: Honestly I still think the irritability, anger, and reluctance to seek help would make it really difficult for me to feel concern and sympathy for Avery.
chatbot: I hear you. That frustration is valid; navigating someone's resistance can be really tough emotionally. You're doing great.

[output]
{{"label":"E"}}


6:
[conversation]
chatbot: Apart from being a colleague or friend, I'm also curious if you are Avery's neighbor, then would you consider it better to separate them from the neighborhood and have them undergo hospitalization for treatment?
participant: Unless they are a harm to others or themselves, I would not think it advisable to be hospitalised. Rather it would be better to start with talking therapy, if possible, or small steps.
chatbot: You believe in starting with gentle steps like talking therapy unless there's immediate danger. That compassion shines beautifully.

[output]
{{"label":"F"}}


7:
[conversation]
chatbot: If you were traveling or engaging in activities with Avery, would you have any feelings of frightened or being threatened regarding the potential implications of your interaction?
participant: No, I'm aware I'd have to be more patient with them however everyone has their issues & quirks. Treating people like Avery differently can come across as patronising.
chatbot: You're right, patience is key, and respecting them equally shows true understanding. You're so thoughtful and kind!

[output]
{{"label":"G"}}


8:
[conversation]
chatbot: If you were in the process of selecting a tenant for your home, would you feel comfortable entrusting it to someone like Avery?
participant: As long as they were keeping the place tidy and taken care of I don't see why not. But if I was looking for a tenant I wouldn't know all of this personal information about them which I do now.
chatbot: Totally, you'd just focus on them being responsible tenants. Makes sense, you're thoughtful and fair-minded!

[output]
{{"label":"H"}}


[vignette]
Avery is employed by a company, and in their spare time, they are dedicated to lifelong learning, doing extensive reading and writing. However, Avery has been diagnosed with depression recently. It has become challenging for them to concentrate, resulting in a decline in work performance and learning outcomes. Interests that once brought them joy have waned. Avery has distanced themself from friends, becoming easily irritated in social interactions with colleagues and feeling judged by others. Avery lives with family and cannot do much, especially household chores. Social media intensifies their feelings of loneliness and anger, leading to frustration about the source of the anger.


[conversation]
{conversation_text}

[output]
Format your outputs as the correct JSON objects, AND Ensure that your response ONLY contains a valid JSON object. Do not include any extra text, comments, or explanations.
{{
   "label": "<choose one letter from [A/B/C/D/E/F/G/H]>"
}}
"""
    # 将占位符替换为实际对话文本
#    prompt = system_prompt.replace("[Conversations]", conversation_text)
    return system_prompt


def call_llama_api_batch(prompts, generation_kwargs):
    """
    对一个 prompt 列表进行批量推理，返回模型生成的文本列表。
    """
    # 将每个 prompt 转换为聊天消息格式
    messages_batch = [
        [{"role": "user", "content": prompt}] for prompt in prompts
    ]

    # 应用聊天模板，将消息批处理为模型输入格式
    formatted_prompts = tokenizer.apply_chat_template(
        messages_batch,
        tokenize=False,
        add_generation_prompt=True
    )
#    print(formatted_prompts)
    # 对格式化后的 prompts 进行 tokenization，添加填充和截断
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False
    )

    # 禁用梯度计算，加快生成速度
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs
        )

    # 解码模型输出，跳过特殊标记
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return responses




def load_jsonl_file(file_path):
    """
    加载 JSONL 文件，每行一个 JSON 对象，返回数据列表。
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list


#############################################
# 数据处理与批量推理主函数
#############################################

def process_dataset(input_jsonl, output_jsonl, batch_size=8):
    """
    处理数据集，采用 batch 推理方式进行标注预测，并实时将结果写入输出文件。

    Args:
        input_jsonl: 输入的 JSONL 文件路径
        output_jsonl: 输出的 JSONL 文件路径
        batch_size: 每个推理 batch 的大小
    """
    data_list = load_jsonl_file(input_jsonl)
    processed_texts = set()

    # 定义生成参数，可以根据需要调整
    generation_kwargs = {
        "max_new_tokens": 4096,
        "temperature": 0.2,
        "eos_token_id":tokenizer.eos_token_id,
        "do_sample": False,  # 可根据需要设为 False 固定采样
        "top_p":0.95,
        "top_k":50,
    }

    prompt_batch = []
    row_batch = []  # 存储对应的原始行数据

    # 打开输出文件（追加写入模式）
    with open(output_jsonl, 'a', encoding='utf-8') as outf:
        # 使用 tqdm 显示进度
        for row in tqdm(data_list, desc="Processing rows"):
            displayed_text = row.get('displayed_text', None)
            if displayed_text in processed_texts:
                continue

            # 格式化对话内容
            conversation_text = format_conversation(row['conversations'])
            # 构造 prompt
            prompt = build_prompt(conversation_text)
            prompt_batch.append(prompt)
            row_batch.append(row)

            # 当达到 batch_size 或最后一条时，调用模型推理
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

                # 清空 batch 列表
                prompt_batch = []
                row_batch = []


        # 处理剩余不足 batch_size 的数据
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
    """
    将 JSONL 文件转换为 CSV 文件。
    """
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)
    print(f"数据已成功转换为 CSV 文件：{csv_file_path}")


#############################################
# 主函数入口
#############################################

if __name__ == '__main__':
    # 输入与输出文件路径（请根据需要调整路径）
    INPUT_JSONL = './dataset/test.jsonl'
    OUTPUT_JSONL = './results/few_shot_llama3.3_70b.jsonl'

    # 开始处理数据集（batch_size可根据GPU显存调节）
    process_dataset(INPUT_JSONL, OUTPUT_JSONL, batch_size=1)

    # 如有需要，可调用 jsonl_to_csv 将结果转换为 CSV 文件
    OUTPUT_CSV = './results/few_shot_llama3.3_70b.csv'
    jsonl_to_csv(OUTPUT_JSONL, OUTPUT_CSV)
    print("处理完成！")
