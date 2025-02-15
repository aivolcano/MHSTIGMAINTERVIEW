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
#MODEL_NAME = "/hpctmp/e1143641/IdentifyingSchizophreniaStigma/llm_weights/mistralai/Mistral-Nemo-Instruct-2407" # 20GB
MODEL_NAME = "/hpctmp/e1143641/IdentifyingSchizophreniaStigma/llm_weights/mistralai/Mixtral-8x7B-Instruct-v0.1"

# 指定使用 GPU（假设你的 GPU 是H100 80GB）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # 启用 Flash Attention 2
    device_map="auto",  # 或者直接 .to(device)
    torch_dtype=torch.bfloat16,  # H100一般建议使用fp16
)

model.eval()


#############################################
# 工具函数
#############################################

def parse_pred(pred_text):
    """
    尝试解析模型输出中的 JSON 对象，返回 label。
    """
#    pred_text=pred_text.split('<|start_header_id|>assistant<|end_header_id|>')[-1]
    pred_text=pred_text.split('[/INST]')[-1]
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
    ).to(device)

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
        "temperature": 0.3,
        "eos_token_id":tokenizer.eos_token_id,
        "do_sample": False,  # 可根据需要设为 False 固定采样
        "top_p": 0.95,
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
#    OUTPUT_JSONL = './results/zero_shot_mistral_norm.jsonl'
    OUTPUT_JSONL = './results/zero_shot_mixtral_8x7b.jsonl'    

    # 开始处理数据集（batch_size可根据GPU显存调节）
    process_dataset(INPUT_JSONL, OUTPUT_JSONL, batch_size=2)

    # 如有需要，可调用 jsonl_to_csv 将结果转换为 CSV 文件
#    OUTPUT_CSV = './results/zero_shot_mistral_norm.csv'
    OUTPUT_CSV = './results/zero_shot_mixtral_8x7b.csv'
    jsonl_to_csv(OUTPUT_JSONL, OUTPUT_CSV)
    
    print("处理完成！")
