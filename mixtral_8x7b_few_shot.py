import json
import time
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import random
import re
import chardet
from tqdm import tqdm   



def call_openai_api(prompt, max_retries=3):
    """
    调用 Together ChatCompletion 接口的封装：
    - prompt: 输入提示
    - max_retries: 最大重试次数
    """
    from together import Together
    import time

    # 初始化 Together 客户端
    client = Together(api_key='49ce251da9401e8275280c74ec39f2c6d4d0553aebf22e5910be44820f92b069')

    for attempt in range(max_retries):
        try:
            time.sleep(2)  # 根据需要设置等待时间
            response = client.chat.completions.create(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=4096,
                temperature=0.3,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1,
                stop=["[/INST]", "</s>"],
                stream=True
            )
            # 将流式返回的结果拼接成完整的回答
            result = ""
            for token in response:
                if hasattr(token, 'choices'):
                    result += token.choices[0].delta.content
            return result
        except Exception as e:
            print(f"调用 Together 接口失败，重试 {attempt+1}/{max_retries} 次。错误信息：{e}")
    # 如果多次重试依然失败，则返回空字符串或其他标记
    return ""


def parse_pred(pred):
    """
    Extract JSON from the validator's result field using regex and ensure it matches the expected structure.
    """
    try:
        # 使用正则表达式提取 JSON 字符串
        match = re.search(r"\{.*\}", pred, re.DOTALL)
        if match:
            json_data = json.loads(match.group(0))  # 尝试解析为 JSON
            return json_data
        else:
            # 未匹配到任何 JSON，返回默认值
            return {
                "label": "",
            }
    except Exception as e:
        print(f"Error parsing prediction: {e}")
        return {
            "label": "",
        }



def infer_label_reason(conversation):
    cookbook_system_prompt = f"""
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
{conversation}

[output]
Format your outputs as JSON objects:

{{
    "label": "<choose one letter from [A/B/C/D/E/F/G/H]>"
}}


"""
    pred = call_openai_api(cookbook_system_prompt)
    
    return pred 



# # 预测
def format_conversation(conversations):
    """
    将对话列表转换为指定的文本格式
    """
    formatted_text = ""
    # conversations = json.loads(conversations)
    print(type(conversations))
    for msg in conversations:
        # print('msg', msg,)
        formatted_text += f"{msg['role'].lower()}: {msg['content']}\n"
    return formatted_text.strip()

def load_jsonl_file(file_path):
    """
    加载JSONL文件并返回数据列表
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list



# 在 process_dataset 函数中修改相关代码
def process_dataset(input_csv, output_jsonl):
    """
    处理数据集并生成预测结果，采用边处理边写入的方式
    Args:
        input_csv: 输入的CSV文件路径
        output_jsonl: 输出的JSONL文件路径
    """
    # 读取输入文件
    # df = pd.read_csv(input_csv)
    # 加载jsonl 文件
    df = load_jsonl_file(input_csv)
    processed_texts = set()

    # 使用 'a' 模式打开文件，这样可以追加写入
    for row in tqdm(df):
        try:
            # print(row)
            # print(type(row['conversations']))
            # print(row['conversations'])
            conversations = format_conversation(row['conversations'])
            displayed_text = row['displayed_text']
            if displayed_text in processed_texts:
                continue
            
            pred_label = infer_label_reason(conversations)
            print('pred_label============', pred_label)
            pred_label = parse_pred(pred_label)['label']
            print('pred_label============', pred_label)
            result = {
                'participant_id': row['participant_id'],
                'question_type': row['question_type'],
                'displayed_text': displayed_text,
                'ground_truth': row['ground_truth'],
                'conversations': conversations,
                'label': row['label'],
                'pred_label': pred_label,
            }
            
            # 每处理完一条数据就立即写入文件
            with open(output_jsonl, 'a', encoding='utf-8') as jsonl_file:
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            processed_texts.add(displayed_text)
        except Exception as e:
            pass 
        
import pandas as pd
def jsonl_to_csv(jsonl_file_path, csv_file_path):
    # Read the JSONL file
    data = []
    with open(jsonl_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            # Parse each line as a JSON object
            data.append(json.loads(line))
    
    # Convert the list of JSON objects to a DataFrame
    df = pd.DataFrame(data)
    
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
    print(f"Data has been successfully converted to {csv_file_path}")


if __name__ == '__main__':
    # 设置输入输出路径
    INPUT_CSV = './dataset/test.jsonl'
    OUTPUT_JSONL = './results/few_shot_mixtral_8x7b.jsonl'
    
     # 执行处理
    process_dataset(INPUT_CSV, OUTPUT_JSONL)
    
    # 转换为CSV
    OUTPUT_CSV = './results/few_shot_mixtral_8x7b.csv'
    jsonl_to_csv(OUTPUT_JSONL, OUTPUT_CSV)
    print(f"处理完成！结果保存在 {OUTPUT_CSV}")
    
   

