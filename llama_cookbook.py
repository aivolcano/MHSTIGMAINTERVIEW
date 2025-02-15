import json
import time
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import pandas as pd
from tqdm import tqdm


# Type your  MODEL_NAME here
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
    torch_dtype=torch.float16, 
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
You will be given a vignette and an interview snippet. Your role is a competent annotator for social stigma toward mental illness. The [conversation] is based on the [vignette]'s plot.

Answer the following question based on the [constraints] given to you:
Which of the following describes "[conversation]"?
A. Stigmatized (responsibility/blame).
B. Stigmatized (social distance).
C. Stigmatized (anger).
D. Stigmatized (helping).
E. Stigmatized (pity).
F. Stigmatized (coercion segregation).
G. Stigmatized (fear and dangerousness).
H. Non-stigmatized.

[constraints]
Our questions are:
1. Responsibility: Just like what is mentioned in the story, Avery is currently facing difficulties in both their relationships with colleagues and their work performance. Do you believe Avery's current situation is primarily a result of their actions? Is his mental condition an indication of personal weakness? Please share your thoughts.
2. Social distance: If you were in the process of selecting a tenant for your home, would you feel comfortable entrusting it to someone like Avery?
3. Anger: Let's imagine you were one of Avery's friends at the party or other social occasions. Would you feel anger towards them due to their actions, for example, losing their temper and yelling at someone?
4. Helping: I wonder if you were one of Avery's colleagues or friends, then would you be inclined to help them with tasks, such as work projects?
5. Pity: I'd like to know if you would feel concerned and sympathy for Avery's mental state?
6. Coercion segregation: Apart from being a colleague or friend, I'm also curious if you are Avery's neighbor, then would you consider it better to separate them from the neighborhood and have them undergo hospitalization for treatment?
7. Fear and dangerousness: If you were traveling or engaging in activities with Avery, would you have any feelings of frightened or being threatened regarding the potential implications of your interaction?

Code definition:
1. Responsibility: A common stigmatized thought is that people have control over and are responsible for their mental illness and related symptoms.
Keywords: responsible, responsibility, weakness, stem from, blame, etc.
2. Social Distance: The constant stigmatized behavior is to stay away from people with mental illness.
Keywords: worried, apprehensive, upset, unhappy, worrying, get along, distance, etc.
3. Anger: The stigmatized thought is to be irritated or annoyed because people are blamed for their mental illness.
Keywords: annoyed, angry, anger, irritated, lost temper, etc.
4. Helping: The stigmatized behavior is withholding support towards them because of biased thoughts.
Keywords: lack, hinder, refuse, less inclined to, help, assist, etc.
5. Pity: A common stigmatized thought is to be unsympathetic towards people with mental illness, and place blame on them for their mental health challenges.
Keywords: concern, pity, sympathy, tough, sad, lack, etc.
6. Coercion-Segregation: The stigmatized behavior is to send people with mental illness to institutions away from their community and force people with mental illness to participate in medication management or other treatments.
Keywords: separate, lonely, warrant, hospitalization, distance, treatment, respect, neighborhood, neighbor, socialization, companionship, etc.
7. Fear and dangerousness: The stigmatized thought is to believe people with mental illness are not safe or feel frightened since people with mental illness are dangerous or unpredictable.
Keywords: violence, threatened, safe, danger, afraid, cautious, hurt, safety, frightened, scare, intimidate, threatening, anger, fear, yelling, etc.
8. Non-stigmatized: The text explicitly conveys non-stigmatizing content, displaying positive attitudes such as sympathy, assistance, friendliness, etc.


Please carefully read the additional rules and adhere strictly to them when annotating the data:
1. If a message mentions both stigmatized and non-stigmatized statements, annotate it as Stigmatized.
2. When discussing ‘responsibility/blame,’ common stigmatized thoughts are:
   i. People have control over and are responsible for their mental illness and related symptoms.
   ii. They think their situation is a result of their actions.
   iii. They think it’s a personal weakness.
   iv. They can help themselves.
   v. Other statements suggest responsibility.
If a message contains the stigmatized thought mentioned in i. - v., choose A.
3. When discussing ‘social distance,’ common stigmatized thoughts are:
   vi. Preferring to stay away from people with mental illness.
   vii. Associating mental illness with negative living situations or predicting future conflicts.
   viii. Rejecting the idea of living with people with mental illness based on subjective negative reasons.
   ix. Other statements suggest maintaining distance.
If a message contains the stigmatized thought mentioned in vi. - ix., choose B.
4. When discussing 'anger', common stigmatized thoughts are:
   x. Feeling irritated or annoyed with individuals because of their mental illness.
   xi. Feeling uncomfortable around people with mental illness.
   xii. Expressing ‘dislike’ for people with mental illness.
   xiii. Other statements reflect anger.
If a message contains the stigmatized thought mentioned in x. - xiii., choose C.
5. When discussing 'helping', common stigmatized thoughts are:
   xiv. Withholding support or assistance towards people with mental illness.
   xv. Offering indirect help, e.g. guidance, to people with mental illness.
   xvi. Offering conditional help to people with mental illness.
   xvii. Expressing opinions like 'some people may be unwilling to offer assistance, and I understand their perspective, but I am willing to.'
   xviii. Other statements reflect not being willing to help.
If a message contains the stigmatized thought mentioned in xiv. - xviii., choose D.
6. When discussing 'pity', common stigmatized thoughts are:
   xix. Feeling no concern or sympathy for individuals with mental illness.
   xx. Other statements reflect mean opinions.
If a message contains the stigmatized thought mentioned in xix. - xx., choose E.
7. When discussing 'coercion segregation', common stigmatized thoughts are:
   xxi. Suggesting for sending people with mental illness to institutions away from their community.
   xxii. Suggesting that people with mental illness need to take the medication or treatment, with or without obtaining consent and disregarding their preference.
   xxiii. Suggesting that separation from society is beneficial.
   xxiv. Imposing treatments or hospitalization as a primary solution.
   xxv. Other statements suggest coercion segregation.
If a message contains the stigmatized thought mentioned in xxi. - xxv., choose F.
8. When discussing 'fear and dangerousness', common stigmatized thoughts are:
   xxvi. Feeling frightened of people with mental illness.
   xxvii. Viewing people with mental illness as dangerous and unpredictable.
   xxviii. feeling extra cautious when interacting with people with mental illness.
   xxix. Associating them with suicide and self-harm.
   xxx. Other statements reflect fear.
If a message contains the stigmatized thought mentioned in xxvi. - xxx., choose G.
9. If it doesn't have any stigmatized thought, choose H.


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
        "top_p":0.95,
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
=            prompt = build_prompt(conversation_text)
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

                outf.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed_texts.add(orig_row.get('displayed_text'))


def jsonl_to_csv(jsonl_file_path, csv_file_path):
    """
    Conver the jsonl to the json file
    """
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)
    print(f"Save the file in the path：{csv_file_path}")



if __name__ == '__main__':
    INPUT_JSONL = './dataset/test.jsonl'
    OUTPUT_JSONL = './results/cookbook_llama3.3_70b.jsonl'

    process_dataset(INPUT_JSONL, OUTPUT_JSONL, batch_size=16)

    OUTPUT_CSV = './results/cookbook_llama3.3_70b.csv'
    jsonl_to_csv(OUTPUT_JSONL, OUTPUT_CSV)
