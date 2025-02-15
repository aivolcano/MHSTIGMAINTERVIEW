
# -*- coding: utf-8 -*-
# @Time    : 2025/2/9 下午10:00
# @Author  : Yancan Chen
# @Email   : yancan@u.nus.edu
# @File    : roberta.py

import json  
import torch
import torch.nn as nn
import pandas as pd
import random
import ast

import numpy as np
import os
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, cohen_kappa_score, classification_report
from collections import Counter
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('social_stigma')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 加载 train.csv 和 test.csv
train_df = pd.read_csv('./dataset/train.csv')
test_df = pd.read_csv('./dataset/test.csv')
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_id = 'hf_models/roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_id)
model = RobertaForSequenceClassification.from_pretrained(model_id, num_labels=8)

# 更新模型的标签映射，确保只使用8个类别
label_dict = {'Non-stigmatized': 0,
              'Stigmatized (responsibility)': 1,
              'Stigmatized (social distance)': 2,
              'Stigmatized (fear/dangerousness)': 3,
              'Stigmatized (anger)': 4,
              'Stigmatized (coercion segregation)': 5,
              'Stigmatized (helping)': 6,
              'Stigmatized (pity)': 7}
id2label_custom = {v: k for k, v in label_dict.items()}  # {0: 'Non-stigmatized', 1: 'Stigmatized (responsibility)', ...}
model.config.id2label = id2label_custom
model.config.label2id = label_dict


def encoder_text(examples):
    texts = []
    # examples["conversations"] 是一个列表，每个元素为该样本的对话列表（对话列表内每个元素为字典，包含 role 和 content）
    for conv in examples["conversations"]:
        # 使用列表解析，对每条对话信息进行拼接
        conv = ast.literal_eval(conv)
        conversation_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in conv])
        texts.append(conversation_text)
    return tokenizer(texts, max_length=512, padding='max_length', truncation=True)


def encoder_label(examples):
    examples['label'] = label_dict[examples['label']]
    return examples

# 对训练集和测试集进行编码
train_dataset = train_dataset.map(encoder_text, batched=True)
train_dataset = train_dataset.map(encoder_label)
test_dataset = test_dataset.map(encoder_text, batched=True)
test_dataset = test_dataset.map(encoder_label)

device = "cuda:0"  # if torch.cuda.is_available() else "cpu"

def classification_metrics(y_true, y_pred, labels):
    classification_scores = {}
    class_counts = Counter(y_true)  # Count occurrences of each class
    overall_accuracy = accuracy_score(y_true, y_pred)
    sum_precision = sum_recall = sum_f1 = 0

    for label in labels:
        y_true_binary = [1 if y == label else 0 for y in y_true]
        y_pred_binary = [1 if y == label else 0 for y in y_pred]

        kappa = cohen_kappa_score(y_true_binary, y_pred_binary)
        recall_val = recall_score(y_true_binary, y_pred_binary)
        precision_val = precision_score(y_true_binary, y_pred_binary)
        f1_val = f1_score(y_true_binary, y_pred_binary)
        support = class_counts[label]
        acc = accuracy_score(y_true_binary, y_pred_binary)

        sum_precision += precision_val
        sum_recall += recall_val
        sum_f1 += f1_val

        classification_scores[label] = {
            'precision': precision_val,
            'recall': recall_val,
            'f1-score': f1_val,
            'support': support,
            'cohen_kappa': kappa,
            'accuracy': acc
        }

    macro_avg_precision = sum_precision / len(labels)
    macro_avg_recall = sum_recall / len(labels)
    macro_avg_f1 = sum_f1 / len(labels)

    weighted_avg_precision = np.average([score['precision'] for score in classification_scores.values()],
                                        weights=[score['support'] for score in classification_scores.values()])
    weighted_avg_recall = np.average([score['recall'] for score in classification_scores.values()],
                                     weights=[score['support'] for score in classification_scores.values()])
    weighted_avg_f1 = np.average([score['f1-score'] for score in classification_scores.values()],
                                 weights=[score['support'] for score in classification_scores.values()])

    classification_scores['accuracy'] = overall_accuracy
    classification_scores['macro avg'] = {'precision': macro_avg_precision, 'recall': macro_avg_recall, 'f1-score': macro_avg_f1}
    classification_scores['weighted avg'] = {'precision': weighted_avg_precision, 'recall': weighted_avg_recall, 'f1-score': weighted_avg_f1}

    return classification_scores

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(predictions, labels, average='weighted')
    recall_val = recall_score(predictions, labels, average='weighted')
    precision_val = precision_score(predictions, labels, average='weighted')
    accuracy_val = accuracy_score(predictions, labels)
    cr = classification_report(predictions, labels)
    cohen_kappa_val = cohen_kappa_score(predictions, labels)
    return {'f1_score': f1,
            'recall_score': recall_val,
            'precison_score': precision_val,
            'accuracy_score': accuracy_val,
            'classification_report': cr,
            'cohen_kappa': cohen_kappa_val}

train_arg = TrainingArguments(
    output_dir='./results/{}'.format(model_id),
    num_train_epochs=3,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    logging_dir='./logs/{}'.format(model_id),
    use_mps_device=False,
    gradient_accumulation_steps=4,
    seed=42,
    learning_rate=1e-5, 
    eval_steps=50,
    save_strategy='steps',
    save_steps=50,
    evaluation_strategy='steps',
    load_best_model_at_end=True,  # load the best model at the end of training
    metric_for_best_model='f1_score',
    report_to='tensorboard',
)
logger.info(train_arg)

trainer = Trainer(
    model=model,
    args=train_arg,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
logger.info('Training in progress')
trainer.train()
logger.info('Finish Training')

logger.info('Evaluating in progress')
trainer.evaluate()
logger.info('Finish Evaluating')

model.save_pretrained('./model_directory/{}'.format(model_id))

### prediction
predictions = trainer.predict(test_dataset)
# 使用更新后的 id2label 进行预测，确保只包含8个类别
predicted_labels = [model.config.id2label[pred] for pred in predictions.predictions.argmax(axis=1)]
# 从数据集中获取真实标签（数字形式）
true_labels = test_dataset['label']
# 将真实标签转换为字符串形式
true_labels_str = [model.config.id2label[label] for label in true_labels]

# 计算分类报告，确保两边都是字符串
classification_rep = classification_report(true_labels_str, predicted_labels, target_names=list(label_dict.keys()))
cohen_kappa_val = cohen_kappa_score(true_labels_str, predicted_labels)

print(classification_rep)
print(cohen_kappa_val)
print('classification_metrics------------------',
      classification_metrics(y_true=true_labels, y_pred=[label_dict[label] for label in predicted_labels], labels=label_dict.values()))
   

# 输出部分：从测试集获取文本数据（此处依然使用 displayed_text，如有需要可改为 conversations 拼接后的文本）
def conversation_to_text(conv_str):
    """
    将 conversations 字段（字符串形式）转换为拼接后的文本。
    """
    try:
        conv = ast.literal_eval(conv_str)
        return " ".join([f"{msg['role']}: {msg['content']}" for msg in conv])
    except Exception as e:
        logger.error("Error parsing conversation: %s", e)
        return ""

# 针对测试集中的每个样本，从 conversations 字段生成文本
test_texts = [conversation_to_text(example["conversations"]) for example in test_dataset]

# 构造输出字典，注意：true_labels_str 与 predicted_labels 均为字符串形式
output_data = {
    'text': test_texts,
    'ground_truth': true_labels_str,
    'pred_label': predicted_labels
}
print(predicted_labels)

output_df = pd.DataFrame(output_data)
output_df.to_csv('./results/roberta_predictions.csv', index=False, encoding='utf-8')
print('Saved successfully!')
