import json
import pandas as pd
from transformers import pipeline, TextClassificationPipeline
import statsmodels.api as sm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# # 1. Load the jsonl data, one json object per row.
data = []
for f_name in ['./mhs_dataset/test.jsonl', './mhs_dataset/train.jsonl']:
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
df = pd.DataFrame(data)

# Make sure the data contains 'text' and 'stigma' attributions
print(df.head())

# 2. Predict the emotion of the text using a pre-trained emotion classification model
classifier = pipeline('text-classification', model='bhadresh-savani/bert-base-uncased-emotion')

def get_emotion(text):
    # Classify the text and return the predicted emotion label
    result = classifier(text)
    print('-------')
    print(result)
    # return result[0]['label']
    return result[0]['score']

# Add the emotion result as a new column 'emotion'
df['emotion'] = df['displayed_text'].apply(get_emotion)

print(df.head())
df.to_csv('./mhs_dataset/train_test_with_emotion.csv',index=False)




# df = pd.read_csv('./mhs_dataset/train_test_with_emotion.csv')

print('--------/n emotion ------', df['emotion'].unique())


# 2. Calculate the percentage of each user for each question_type
qt_count = df.groupby(['participant_id', 'ground_truth']).size().reset_index(name='count_qt')
total_qt = qt_count.groupby('participant_id')['count_qt'].sum().reset_index(name='total_qt')
qt_count = pd.merge(qt_count, total_qt, on='participant_id')
qt_count['prop_qt'] = qt_count['count_qt'] / qt_count['total_qt']
# 转换为宽表，每行一个 participant_id，列为各 ground_truth 的占比
qt_wide = qt_count.pivot(index='participant_id', columns='ground_truth', values='prop_qt').fillna(0)

# 3. Calculate the proportion of each user's emotions
em_count = df.groupby(['participant_id', 'emotion']).size().reset_index(name='count_em')
total_em = em_count.groupby('participant_id')['count_em'].sum().reset_index(name='total_em')
em_count = pd.merge(em_count, total_em, on='participant_id')
em_count['prop_em'] = em_count['count_em'] / em_count['total_em']
# Convert to a wide table with one participant_id per row and the proportion of each emotion
em_wide = em_count.pivot(index='participant_id', columns='emotion', values='prop_em').fillna(0)

# 4. Make sure that all expected emotion columns are present, and fill them with 0 even if some participants do not have a certain emotion recorded
expected_emotions = ['joy', 'love', 'surprise', 'anger', 'fear', 'sadness', 'neutral']
em_wide = em_wide.reindex(expected_emotions, axis=1, fill_value=0)
# Rename the columns of the emotion wide table to avoid the same names as ground_truth (add the prefix "emo_")
em_wide = em_wide.rename(columns=lambda x: f"emo_{x}")

# Check the columns of the emotion wide table
print("em_wide columns:", em_wide.columns.tolist())

# 5. Merge summary data for interaction topic and sentiment (participant_id as in-index join)
merged_df = pd.merge(qt_wide, em_wide, left_index=True, right_index=True, how='inner')
print("merged_df columns:", merged_df.columns.tolist())

# 6. For each emotion (dependent variable), OLS regression models were built with each interaction topic (independent variable)
predictors = list(qt_wide.columns)  # Independent variable: Proportion of each question type
for emo in expected_emotions:
    emotion_col = f"emo_{emo}"
    y = merged_df[emotion_col]      # Dependent variable: Proportion of corresponding emotions
    X = merged_df[predictors]         # Independent variable: Proportion of each interaction topic
    X = sm.add_constant(X)            # Add an intercept term
    model = sm.OLS(y, X).fit()
    print(f"OLS regression for emotion: {emo} (column: {emotion_col})")
    print(model.summary())
    print("\n")

