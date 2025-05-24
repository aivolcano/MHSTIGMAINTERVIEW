The dataset is partinationed into trainset and testset. 

### Label Distribution

Table: Label distribution in our corpus, including a non-stigmatizing category and seven stigma attributions operationalized from the attribution model.
| Label                  | # Snippet | %     |
|------------------------|-----------|-------|
| Non-stigmatizing       | 2,232     | 53.90 |
| **Stigmatizing**       | -         | -     |
| &nbsp;&nbsp;Responsibility         | 394       | 9.51  |
| &nbsp;&nbsp;Social Distance        | 379       | 9.15  |
| &nbsp;&nbsp;Anger                 | 298       | 7.20  |
| &nbsp;&nbsp;Helping               | 158       | 3.82  |
| &nbsp;&nbsp;Pity                  | 42        | 1.01  |
| &nbsp;&nbsp;Coercive Segregation | 271       | 6.54  |
| &nbsp;&nbsp;Fear                 | 367       | 8.86  |
| **Total**              | 4,141     | 100   |



### Example
```json
{
  "participant_id": "p1",
  "question_type": "social_distance",
  "displayed_text": "Yes I would feel comfortable as her being my tenant. You just have to go by feeling you will know upon meeting the person.",
  "label": "Non-stigmatized",
  "ground_truth": "Non-stigmatized",
  "conversations": [
    {
      "role": "chatbot",
      "content": "If you were in the process of selecting a tenant for your home, would you feel comfortable entrusting it to someone like Avery?"
    },
    {
      "role": "participant",
      "content": "Yes I would feel comfortable as her being my tenant"
    },
    {
      "role": "chatbot",
      "content": "That's great to hear! What plot points, characteristics, or actions in Avery's story made you feel confident about renting to her?"
    },
    {
      "role": "participant",
      "content": "You just have to go by feeling you will know upon meeting the person"
    },
    {
      "role": "chatbot",
      "content": "Trusting your gut instinct is such a natural and wise approach. It's amazing how intuition often guides us right!"
    }
  ]
}
```
