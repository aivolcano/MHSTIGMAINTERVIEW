# MHSTIGMAINTERVIEW





## 1. Datasets Description
Please check under the [dataset folder](./dataset/).

## 2. Run the experiments
### Set up environment 
```shell
# install latest version torch 

# Install the packages
pip install transformers scikit-learn pandas numpy

# install the flash_attention_2
pip3 install flash-attn --no-build-isolation
```

### Type your openai key token in the scripts if you want to run gpt-4o model.


### Run the scripts in the multiple GPUs

Run LLaMA family model
```python
python llama_cookbook.py
python llama_few_shot.py
python llama_zero-shot.py
```

Run Mistral family model

```python
python mistral_cookbook_norm.py
python mistral_few_shot_norm.py
python mistral_zero_shot_norm.py

python mixtral_8x7b_cookbook_prompt.py
python mixtral_8x7b_few_shot.py
python mixtral_8x7b_zero_shot.py
```


Run the roberta
```python 
python roberta.py
```
