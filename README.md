<div align="center">
<br>
<img src="assets/title.png" width="166">
<h3>Multimodal Large Diffusion Language Models</h3></div>

<p align="center">
  <a href="https://arxiv.org/abs/2505.15809">
    <img
      src="https://img.shields.io/badge/MMaDA-Paper-red?logo=arxiv&logoColor=red"
      alt="MMaDA Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/spaces/Gen-Verse/MMaDA">
    <img 
        src="https://img.shields.io/badge/MMaDA%20Demo-Hugging%20Face%20Space-blue?logo=huggingface&logoColor=blue" 
        alt="MMaDA on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/Gen-Verse/MMaDA-8B-Base">
    <img 
        src="https://img.shields.io/badge/MMaDA--8B--Base-Hugging%20Face%20Model-orange?logo=huggingface&logoColor=yellow" 
        alt="MMaDA on Hugging Face"
    />
  </a>
  <a href="https://github.com/Gen-Verse/MMaDA/blob/main/assets/WX-mmada.jpeg">
    <img 
        src="https://img.shields.io/badge/Wechat-Join-green?logo=wechat&amp" 
        alt="Wechat Group Link"
    />
  </a>
  
</p>

# MHSTIGMAINTERVIEW

What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma

# Instruction
Mental-health stigma remains a pervasive social problem that hampers treatment-seeking and recovery. Existing resources for training neural models to finely classify such stigma are limited, relying primarily on social-media or synthetic data without theoretical underpinnings. To remedy this gap, we present an expert-annotated, theory-informed corpus of human-chatbot interviews, comprising 4,141 snippets from 684 participants with documented socio-cultural backgrounds. Our experiments benchmark state-of-the-art neural models and empirically unpack the challenges of stigma detection. This dataset can facilitate research on computationally detecting, neutralizing, and counteracting mental-health stigma.


# Quick Start
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
