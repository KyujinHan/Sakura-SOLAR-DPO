# Sakura-SOLAR-DPO  

**Sakura-SOLAR project; Purpose is `the global LLM rank 1.`**    
I noted that all most of things about the PlatYi models, which is `the global LLM Rank 1 on December, 2023`.  
⭐So, please give me a star⭐~~!!   

I hope, the opensource more and more develop!😄😄  

# Contents  
- [Model Performance](https://github.com/KyujinHan/PlatYi-34?tab=readme-ov-file#model-performance)  
- [Hyperparameters & Prompt](https://github.com/KyujinHan/PlatYi-34?tab=readme-ov-file#hyperparameters--prompt)  
- [Some Insight](https://github.com/KyujinHan/PlatYi-34?tab=readme-ov-file#some-insight)  
- [References](https://github.com/KyujinHan/PlatYi-34?tab=readme-ov-file#references)  
  
# (Quick) Model lists
- (Coming soon...)

# Introduction
- Recently, I created the [Ko-platypus🥮](https://github.com/Marker-Inc-Korea/KO-Platypus) LLM, which was `Korean LLM Rank 1`.  
   
# News
(Coming soon...)🤗🤗

# Model Performance
| Model | Average | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [PlatYi-34B-Llama-Q](https://huggingface.co/kyujinpy/PlatYi-34B-Llama-Q) | 71.13 | 65.70 | 85.22 | 78.78 | 53.64 | 83.03 | 60.42 |
| [PlatYi-34B-Q](https://huggingface.co/kyujinpy/PlatYi-34B-Q) | 69.86 | 66.89 | 85.14 | 77.66 | 53.03 | 82.48 | 53.98 |
| [PlatYi-34B-LoRA](https://huggingface.co/kyujinpy/PlatYi-34B-LoRA) | 68.1 | 67.15 | 85.37 | 78.46 | 53.32 | 83.66 | 40.64 |  
| [Yi-34B-Llama](https://huggingface.co/chargoddard/Yi-34B-Llama) | 70.95 | 64.59 | 85.63 | 76.31 | 55.60 | 82.79 | 60.80 |
| [Yi-34B](https://huggingface.co/01-ai/Yi-34B) | 69.42 | 64.59 | 85.69 | 76.35 | 56.23 | 83.03 | 50.64 |
> Follow up as [link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).  

# Hyperparameters & Prompt
- 😎kyujinpy/Sakura-SOLAR-Instruct
```
(merge information)
```
---
- 😎kyujinpy/Sakura-SOLAR-Instruct-DPO-v1
   
| Hyperparameter      | kyujinpy/Sakura-SOLAR-Instruct-DPO-v1 |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_4bit        | True   |
| learning rate       | 2e-5   |
| batch size          | 16     |
| microbatch  size    | 1      |
| warmup steps        | 100    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | cosine |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| lora target modules | gate_proj, up_proj, down_proj |
| cutoff length       | 4096   |
| train on inputs     | False  |
| group by length     | False  |
| add eos token       | False  |
| Datasets            | [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) |  
```
### User:

### Assistant: 
```
---  
- 😎kyujinpy/Sakura-SOLAR-Instruct-DPO-v2
  
| Hyperparameter      | kyujinpy/Sakura-SOLAR-Instruct-DPO-v2 |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_4bit        | True   |
| learning rate       | 4e-4   |
| batch size          | 16     |
| microbatch  size    | 1      |
| warmup steps        | 0      |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | cosine |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| lora target modules | gate_proj, up_proj, down_proj |
| cutoff length       | 4096   |
| train on inputs     | False  |
| group by length     | False  |
| add eos token       | False  |
| Datasets            | [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) |  
```
### User:

### Assistant:
```
   
# Some Insight
(Coming soon)🤩🤩

# TODO
- [ ] Share code
- [ ] Share hyperparameters
- [ ] Share insight
- [ ] Share datasets

# References
- [Platypus](https://platypus-llm.github.io/)  
- [Ko-platypus](https://github.com/Marker-Inc-Korea/KO-Platypus)  
