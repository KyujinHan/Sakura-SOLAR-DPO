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
| [Yi-34B](https://huggingface.co/01-ai/Yi-34B) | 69.42 | 64.59 | 85.69 | 76.35 | 56.23 | 83.03 | 50.64 |
| [SOLAR-10.7B-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-v1.0) | 660.4 | 61.95 | 84.60 | 65.48 | 45.04 | 83.66 | 55.50 |
> Follow up as [link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).  

# Hyperparameters & Prompt
- 😎kyujinpy/Sakura-SOLAR-Instruct
```
(Coming soon...)
```
---
- 😎kyujinpy/Sakura-SOLAR-Instruct-DPO-v1
   
| Hyperparameter      | kyujinpy/Sakura-SOLAR-Instruct-DPO-v1 |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_8bit        | True   |
| learning rate       | 1e-6   |
| batch size          | 32     |
| microbatch  size    | 2      |
| warmup ratio        | 0.1    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | linear |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| beta                | 0.1    |
| optim               | adamw  |
| lora target modules | `embed_tokens, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head` |
| cutoff length       | 4096   |
| Datasets            | [argilla/distilabel-math-preference-dpo](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo) |  
| Base Model          | [kyujinpy/Sakura-SOLAR-Instruct](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct) |
```
### User:

### Assistant: 
```
> Prompting  
---  
- 😎kyujinpy/Sakura-SOLAR-Instruct-DPO-v2
  
| Hyperparameter      | kyujinpy/Sakura-SOLAR-Instruct-DPO-v2 |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_8bit        | True   |
| learning rate       | 3e-7   |
| batch size          | 32     |
| microbatch  size    | 2      |
| warmup ratio        | 0.1    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | linear |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| beta                | 0.1    |
| optim               | adamw  |
| lora target modules | `embed_tokens, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head` |
| cutoff length       | 4096   |
| Datasets            | [argilla/distilabel-math-preference-dpo](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo) |  
| Base Model          | [kyujinpy/Sakura-SOLAR-Instruct](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct) |
```
### User:

### Assistant:
```
> Prompting
  
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
