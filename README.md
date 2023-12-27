# Sakura-SOLAR-DPO  
<img src='./sakura.png' width=512>

**Sakura-SOLAR project;**    
I noted that all most of things about the **Sakura-SOLAR models** which is `the global LLM Rank 1 on December, 2023`.  
⭐So, please give me a star⭐~~!!   

I hope, the open-source more and more develop!😄😄  

# Contents  
- [Model Performance](https://github.com/KyujinHan/PlatYi-34?tab=readme-ov-file#model-performance)  
- [Hyperparameters & Prompt](https://github.com/KyujinHan/PlatYi-34?tab=readme-ov-file#hyperparameters--prompt)  
- [Some Insight](https://github.com/KyujinHan/PlatYi-34?tab=readme-ov-file#some-insight)  
- [References](https://github.com/KyujinHan/PlatYi-34?tab=readme-ov-file#references)  
  
# (Quick) Model lists
- [🌸kyujinpy/Sakura-SOLAR-Instruct](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct)
- [🌸kyujinpy/Sakura-SOLAR-Instruct-DPO-v1](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct-DPO-v1)
- [🌸kyujinpy/Sakura-SOLAR-Instruct-DPO-v2](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct-DPO-v2)
- [🌸kyujinpy/Sakura-SOLRCA-Instruct-DPO🐋](https://huggingface.co/kyujinpy/Sakura-SOLRCA-Instruct-DPO)
- [🌸🐋kyujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v1📐](https://huggingface.co/kyujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v1)
- [🌸🐋kyujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v2📐](https://huggingface.co/kyujinpy/Sakura-SOLRCA-Math-Instruct-DPO-v2)

# Introduction
- Recently, I created the [Ko-platypus🥮](https://github.com/Marker-Inc-Korea/KO-Platypus) LLM, which was `Open-Ko LLM Rank 1`.
- Also, I created the [🌸kyujinpy/Sakura-SOLAR-Instruct](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct) LLM, which is `Open LLM Rank 1`.
- **I loved open-source, I wanted to share everything about the model that won first rank.**
- I hope this GitHub helps a lot of people.😎😎
     
# News 
- 2023.12.28
    - **Rank1** (Open LLM leaderboard): **🌸kyujinpy/Sakura-SOLAR-Instruct** 

# Model Performance
| Model | Average | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) | 74.20 | 71.08 | 88.16 | 66.21 | 71.43 | 83.58 | 64.75 |
| [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) | 72.62 | 70.22 | 87.63 | 71.16 | 64.58 | 81.37 | 60.73 |
| [PlatYi-34B-Llama-Q](https://huggingface.co/kyujinpy/PlatYi-34B-Llama-Q) | 71.13 | 65.70 | 85.22 | 78.78 | 53.64 | 83.03 | 60.42 |
| [Yi-34B](https://huggingface.co/01-ai/Yi-34B) | 69.42 | 64.59 | 85.69 | 76.35 | 56.23 | 83.03 | 50.64 |
> Follow up as [link](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).  

# Hyperparameters & Prompt
- `😎kyujinpy/Sakura-SOLAR-Instruct`
```
slices:
  - sources:
      - model: VAGOsolutions/SauerkrautLM-SOLAR-Instruct
        layer_range: [0, 48]
      - model: upstage/SOLAR-10.7B-Instruct-v1.0
        layer_range: [0, 48]
        
merge_method: slerp
base_model: upstage/SOLAR-10.7B-Instruct-v1.0

parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5 # fallback for rest of tensors
tokenizer_source: union
    
dtype: float16
```
---
- `😎kyujinpy/Sakura-SOLAR-Instruct-DPO-v1`
   
| Hyperparameter      | kyujinpy/Sakura-SOLAR-Instruct-DPO-v1 |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_8bit        | True   |
| learning rate       | 1e-6   |
| batch size          | 32     |
| micro batch size    | 2      |
| warmup ratio        | 0.1    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | linear |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| beta                | 0.1    |
| optim               | adamw_torch |
| bf16                | True   |
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
- `😎kyujinpy/Sakura-SOLAR-Instruct-DPO-v2`
  
| Hyperparameter      | kyujinpy/Sakura-SOLAR-Instruct-DPO-v2 |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_8bit        | True   |
| learning rate       | 1e-5   |
| batch size          | 32     |
| micro batch size    | 2      |
| warmup ratio        | 0.1    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | linear |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| beta                | 0.1    |
| optim               | paged_adamw_32bit |
| bf16                | True   |
| lora target modules | `embed_tokens, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head` |
| cutoff length       | 4096   |
| Datasets            | [argilla/distilabel-math-preference-dpo](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo) |  
| Base Model          | [kyujinpy/Sakura-SOLAR-Instruct](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct) |
```
### User:

### Assistant:
```
---
- `😎kyujinpy/Sakura-SOLRCA-Instruct-Dpo`
  
| Hyperparameter      | kyujinpy/Sakura-SOLRCA-Instruct-Dpo |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_8bit        | True   |
| learning rate       | 5e-7   |
| batch size          | 32     |
| micro batch size    | 1      |
| warmup ratio        | 0.1    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | linear |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| beta                | 0.1    |
| optim               | paged_adamw_32bit |
| bf16                | True   |
| lora target modules | `embed_tokens, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head` |
| cutoff length       | 4096   |
| Datasets            | [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) |  
| Base Model          | [kyujinpy/Sakura-SOLAR-Instruct](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct) |
```
### User:

### Assistant:
```
---
- `😎kyujinpy/Sakura-SOLRCA-Math-Instruct-Dpo-v1`
  
| Hyperparameter      | kyujinpy/Sakura-SOLRCA-Math-Instruct-Dpo-v1 |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_8bit        | True   |
| learning rate       | 5e-7   |
| batch size          | 32     |
| micro batch size    | 2      |
| warmup ratio        | 0.1    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | linear |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| beta                | 0.1    |
| optim               | paged_adamw_32bit |
| bf16                | True   |
| lora target modules | `embed_tokens, q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head` |
| cutoff length       | 4096   |
| Datasets            | [kyujinpy/orca_math_dpo](https://huggingface.co/datasets/kyujinpy/orca_math_dpo) |  
| Base Model          | [kyujinpy/Sakura-SOLAR-Instruct](https://huggingface.co/kyujinpy/Sakura-SOLAR-Instruct) |
```
### User:

### Assistant:
```
---
- `😎kyujinpy/Sakura-SOLRCA-Math-Instruct-Dpo-v2`
  
| Hyperparameter      | kyujinpy/Sakura-SOLRCA-Math-Instruct-Dpo-v2 |
|---------------------|--------|
| LoRA method         | LoRA   |
| load_in_8bit        | True   |
| learning rate       | 5e-7   |
| batch size          | 32     |
| micro batch size    | 2      |
| warmup ratio        | 0.1    |
| epochs              | 1      |
| weight decay        | 0.     |
| lr scheduler        | linear |
| lora alpha          | 16     |
| lora rank           | 16     |
| lora dropout        | 0.05   |
| beta                | 0.1    |
| optim               | paged_adamw_32bit |
| bf16                | True   |
| lora target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head` |
| cutoff length       | 4096   |
| Datasets            | [kyujinpy/orca_math_dpo](https://huggingface.co/datasets/kyujinpy/orca_math_dpo) |  
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
- [x] Share hyperparameters
- [ ] Share insight
- [x] Share datasets

# References
- [Platypus](https://platypus-llm.github.io/)  
- [Ko-platypus](https://github.com/Marker-Inc-Korea/KO-Platypus)  
