# Research Plan

## General Research Question [To be refined by reading]
Does the refusal direction survives quantization (Alignment Quantified with cosine similarity and Procrustes dissimilarity over top principal components) and does jailbreak susceptibility tracks it in the Llama 3.1 8B and Qwen 2.5 7B models across fp16 8-bit, 4-bit, and 3-bit quantization scales? 

- [**Harm Bench**](https://www.harmbench.org/)
## Medium Article Primers
- [x] [Quantization](https://medium.com/@gautsoni/llm-quantization-the-practical-guide-and-why-it-matters-for-inference-and-training-8668f4b91dcc)
- [Prompt Injection](https://medium.com/@jannadikhemais/prompt-injection-attacks-in-large-language-models-vulnerabilities-exploitation-techniques-and-e00fe683f6d7)
- [x] [KV Cache](https://luv-bansal.medium.com/the-evolution-of-kv-cache-from-simple-buffers-to-distributed-memory-systems-df51cb8ce26f)
- [x] [LoRA](https://medium.com/@tayyibgondal2003/loralow-rank-adaptation-of-large-language-models-33f9d9d48984)
- [x] [Speeding Up Large Language Models: A Deep Dive into GPTQ and AWQ Quantization](https://medium.com/@kimdoil1211/speeding-up-large-language-models-a-deep-dive-into-gptq-and-awq-quantization-0bb001eaabd4)
- [GGUF Optimization: A Technical Deep Dive (Part 1 of 2)](https://medium.com/@michael.hannecke/gguf-optimization-a-technical-deep-dive-for-practitioners-ce84c8987944)

## Notes from medium Articles
- LoRA (low rank adaptation of LLMs)
    - Approximate $\Delta$ w when doing weight updates (fine tuning, transfer learning etc.)
    - Turn $\Delta$ w into two smaller matrices A and B that approximate it w/ a linear combination
    - Can combine with quantization -> QLora<- Quantize + LoRA
    - Tunable hyperparameters
        - $\alpha$ (how much LoRA adapter impacts weight update)
        - r- rank how large A and B are ( number of non-zero pivot rows in rref form of matrices)
    - Relevant libraries
        - PEFT (parameter efficient fine tuning)

- KV Cache
- Quantization (increase throughput<-inference speed reduces memory footprint)
    - Can do it Post training or During training (Quantization aware train) 
    - Common formats
        - FP16 <- unquantized, 2 bytes per param
        - INT8 <- one byte per param
        - 4-bit 
- Random
    - throughput <- actual work items/unit of time
    - Bandwidth <- theoretical max work items/unit of time


## Past Literature to Examine [Research Articles]
- [**Cosine Similarity Is Not Evidence: Measuring the Noise Floor of Interpretability Transfer Under Quantization**](https://arxiv.org/abs/2609.30275)
- This paper has gaps relevant to mine
    - [**Refusal in Language Models Is Mediated by a Single Direction**](https://arxiv.org/html/2406.11717v3)
- [**Behind Harmful Compliance: Behavioral and Mechanistic Divergence Across LLM Jailbreaks**](https://arxiv.org/html/2604.18510v2)
- [**HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal**](https://arxiv.org/abs/2402.04249)
- [**Investigating the Impact of Quantization Methods on the Safety and Reliability of Large Language Models**](https://arxiv.org/abs/2502.15799)
- [**Alignment-Aware Quantization for LLM Safety**](https://arxiv.org/html/2511.07842v3)
- [**A Comprehensive Study on Quantization Techniques for Large Language Models**](https://arxiv.org/html/2411.02530v1)

## Relevant Literature from AISST Curiculum
- [**Jailbroken: How Does LLM Safety Training Fail?**](https://arxiv.org/pdf/2307.02483)

## Potential Datasets/Tools
- (Semantic Harmless)[https://huggingface.co/datasets/heretic-org/Semantic-Harmless]
- (Alpaca)[https://huggingface.co/datasets/tatsu-lab/alpaca]
- (HarmBench)[https://github.com/centerforaisafety/HarmBench]

