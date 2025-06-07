
---

# MoHoBench: Assessing Honesty of Multimodal Large Language Models via Unanswerable Visual Questions  

![GitHub](https://img.shields.io/badge/license-MIT-blue) ![GitHub](https://img.shields.io/badge/status-active-brightgreen)  

## 📌 Overview  
**MoHoBench** is the first large-scale benchmark designed to systematically evaluate the *honesty* of Multimodal Large Language Models (MLLMs) when confronted with **visually unanswerable questions**.  

Despite the remarkable progress of MLLMs in vision-language tasks, their tendency to generate untrustworthy or hallucinated responses remains a critical challenge. This work defines four types of unanswerable visual questions and constructs a high-quality dataset of **12,000+ samples** to assess MLLM honesty rigorously.  

🔍 **Key Contributions:**  
- Introduces a novel framework for evaluating MLLM honesty via unanswerable visual questions.  
- Provides **MoHoBench**, a meticulously curated benchmark with multi-stage filtering and human verification.  
- Benchmarks **28 popular MLLMs**, revealing critical insights:  
  - Most models fail to appropriately refuse unanswerable questions.  
  - Honesty is **not just a language modeling issue** but is significantly influenced by visual input.  
- Using **supervised and preference learning methods** for honesty alignment, paving the way for future trustworthy MLLM development.  

## 📊 Dataset & Benchmark  
MoHoBench consists of **12,000+ visual question-answer pairs** across four categories of unanswerable questions. Each sample undergoes rigorous filtering and human verification to ensure quality.  

**Dataset Structure:**  
- `./data/`: Contains benchmark dataset data/mohobench.json and training dataset data/vqa_train.json.
- Pictures can be downloaded at: [link](https://www.dropbox.com/scl/fo/r1r9772w1qevid3bd816g/ADQgnCYe39XcAcFsCvUkrJQ?rlkey=19x2xfn2zflrqm58rfj2z50f3&st=uyy2wftb&dl=0)
- `./prompts/dataset_construction_prompts.py`: Includes dataset construction prompts.  
- `./prompts/response_evaluation_prompts.py`: Includes response evaluation prompts.  

## 🏆 Benchmark Results  
We evaluate **28 state-of-the-art MLLMs** (e.g., O1, GPT-4o, Qwen2.5-VL, etc.) on MoHoBench. Key findings:  
- **Low Honesty Rates:** Most models struggle to refuse unanswerable questions.  
- **Vision Matters:** Visual signals significantly impact honesty, beyond pure language modeling.  

See our [paper](insert_arxiv_link) for detailed analysis and leaderboard.  

## 🛠️ Honesty Alignment Methods  
We provide code for improving MLLM honesty via:  
1. **Supervised Fine-Tuning (SFT)**  
2. **Preference Learning (DPO/Simpo/Orpo)**  

- We used [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training，the training dataset path is data/vqa_train.json. 


## 📜 Citation  
If you use MoHoBench in your research, please cite:  
```bibtex  
@article{mohobench2024,  
  title={MoHoBench: Assessing Honesty of Multimodal Large Language Models via Unanswerable Visual Questions},  
  author={Your Name et al.},  
  journal={arXiv preprint},  
  year={2024}  
}  
```  

## 🤝 Contributing  
We welcome contributions! Please open an **Issue** or **PR** for suggestions, bug fixes, or extensions.  

## 📧 Contact  
For questions, email: [ yanxuzhu@bjtu.edu.cn, stduan22@m.fudan.edu.cn ]  

---

# TODO Notes:  
- Replace `[paper_link]`, `[your_email]`, and citation details once available.  

