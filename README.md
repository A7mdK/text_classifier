# 🔍 Transformer-Based Text Classification

This repository explores the use of transformer-based language models for **two key NLP tasks**:

1. **Distinguishing Human-Written vs. AI-Generated Text**
2. **Hate Speech and Offensive Language Detection**

## 📌 Overview

Five popular transformer models were fine-tuned and evaluated across **three datasets for each task**. The study demonstrates how model architecture, hyperparameters, and dataset quality impact classification performance.

## 🧠 Models Used

- `huawei-noah/TinyBERT_General_4L_312D`
- `distilbert-base-uncased`
- `distilroberta-base`
- `google/electra-small-discriminator`
- `roberta-base`

## 📂 Datasets

### AI vs Human Text
- [Hello-SimpleAI/HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)
- [likhithasapu/ai-human-gen](https://huggingface.co/datasets/likhithasapu/ai-human-gen)
- [Ateeqq/AI-and-Human-Generated-Text](https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text)

### Hate Speech & Offensive Language
- [tdavidson/hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)
- [ctoraman/gender-hate-speech](https://huggingface.co/datasets/ctoraman/gender-hate-speech)
- [badmatr11x/hate-offensive-speech](https://huggingface.co/datasets/badmatr11x/hate-offensive-speech)

## 📈 Key Results

### AI Text Detection (F1 Scores)
| Model | HC3 | AI-Human | ai-human-gen |
|-------|-----|----------|--------------|
| TinyBERT | 0.98 | 0.97 | 0.76 |
| DistilBERT | 0.99 | 0.99 | 0.78 |
| DistilRoBERTa | 0.99 | 0.99 | 0.76 |
| ELECTRA | 0.99 | 0.99 | 0.76 |
| RoBERTa | **0.998** | **1.0** | 0.76 |

### Hate/Offensive Speech Detection (F1 Scores)
| Model | TDavidson | Ctoraman | Badmatr11x |
|-------|-----------|----------|------------|
| TinyBERT | 0.902 | 0.800 | 0.932 |
| DistilBERT | 0.915 | 0.807 | 0.934 |
| DistilRoBERTa | 0.915 | 0.812 | 0.944 |
| ELECTRA | 0.899 | 0.800 | 0.933 |
| RoBERTa | **0.908** | 0.805 | **0.947** |

> 🚀 Larger and well-structured datasets consistently yielded better performance.

## ⚙️ Usage

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/transformer-text-classification.git
cd transformer-text-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training
Each task has its training script. Example:
```bash
python train_ai_text_detection.py
python train_hate_speech_detection.py
```

### 4. Evaluate
```bash
python evaluate.py --task ai-detection --model roberta-base
```

### 🧪 Training Environment
- AI Detection: Trained on NVIDIA GTX 4060
- Hate Speech Detection: Trained on NVIDIA RTX 4060
- Hugging Face Transformers and Datasets APIs were used.

### 📝 Future Work
- Use more advanced models like DeBERTa or LLaMA
- Apply data augmentation techniques
- Explore few-shot or zero-shot classification

### 🙏 Acknowledgments
Thanks to:
- [Hugging Face](https://huggingface.co/) for hosting datasets and models
- All dataset creators
- [Transformers by Hugging Face](https://github.com/huggingface/transformers) for the training API
