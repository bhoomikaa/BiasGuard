# BiasGuard

**Automated Bias Detection in Text using Generative & Transfer Learning**  
*Python · PyTorch · Hugging Face · AWS · Transformers (BERT | RoBERTa | LLaMA2)*

---

## 🚀 Project Overview  
BiasGuard detects and mitigates societal bias in large-scale text data through a cloud-native AI pipeline. The system:  
- Scrapes and preprocesses Reddit data for bias-related content  
- Fine-tunes large-language models (BERT, RoBERTa, LLaMA2) for bias detection  
- Achieves **82% accuracy** on real-world political sentiment classification  
- Employs mixed-precision training and differential learning rates to reduce overfitting and cut training time by **~40%**

---

## 🧠 Key Features  
- **End-to-end pipeline**: Ingestion → Pre-processing → Model Training → Evaluation → Deployment  
- **Transfer-learning & LLMs**: Customised fine-tuning of state-of-the-art transformers  
- **Cloud infrastructure**: Built on AWS (Lambda, S3, SageMaker) for scalable processing  
- **Real-time results**: Rapid inference on new text data with low latency  

---

## 🛠️ Tech Stack  
- **Languages**: Python ≥ 3.10  
- **Deep-Learning**: PyTorch, Transformers (Hugging Face), Mixed Precision  
- **Data & Cloud**: AWS (S3, Lambda, SageMaker), Docker  
- **Best Practices**: Differential learning rates, early-stopping, clean modular architecture  

---

## 📁 Repo Structure  
BiasGuard/      
├── notebooks/ # Exploratory analysis & bias-model prototyping         
├── scripts/ # Data-ingestion, preprocessing & training scripts            
├── README.md # This file            
└── requirements.txt # Project dependencies              
## 🔧 Quick Start                 
```bash
git clone https://github.com/bhoomikaa/BiasGuard.git
cd BiasGuard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Run training or inference scripts as needed…

