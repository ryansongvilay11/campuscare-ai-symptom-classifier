# 🏥 CampusCare AI — Symptom Classification LLM  
A responsibility-aligned medical guidance assistant built by fine-tuning Llama-3.2-3B-Instruct using **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**.  
CampusCare AI provides calm, concise, and safe guidance for common student illnesses through a reproducible ML pipeline and a deployed Gradio interface.

---
## 🚀 Quick Start (Load the Final Model — No Training Needed)
If you want to skip all training steps, simply uncomment and run the first code cell at the top of the Colab notebook — it loads the final DPO model directly from Hugging Face.

```
!pip install unsloth accelerate transformers datasets gradio

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "ras6899/symptom-sft-dpo",  
    max_seq_length=2048,
    load_in_4bit=True,           # Efficient inference
)

FastLanguageModel.for_inference(model)
model.eval()

print("Model loaded successfully!")
```


---

## 🎯 Problem, Business Context & Target Audience

### The Problem  
College students frequently rely on internet searches to interpret their symptoms. Traditional search engines return a wide range of possible conditions—from minor colds to life-threatening illnesses—causing unnecessary anxiety and confusion.  
Example: “Persistent cough and chest pain” → results range from mild infections to pneumonia or lung disease.

There is a need for a focused, responsibility-aligned tool that responds safely, calmly, and within a tightly controlled medical scope.

### Our Solution — CampusCare AI  
CampusCare AI helps students interpret symptoms by mapping them to a limited set of **seven common illnesses**, plus **Unknown**:

- Cold  
- Flu  
- COVID-19  
- Allergies  
- Strep throat  
- Stomach flu  
- Bronchitis  
- **Unknown** (catch-all for vague or out-of-scope cases)

The model provides:
- A **single condition** (never multiple)  
- A **short, calm guidance paragraph**  
- A **fixed safety note**  

CampusCare is **not** a diagnostic tool. It avoids alarming or speculative results and keeps communication safe and concise.

---

## 🧠 Techniques Implemented & System Design

### Technique 1 — Supervised Fine-Tuning (SFT)
SFT teaches the model to:
- Associate symptoms with a *single* condition  
- Provide structured guidance  
- Output “Unknown” when unsure  
- Avoid hallucinating untrained diseases  

**Training Data:**  
`dataset_sft.csv` — rows of symptom → condition/guidance examples  
~10 examples per condition ensure variation (e.g., flu with or without fever).

**Model Architecture:**  
- Base: Llama-3.2-3B-Instruct  
- Framework: Unsloth for efficient training  
- LoRA (r=8, alpha=16) after testing to avoid overfitting or repetition  

---

### Technique 2 — Direct Preference Optimization (DPO)
DPO refines **behavior**, not core knowledge.  
It optimizes for:
- More empathetic tone  
- Shorter, calmer guidance  
- No extra disclaimers  
- No lists  
- Always one condition  
- No hallucinations  
- Consistency across phrasing variations  

**Training Data:**  
`dataset_dpo.jsonl` — each row contains:
- prompt  
- chosen output  
- rejected output  

**Key Hyperparameters:**  
- **β = 0.1** (reduces oversteering; stabilizes training)  
- Conservative learning rate and epoch count  

---

## 📈 Quantitative Evaluation (KPIs)

### Technique 1 — SFT
| Metric | Baseline (Llama-3B) | After SFT |
|--------|----------------------|------------|
| **Condition Accuracy** | 30% | **100%** |
| **Format Adherence** | 10% | **100%** |

### Technique 2 — DPO
| Metric | Baseline (SFT) | After DPO |
|--------|-----------------|------------|
| **Preference Win Rate** | 33% | **83%** |

DPO delivered clearer, calmer, and more consistent responses in nearly all cases.

### 💬 Example Model Outputs

```
Input:

I have a fever, sore throat, and body aches.


Output:

Condition: Flu
Guidance: The symptoms are commonly associated with flu. Next steps are to rest, drink plenty of fluids, and manage fever/body aches with OTC medications. High-risk individuals should consider contacting a provider regarding antivirals. Safety note: This is not a diagnosis. Please consult a healthcare professional if symptoms worsen or persist.
```

```
Input:

My eyes are itchy and my nose won’t stop running.


Output:

Condition: Allergies
Guidance: It's possible that your symptoms are related to allergies. Try avoiding known triggers like pollen, dust, or certain foods, and take an antihistamine to see if it relieves your symptoms. Stay hydrated by drinking plenty of water. Safety note: This is not a diagnosis. Please consult a healthcare professional if symptoms worsen or persist.
```

---

## 🧩 Repository Structure

```
├── campus_care_ai.ipynb         # Full SFT → restart → DPO → evaluation → UI pipeline
├── dataset_sft.csv               # SFT training dataset
├── dataset_dpo.jsonl             # DPO preference dataset
├── README.md                     # Documentation
└── LICENSE                       # MIT License
```

---

## 🔁 Reproducing the Full Pipeline (Deterministic)

### **Step 0 – Install Dependencies**
```bash
pip install unsloth accelerate transformers trl datasets gradio
```

---

### **Step 1 – Run Supervised Fine-Tuning (SFT)**  
Open `campus_care_ai.ipynb` and run the section titled:

**Technique 1: Supervised Fine-Tuning (SFT)**

**SFT results are shown Figure 3 of the report


This section trains the LoRA adapters and saves the model to:

```
model-medical-sft-final/
```

---

### **Step 2 – REQUIRED Runtime Reset**
You MUST restart before beginning DPO. Do this by running the section titled:

**Transition Block**

```python
import os
os.kill(os.getpid(), 9)   # <-- HARD RESTART (safe for Colab)
```

---

### **Step 3 – Run Direct Preference Optimization (DPO)**  
Now run the section titled:

**Technique 2: Direct Preference Optimization (DPO)**

**DPO results are shown Figure 4 of the report


This section produces the final aligned model:

```
symptom_sft_dpo/
```
---

### **Step 4 – Evaluate the Final Model**  
Run the evaluation section:

**DPO Evaluation (Strict Format Tests)**

This reproduces the metrics and all example outputs shown in the report.

---

### **Step 5 – Launch the Gradio App**
```python
demo.launch()
```

This opens the CampusCare AI interface.

---
## CampusCare AI was developed by Ryan Songvilay and Ruthie Bai
Contributions were shared across:
- Dataset creation (SFT and DPO)
- Model fine-tuning, debugging, and evaluation
= System design and pipeline structuring
- Writing analysis, KPIs, and report documentation
- Gradio UI development and deployment

