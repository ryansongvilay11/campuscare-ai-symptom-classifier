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
- **Unknown** (catch-all for vague, irrelevant, and out-of-scope cases)

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

**SFT results can be found in section "SFT Evaluation" and are shown as Figure 3 in the report**

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

**DPO results can be found in section "DPO Evaluation" and are shown as Figure 4 in the report**

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
├── CampusCare_AI.ipynb         # Full SFT → restart → DPO → evaluation → UI pipeline
├── dataset_sft.csv               # SFT training dataset
├── dataset_dpo.jsonl             # DPO preference dataset
├── README.md                     # Documentation
└── LICENSE                       # MIT License
```

---

## 🔁 Reproducing the Full Pipeline (Deterministic)
Before running the full script, make sure you have both training datasets uploaded into your local Colab folder

### **Step 0 – Install Dependencies**
```bash
pip install unsloth accelerate transformers trl datasets gradio
```

---

### **Step 1 – Run Supervised Fine-Tuning (SFT)**  
Open `CampusCare_AI.ipynb` and run the section titled:

**Technique 1: Supervised Fine-Tuning (SFT)**

This section:
- Loads Llama-3.2–3B-Instruct
- Applies LoRA adapters
- Loads dataset_sft.csv
- Trains the model to map symptoms → one correct condition
- Teaches the structured 3-line output format
- Saves the trained LoRA model to:

```
model-medical-sft-final/
```

**SFT results can be found in section "SFT Evaluation" and are shown as Figure 3 in the report**

---

### **Step 2 – REQUIRED Runtime Reset**
Before running DPO, you must restart the Colab runtime.
This is required because Unsloth monkey-patches TRL’s trainer.

**Run the Transition Block**

```python
import os
os.kill(os.getpid(), 9)   # <-- HARD RESTART (safe for Colab)
```

After the restart finishes, scroll down to Technique 2 and continue.

---

### **Step 3 – Run Direct Preference Optimization (DPO)**  
Now run the section titled:

**Technique 2: Direct Preference Optimization (DPO)**



This stage:
- Reloads the SFT model
- Loads dataset_dpo.jsonl
- Optimizes tone, empathy, format discipline, and safety
- Removes extra disclaimers
- Discourages hallucination and multi-condition outputs
- Saves the final aligned symptom classifier to:

```
symptom_sft_dpo/
```

**DPO results can be found in section "DPO Evaluation" and are shown as Figure 4 in the report**

---

### **Step 4 – Evaluate the Final Model**  
Run the evaluation section titled:

**DPO Evaluation**

This section reproduces every output shown in the report:
- Condition classification consistency
- “Unknown” handling for vague inputs
- Strict 3-line formatting
- Behavior improvements from DPO
- Sample comparisons vs. SFT baseline
  
---

### **Step 5 – Launch the Gradio App**
```python
demo.launch()
```

This opens the CampusCare AI web application, where users can:
- Type symptoms
- Receive safe, concise, aligned guidance
- View model output in the final production UI

---
## CampusCare AI was developed by Ryan Songvilay and Ruthie Bai
Contributions were shared across:
- Dataset creation (SFT and DPO)
- Model fine-tuning, debugging, and evaluation
= System design and pipeline structuring
- Writing analysis, KPIs, and report documentation
- Gradio UI development and deployment

