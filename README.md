# Figurative Language Recognition System (NLP Assignment)

This repository contains my submission for **NLP: CS60075 (Autumn Semester 2025, IIT Kharagpur)**.

## 📂 Files
- `NLP_Assignment_1_21CS10072.ipynb` → Main Jupyter Notebook (contains code, results, and explanations)
- `NLP_Assignment_1_21CS10072.pdf` → Report submission (summarized observations)

## 📊 Tasks
1. **POS Tagger Implementation (from scratch)**
   - Trained using the Treebank corpus from NLTK.
   - Implemented using the Viterbi Algorithm.

2. **Vanilla Figurative Language Recognizer**
   - Dataset: [V-FLUTE](https://huggingface.co/datasets/ColumbiaNLP/V-FLUTE) (via Hugging Face `datasets` library).
   - Classifier trained on sentence embeddings.

3. **Improved Figurative Language Recognizer**
   - Integrated POS tag features into sentence embeddings.
   - Compared performance against the vanilla recognizer.

4. **Report**
   - Findings and observations are compiled into the report (PDF).


## ⚙️ Installation
Clone this repository and install dependencies:

```bash
git clone https://github.com/tohutosema77/Figurative-Language-Recognition-System-NLP.git
cd Figurative-Language-Recognition-System-NLP
pip install -r requirements.txt

📚 Dataset Usage
1. NLTK Treebank Corpus (for POS Tagging)
import nltk
nltk.download('treebank')
from nltk.corpus import treebank

2. V-FLUTE Corpus (for Figurative Language Recognition)
from datasets import load_dataset
dataset = load_dataset("ColumbiaNLP/V-FLUTE")

🚀 Running the Notebook

Start Jupyter and open the notebook:

jupyter notebook Figurative_Language_Recognition.ipynb


Run the cells step by step to train models and view results.


---

## 📄 `requirements.txt`

Save this as **requirements.txt** in your repo root:



nltk
scikit-learn
datasets
huggingface-hub
numpy
pandas


---

✅ With this setup:  
- **README.md** → clear documentation for humans.  
- **requirements.txt** → exact dependencies for pip.  

---

Do you also want me to prepare a **starter notebook template (`Figurative_Language_Recognition.ipyn