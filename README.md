# Fine-Tuning BERT for Part-of-Speech Tagging (CONLL-U)

This project implements a **Part-of-Speech (POS) tagging model** using `bert-base-uncased`, fine-tuned for **token-level classification** on linguistic data in **CONLL-U format**.

Built using **PyTorch** and **Hugging Face Transformers**, the system aligns tokens and labels with custom dataset loaders and training logic, enabling robust experimentation on POS tagging tasks.

> Project created as part of my MSc coursework in Computational Linguistics @ UCL.

---

## Objective

To build a reusable and interpretable token classification pipeline using a pre-trained Transformer for **grammatical structure prediction** — a foundational task in NLP pipelines, especially for **downstream parsing, NER, and language modeling**.

---

## 🛠Technologies Used

| Category        | Libraries/Tools                         |
|------------------|------------------------------------------|
| Language         | Python 3.7+  
| Core Frameworks  | PyTorch, Hugging Face `transformers`, `datasets`  
| Utilities        | `tqdm`, `numpy`, `scikit-learn` (optional)  
| Data Format      | CONLL-U (Universal Dependencies)

---

## Key Features

- Fine-tunes `bert-base-uncased` using `BertForTokenClassification`  
- Processes **CONLL-U** data with token/label alignment logic  
- Implements modular custom classes:
  - `BertPOSDataset` – loads and aligns tokenized data with POS tags
  - `BertPOSModel` – wraps the BERT model for token classification
  - `Trainer` – handles training, evaluation, and metrics
  - `utils.py` – parses CONLL-U and supports I/O operations  
- Saves model checkpoints for reuse or transfer learning

---

## Project Structure

```
.
├── data/               # Contains CONLL-U format data files (e.g., train.conllu, dev.conllu)
├── models/             # Stores the saved trained model checkpoint (output of main.py)
├── src/                # Source code directory
│   ├── dataset.py      # Custom PyTorch Dataset class for CONLL-U
│   ├── model.py        # Custom PyTorch BERT model class
│   ├── trainer.py      # Training and evaluation logic class
│   ├── utils.py        # Utility functions (e.g., CONLL-U reader)
│   └── main.py         # Main script to run training & evaluation
├── requirements.txt    # Project dependencies
├── .gitignore          # Git ignore file
├── LICENSE             # MIT License file
└── README.md           # This file
```

---

## How to Run

1. Clone the repo:
```
bash
git clone https://github.com/kanincityy/bert_pos.git
cd bert_pos
```
2. Set up your environment:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3. Prepare your your CONLL-U data:
- Place your `train.conllu` and `dev.conllu` in the `data/` directory
5. Run the training pipeline:
```
python src/main.py
```
This will:
- Load and tokenize data
- Fine-tune BERT
- Evaluate model during/after training
- Save model to `models/pos_tagging_model.pth`

---

## Why POS Tagging

While often considered a foundational task, POS tagging remains critical for:
- Syntactic analysis in low-resource or morphologically rich languages
- Preprocessing in NER, parsing, and rule-based NLP systems
- Evaluating how well Transformer models adapt to fine-grained token-level tasks
This project showcases my understanding of token classification, dataset design, and custom training workflows in PyTorch.

---

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Happy coding! ✨🐇
