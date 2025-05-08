# BERT Fine-Tuning for Part-of-Speech (POS) Tagging (PyTorch & Hugging Face)

This repository contains a project implementing Part-of-Speech (POS) tagging using a fine-tuned BERT model (`bert-base-uncased` or similar). The model processes sentences in **CONLL-U format** and assigns a grammatical tag (e.g., Noun, Verb, Adjective) to each token.

This project utilises PyTorch and Hugging Face libraries (`transformers`, `datasets`).

---

### Key Features & Implementation Details

*   **Task:** Part-of-Speech (POS) Tagging on the token level.
*   **Model:** Leverages a pre-trained BERT model fine-tuned specifically for token classification using `BertForTokenClassification` from the Hugging Face `transformers` library.
*   **Data Format:** Processes input data structured in the standard **CONLL-U format**.
*   **Custom Components:** Includes:
    *   `BertPOSDataset` (`src/dataset.py`): Custom PyTorch Dataset for handling CONLL-U data loading, tokenization alignment, and tag mapping.
    *   `BertPOSModel` (`src/model.py`): Defines the wrapper around the Hugging Face model.
    *   `Trainer` (`src/trainer.py`): Manages the training loop, validation, optimization, and evaluation.
*   **Utility Functions:** Provides helpers (`src/utils.py`) for reading and parsing CONLL-U files.

### 🛠️ Technologies Used

*   Python 3.7+
*   PyTorch
*   Hugging Face `transformers`
*   Tqdm (for progress bars)
*   *(Add any other specific libraries used, e.g., `numpy`)*

### Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kanincityy/bert_pos.git
    cd bert_pos
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```
3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Data:** Ensure your training (`train.conllu`) and development (`dev.conllu`) files are placed inside the `data/` directory.

### Usage

Run the main training script from the **root** directory of the project:

```bash
python src/main.py
```

This script will:

1.  Load and preprocess the CONLL-U data from the `data/` directory using `BertPOSDataset`.
2.  Load the pre-trained BERT tokenizer and the `BertPOSModel`.
3.  Instantiate the `Trainer`.
4.  Fine-tune the model on the training data (`train.conllu`).
5.  Evaluate the model on the development set (`dev.conllu`) during/after training.
6.  Save the final trained model weights to `models/pos_tagging_model.pth`.

### Project Structure

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

### License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Happy coding! ✨🐇
