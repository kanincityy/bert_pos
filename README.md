# BERT Part-of-Speech Tagging

This project implements a **part-of-speech (POS) tagging model** that utilizes a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model. It processes sentence data in **CONLL-U format** and predicts syntactic dependencies between words in a sentence.

## Project Overview

The model is designed to accurately assign POS tags to words in a sentence. It leverages the power of pretrained BERT embeddings and fine-tuning for effective training. The trained model can be used to predict POS tags for new sentences.

## Files and Structure

- `data/`: Contains the training (`train.conllu`) and development (`dev.conllu`) datasets in CONLL-U format.
- `models/`: Stores the saved trained model (`pos_tagging_model.pth`).
- `src/`: Contains the Python source code.
    - `dataset.py`: Defines the `BertPOSDataset` for data loading and tokenization.
    - `model.py`: Defines the `BertPOSModel` using `BertForTokenClassification`.
    - `trainer.py`: Implements the `Trainer` class for model training and evaluation.
    - `utils.py`: Contains utility functions, such as `read_conllu_file`.
    - `main.py`: The core script that orchestrates the training and evaluation process.
- `requirements.txt`: Lists the project's dependencies.
- `README.md`: This file, providing project information.
- `.gitignore`: Specifies files to be ignored by Git.

## Features

- Uses a pretrained BERT model fine-tuned for POS tagging.
- Processes data in CONLL-U format.
- Implements a custom `BertPOSDataset` for efficient data loading and tokenization.
- Uses a `Trainer` class for modular training and evaluation.
- Saves the trained model for future use.

## Requirements

- `torch`
- `transformers`
- `tqdm`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Run
If you'd like to try out my projects, follow these steps:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/kanincityy/bert_pos.git
    ```
2. Navigate to the specific project's directory:
    ```bash
    cd <bert_pos>
    ```
3. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Place your CoNLL-U data: Ensure train.conllu and dev.conllu are in the data/ directory.
6. Run the training script:
    ```bash
    python src/main.py
    ```
    This will train the BERT POS tagging model and save it to models/pos_tagging_model.pth.

## Contributing 

This project is a reflection of my learning, but feel free to fork the repository and contribute if you have ideas or improvements!

## License 

This repository is licensed under the MIT License. See the LICENSE file for details.

---

Happy coding! ✨🐇