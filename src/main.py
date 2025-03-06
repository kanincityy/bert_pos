import torch
from transformers import AutoTokenizer, AdamW, BertForTokenClassification

from dataset import BertPOSDataset
from utils import read_conllu_file
from trainer import Trainer  # Import the Trainer class
from model import BertPOSModel  # Import the new model class

def main():
    # 1. Load Data and Create Label Index
    train_data_sentences, unique_labels = read_conllu_file("data/en_ewt-ud-train.conllu")
    label2index = {label: i for i, label in enumerate(unique_labels)}
    num_labels = len(label2index)

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    # 3. Prepare Datasets
    max_length = 512
    train_dataset = BertPOSDataset(train_data_sentences, tokenizer, label2index, max_length)
    dev_data_sentences, _ = read_conllu_file("data/en_ewt-ud-dev.conllu")
    dev_dataset = BertPOSDataset(dev_data_sentences, tokenizer, label2index, max_length)

    # 4. Initialize Model, Optimizer, and Trainer
    model = BertPOSModel(num_labels) #use the new model class.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    trainer = Trainer(model, train_dataset, dev_dataset, optimizer, device, batch_size=32, epochs=2)

    # 5. Train the Model
    trainer.train()

if __name__ == "__main__":
    main()
