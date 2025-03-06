import torch
from torch.utils.data import DataLoader
import tqdm
import os

class Trainer:
    def __init__(self, model, train_dataset, dev_dataset, optimizer, device, batch_size=32, epochs=4, save_path="models/pos_tagging_model.pth"):
        self.model = model.to(device)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.save_path = save_path #add save path

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            epoch_losses = []
            for batch in tqdm.tqdm(self.train_dataloader):
                for x in batch:
                    batch[x] = batch[x].to(self.device)

                self.optimizer.zero_grad()
                output = self.model(**batch)
                loss = output.loss
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

                if len(epoch_losses) % 1000 == 0:
                    print(f"\nEpoch: {epoch}, current loss: {sum(epoch_losses)/len(epoch_losses)}")

            print(f"\nEpoch: {epoch}, final loss: {sum(epoch_losses)/len(epoch_losses)}")
            self.evaluate()

        # Save the model after training is complete.
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True) #make sure the directory exists.
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to: {self.save_path}")

    def evaluate(self):
        self.model.eval()
        preds = []
        true_labels = []

        with torch.no_grad():
            for batch in self.dev_dataloader:
                input_data = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                labels = batch["labels"].flatten()

                output = self.model(**input_data)
                y_hat = output.logits.cpu().topk(1, dim=-1).indices.flatten()

                true_labels.extend(labels.tolist())
                preds.extend(y_hat.tolist())

        correct = 0
        n = 0
        for i in range(len(true_labels)):
            if true_labels[i] == self.dev_dataloader.dataset.pad_label_id:
                continue
            n += 1
            if true_labels[i] == preds[i]:
                correct += 1

        print(f"Dev Accuracy: {correct / n * 100:.2f}%")