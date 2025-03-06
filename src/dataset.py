import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

# Define Dataset class
class BertPOSDataset(Dataset):
  # Initialise Dataset class with self, sentences (data), tokenizer (from HF),
  # tag2index, and max_length (512) to trunkate sentences
    def __init__(self, sentences, tokenizer, tag2index, max_length = 512):
        """
        Args:
            sentences: List of sentences, where each sentence is a list of (token, tag) tuples.
            tokenizer: A Hugging Face tokenizer.
            tag2index: Dictionary mapping tag names to indices.
            max_length: Maximum sequence length for tokenized inputs.
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.tag2index = tag2index
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        # By convention, a label id of -100 will be ignored when computing the loss
        self.pad_label_id = -100

    # Return length of sentence input
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # This is a shorthand for creating two lists
        # of tokens and tags for all the tuples in a
        # specific sentence
        tokens, tags = zip(*self.sentences[idx])


        # Call the tokeniser to transform tokens into the
        # BERT-tokenised input:
        tokenized_input = tokenizer(tokens,is_split_into_words=True,
                                    padding='max_length', truncation=True,
                                    max_length=self.max_length,
                                    return_tensors='pt')

        # Extract the token indices
        # squeeze returns a tensor of x dimension
        input_ids = tokenized_input['input_ids'].squeeze(0)
        # Extract the attention mask
        attention_mask = tokenized_input['attention_mask'].squeeze(0)

        # The rest of the function aligns the labels to the tokenised sequence
        word_ids = tokenized_input.word_ids()  # Maps subwords to their original words
        labels = [self.pad_label_id] * len(word_ids)

        prev_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != prev_word_idx:  # Only label the first subword of a token
                labels[i] = self.tag2index[tags[word_idx]]
            prev_word_idx = word_idx

        labels = labels + [self.pad_label_id] * (self.max_length - len(labels))  # Ensure consistent length
        # truncate the length of the list of labels
        labels = torch.LongTensor(labels[:self.max_length])

        # return a dictionary of input_ids, attention_mask, and labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels}
    