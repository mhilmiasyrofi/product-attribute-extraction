import numpy as np
import pandas as pd
import string
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class NerShopeeDataset(Dataset):
    # Static constant variable
    LABELS = ['I-BRAND', 'I-NAME', 'I-TYPE', 'I-COLOR', 'I-SIZE', 'I-GENDER', 'I-AGE', 'I-MATERIAL', 'I-THEME', "I-MASS", "I-DIMENSION", "I-QUANTITY", 
              'B-BRAND', 'B-NAME', 'B-TYPE', 'B-COLOR', 'B-SIZE', 'B-GENDER', 'B-AGE', 'B-MATERIAL', 'B-THEME', "B-MASS", "B-DIMENSION", "B-QUANTITY",
            "MISC",'O']
    LABEL2INDEX = {}
    INDEX2LABEL = {}
    for i, l in enumerate(LABELS):
        LABEL2INDEX[l] = i
        INDEX2LABEL[i] = l

    NUM_LABELS = len(LABELS)

    def load_dataset(self, path):
        # Read file
        data = open(path, 'r').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                token, label = line[:-1].split('\t')
                sentence.append(token)
                if label not in self.LABELS :
                    seq_label.append(self.LABEL2INDEX["MISC"])
                else :
                    seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset

    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']

        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1]  # For CLS

        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(
                word, add_special_tokens=False)
            subword_to_word_indices += [
                word_idx for i in range(len(subword_list))]
            subwords += subword_list

        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]

        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']

    def __len__(self):
        return len(self.data)


class NerDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NerDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len

    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))

        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full(
            (batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full(
            (batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i, :len(subwords)] = subwords
            mask_batch[i, :len(subwords)] = 1
            subword_to_word_indices_batch[i, :len(
                subwords)] = subword_to_word_indices
            seq_label_batch[i, :len(seq_label)] = seq_label

            seq_list.append(raw_seq)

        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
