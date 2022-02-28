import os
import sys
import logging
import time
from datetime import datetime
import pytz

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm

from transformers import BertConfig, BertTokenizer
from nltk.tokenize import word_tokenize

import seqeval
import seqeval.metrics as seqeval_metrics

from modules.word_classification import BertForWordClassification
from modules.forward_fn import forward_word_classification
from modules.metrics import ner_metrics_fn
from modules.data_utils import NerShopeeDataset, NerDataLoader
from modules.utils import set_seed, customTime


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

def save(fpath, model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, fpath)


if __name__ == "__main__":

    # Set random seed
    set_seed(26092020)

    model_version = "large"
    use_regularization = True
    save_freq = 5

    model_dir = "models/bert-{}/".format(model_version)

    batch_size = 20
    eval_batch_size = 32
    max_seq_len = 128
    learning_rate = 5e-5

    model_dir = "{}{}_{}_{}".format(
        model_dir, batch_size, max_seq_len, learning_rate)
    if use_regularization:
        model_dir += "_regularization"

    model_dir += "/"

    # Train
    n_epochs = 30

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(model_dir, 'output.log')),
            logging.StreamHandler()
        ])
    logging.Formatter.converter = customTime

    logger.info("Model: indobert-{}".format(model_version))
    logger.info("Save model checkpoint every {} epoch".format(save_freq))

    model_name = "indobenchmark/indobert-{}-p1".format(model_version)

    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = NerShopeeDataset.NUM_LABELS
    

    # Instantiate model
    model = BertForWordClassification.from_pretrained(
        model_name, config=config)

    train_dataset_path = 'data/ecommerce-train.txt'
    train_dataset = NerShopeeDataset(
        train_dataset_path, tokenizer, lowercase=True)
    train_loader = NerDataLoader(dataset=train_dataset, max_seq_len=max_seq_len,
                                 batch_size=batch_size, num_workers=16, shuffle=True)
    
    valid_dataset_path = 'data/ecommerce-test.txt'
    valid_dataset = NerShopeeDataset(
        valid_dataset_path, tokenizer, lowercase=True)
    valid_loader = NerDataLoader(dataset=valid_dataset, max_seq_len=max_seq_len,
                                 batch_size=eval_batch_size, num_workers=16, shuffle=False)

    w2i, i2w = NerShopeeDataset.LABEL2INDEX, NerShopeeDataset.INDEX2LABEL

    optimizer = None
    if use_regularization:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.1},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters, lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.cuda()

    logger.info("Model: indobert-{}".format(model_version))
    logger.info("Batch Size: {}".format(batch_size))
    logger.info("Max Seq Length: {}".format(max_seq_len))
    logger.info("Learning Rate: {}".format(learning_rate))
    logger.info("Epochs: {}".format(n_epochs))

    min_loss = sys.maxsize
    max_f1 = 0
    for epoch in range(n_epochs):
        if epoch == 10:
            for g in optimizer.param_groups:
                g['lr'] = 3e-5
        elif epoch == 15:
            for g in optimizer.param_groups:
                g['lr'] = 2e-5
        elif epoch ==19:
            for g in optimizer.param_groups:
                g['lr'] = 1e-5
        elif epoch ==23:
            for g in optimizer.param_groups:
                g['lr'] = 5e-6

        model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward model
            loss, batch_hyp, batch_label = forward_word_classification(
                model, batch_data[:-1], i2w=i2w, device='cuda')

            tr_loss = loss.item()

            if np.isnan(tr_loss):
                break
            else:
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss = total_train_loss + tr_loss

                # Calculate metrics
                list_hyp += batch_hyp
                list_label += batch_label

                train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                                                                                           total_train_loss/(i+1), get_lr(optimizer)))

        # Calculate train metric
        metrics = ner_metrics_fn(list_hyp, list_label)
        logger.info("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch+1),
                                                                       total_train_loss/(i+1), metrics_to_string(metrics), get_lr(optimizer)))

        # save a checkpoint every 5 epoch
        # or save checkpoints for the last 3 models
        # if (epoch > 0 and epoch % save_freq == 0) or epoch >= max(n_epochs - 3, 0):
        #     model_path = "{}model-{}.pth".format(model_dir, epoch+1)
        #     save(model_path, model, optimizer)
        #     logger.info("save model checkpoint at {}".format(model_path))

        # Evaluate on validation
        model.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]
            loss, batch_hyp, batch_label = forward_word_classification(
                model, batch_data[:-1], i2w=i2w, device='cuda')

            # Calculate total loss
            valid_loss = loss.item()
            total_loss = total_loss + valid_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            metrics = ner_metrics_fn(list_hyp, list_label)

            pbar.set_description("VALID LOSS:{:.4f} {}".format(
                total_loss/(i+1), metrics_to_string(metrics)))

        metrics = ner_metrics_fn(list_hyp, list_label)
        logger.info("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch+1),
                                                             total_loss/(i+1), metrics_to_string(metrics)))

        if total_loss/(i+1) < min_loss and metrics["F1"] >= max_f1:
            min_loss = total_loss/(i+1)
            max_f1 = metrics["F1"]

            model_path = "{}model-best.pth".format(model_dir)
            logger.info("save model checkpoint at {}".format(model_path))
            save(model_path, model, optimizer)
            # https://github.com/huggingface/transformers/issues/7849

            score = seqeval_metrics.f1_score(list_label, list_hyp)
            print(' - f1: {:04.2f}'.format(score * 100))
            print(seqeval_metrics.classification_report(list_label, list_hyp, digits=4))
