import re
import os
import random
from collections import Counter

import time
from datetime import datetime
import pytz

import numpy as np
import pandas as pd
import string
import torch

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from transformers import BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup

from nltk.tokenize import word_tokenize

from modules.data_utils import NerShopeeDataset, NerDataLoader
from modules.metrics import ner_metrics_fn
from modules.forward_fn import forward_word_classification
from modules.word_classification import BertForWordClassification

# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory() # create stemmer
stemmer = factory.create_stemmer()


def customTime(*args):
    """
    custom time zone for logger
    """
    utc_dt = pytz.utc.localize(datetime.utcnow())
    converted = utc_dt.astimezone(pytz.timezone("Singapore"))
    return converted.timetuple()

def stem_texts(sentence) :
    """
    Stemming process using Sastrawi
    """
    return stemmer.stem(sentence)

def remove_hex(text):
    """
    Remove Hex
    Example: 
    "\xe3\x80\x90Ramadan\xe3\x80\x91Dompet wanita multi-fungsi gesper dompet multi-card"
    """
    res = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i+1 < len(text) and text[i+1] == "x":
            i += 3
            res.append(" ")
        else:
            res.append(text[i])
        i += 1
    # text = text.encode('utf-8')
    # text = text.encode('ascii', 'ignore')
    # text = text.encode('ascii', errors='ignore')
    # text = unicode(text)
    # text = re.sub(r'[^\x00-\x7f]', r'', text)
    # filter(lambda x: x in printable, text)
    return "".join(res)

def remove_punctuation(text):
    """
    Removing punctuations
    """
    return re.sub(r'[^\w\s]', r' ', text)

def remove_space_between_quantity(text):
    """
    200 ml -> 200ml
    3 kg -> 3kg
    200 x 200 -> 200x200
    3 in 1 -> 3in1
    Example: "Double Tape DOUBLE FOAM TAPE 55 mm 45 m 45 makan   2000 x 2000 scs"
    """
    text = re.sub(r"([1-9][0-9]*)(in|inch)( |$)", r'\1inch ', text)
    text = re.sub(r"([1-9][0-9]*)(m|meter)( |$)", r'\1m ', text)
    text = re.sub(r"([1-9][0-9]*)(mm|milimeter)( |$)", r'\1mm ', text)
    text = re.sub(r"([1-9][0-9]*)(cm|centimeter)( |$)", r'\1ccm ', text)
    text = re.sub(r"([1-9][0-9]*)(pc|pcs|potong)( |$)", r'\1pcs ', text)
    text = re.sub(r"([1-9][0-9]*)(y|year|thn|tahun)( |$)", r'\1tahun ', text)
    text = re.sub(r"([1-9][0-9]*)(k|kilo|kg|kilogram)( |$)", r'\1kg ', text)
    text = re.sub(r"([1-9][0-9]*)(g|gr|gram)( |$)", r'\1gr ', text)
    text = re.sub(r"([1-9][0-9]*)(l|liter)( |$)", r'\1l ', text)
    text = re.sub(r"([1-9][0-9]*)(ml|mililiter)( |$)", r'\1ml ', text)
    text = re.sub(r"([1-9][0-9]*) (in|inch)( |$)", r'\1inch ', text)
    text = re.sub(r"([1-9][0-9]*) (m|meter)( |$)", r'\1m ', text)
    text = re.sub(r"([1-9][0-9]*) (mm|milimeter)( |$)", r'\1mm ', text)
    text = re.sub(r"([1-9][0-9]*) (cm|centimeter)( |$)", r'\1ccm ', text)
    text = re.sub(r"([1-9][0-9]*) (pc|pcs|potong)( |$)", r'\1pcs ', text)
    text = re.sub(r"([1-9][0-9]*) (y|year|thn|tahun)( |$)", r'\1tahun ', text)
    text = re.sub(r"([1-9][0-9]*) (k|kilo|kg|kilogram)( |$)", r'\1kg ', text)
    text = re.sub(r"([1-9][0-9]*) (g|gr|gram)( |$)", r'\1gr ', text)
    text = re.sub(r"([1-9][0-9]*) (l|liter)( |$)", r'\1l ', text)
    text = re.sub(r"([1-9][0-9]*) (ml|mililiter)( |$)", r'\1ml ', text)
    text = re.sub(r"([1-9][0-9]*) (yard|set|lembar|tablet|kaplet|buah|box|sachet|pasang|gb|watt)( |$)", r'\1\2 ', text)
    
    text = re.sub(r"([1-9][0-9]*) (x) ([1-9][0-9]*)", r'\1x\3', text)
    text = re.sub(r"([1-9][0-9]*) (in) ([1-9][0-9]*)", r'\1in\3', text)
    return text

def remove_multiple_whitespace(text):
    """
    remove multiple whitespace
    it covers tabs and newlines also
    """
    return re.sub(' +', ' ', text.replace('\n', ' ').replace('\t', ' ')).strip()


def clean_text(sentence):
    """
    clean raw text form shopee dataset
    """
    text = str(sentence)
    text = text.lower()
    text = remove_hex(text)
    text = remove_punctuation(text)
    # text = stem_texts(text)
    text = remove_multiple_whitespace(text)
    text = remove_space_between_quantity(text)
    words = text.split(" ")
    res = []
    for w in words:
        if w not in res:
            res.append(w)
    return " ".join(res)


def translate_english_to_indonesia(sentences, batch_size=128):
    """
    Translate text from English into Indonesia using a Transformer
    https://huggingface.co/Helsinki-NLP/opus-mt-en-id/tree/main
    """
    tokenizer = AutoTokenizer.from_pretrained("../input/en-id-converter")
    txt_model = AutoModelForSeq2SeqLM.from_pretrained("../input/en-id-converter")

    txt_model.cuda()
    txt_model.eval()

    trans_texts = []
    CHUNK = batch_size

    # print('translating texts')
    CTS = len(sentences)//CHUNK
    if len(sentences) % CHUNK != 0:
        CTS += 1
    for i, j in tqdm(enumerate(range(CTS)), total=CTS):
        a = j*CHUNK
        b = (j+1)*CHUNK
        b = min(b, len(sentences))
        input_ids = tokenizer(
            sentences[a:b], return_tensors="pt", truncation=True, padding=True).input_ids.cuda()
        outputs = txt_model.generate(
            input_ids=input_ids, num_return_sequences=1)
        val = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        trans_texts.extend(val)
    return trans_texts


def create_bert_label_for_prediction(text):
    """
    Create label formating to be consumed by BERT
    """
    s = text.lower()
    tokens = s.split(" ")
    n = len(tokens)
    tags = []
    i = 0
    while i < n:
        tags.append("O")
        i += 1

    if tokens[-1] != ".":
        tokens.append(".")
        tags.append("O")

    res = ""

    for token, tag in zip(tokens, tags):
        res += "{}\t{}\n".format(token, tag)
    return res


def save_file(filename, arr):
    f = open(filename, "w+",  encoding='utf-8')
    for s in arr:
        f.write(s.encode("ascii", "ignore").decode())
        f.write("\n")
    f.close()

def extract_product_features(model_name, pretrained_model_path, dataset_path, batch_size=256, max_seq_len=64):

    if not dataset_path: 
        raise ValueError("Dataset path can't empty")

    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    config.num_labels = NerShopeeDataset.NUM_LABELS

    w2i, i2w = NerShopeeDataset.LABEL2INDEX, NerShopeeDataset.INDEX2LABEL

    # Instantiate model
    model = BertForWordClassification.from_pretrained(
        model_name, config=config)

    checkpoint = torch.load(pretrained_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.cuda()

    # Evaluate on validation
    model.eval()
    torch.set_grad_enabled(False)

    test_dataset = NerShopeeDataset(
        dataset_path, tokenizer, lowercase=True)

    test_loader = NerDataLoader(dataset=test_dataset, max_seq_len=max_seq_len,
                                batch_size=batch_size, num_workers=16, shuffle=False)
    

    list_hyp = []
    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        _, batch_hyp, _ = forward_word_classification(
            model, batch_data[:-1], i2w=i2w, device='cuda')
        list_hyp += batch_hyp

    return list_hyp


def reformat_text_feature(text, bert_label):
    if text == "":
        return ""

    text = text.split(" ")

    contexts = ["BRAND", "NAME", "TYPE", "COLOR", "SIZE", "GENDER",
                "AGE", "MATERIAL", "THEME", "MASS", "DIMENSION", "QUANTITY"]

    features = {}
    for c in contexts:
        features[c] = []

    if len(text) != len(bert_label):
        print(text)
        print(bert_label)
        assert False
    i = 0
    while i < len(text):
        for c in contexts:
            if c in bert_label[i] and text[i] != "":
                features[c].append(text[i])
                break
        i += 1

    for c in contexts:
        features[c] = sorted(features[c])

    return features


def extract_title(text, bert_label):
    if text == "":
        return ""

    text = text.split(" ")

    contexts = ["BRAND", "NAME", "TYPE", "COLOR", "SIZE", "GENDER",
                "AGE", "MATERIAL", "THEME", "MASS", "DIMENSION", "QUANTITY"]

    tokens = {}
    for c in contexts:
        tokens[c] = ""

    if len(text) != len(bert_label):
        print(text)
        print(bert_label)
        assert False
    i = 0
    while i < len(text):
        is_context = False
        for c in contexts:
            if c in bert_label[i]:
                is_context = True
                tokens[c] += text[i] + " "
                i += 1
                while i < len(text) and c in bert_label[i]:
                    tokens[c] += text[i] + " "
                    i += 1
                tokens[c] = tokens[c][:]
                break
        if not is_context:
            i += 1

    res = ""
    for c in contexts:
        res += tokens[c] + " "
    res = remove_multiple_whitespace(res)
    if res == "":
        return " ".join(text)
    return res


def drop_last(arr):
    """
    drop the last label because we add dot "." in the last
    """
    return arr[:-1]

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
