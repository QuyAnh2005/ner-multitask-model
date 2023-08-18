import json
import os
import argparse

import numpy as np

PAD = "PAD"
PAD_token = 0


def read_file(file_path, split_line='\n', split_tag=','):
    sentences, pos_sentences, ner_sentences = [], [], []
    with open(file_path) as file_obj:
        data = file_obj.readlines()

    sentence, pos_sentence, ner_sentence = [], [], []
    for line in data:
        # End a sentence
        if line.startswith(split_line):
            sentences.append(sentence)
            pos_sentences.append(pos_sentence)
            ner_sentences.append(ner_sentence)

            # Reset
            sentence, pos_sentence, ner_sentence = [], [], []
        else:
            sentence.append(f"{split_tag}".join(line.split(split_tag)[:-2]))
            pos_sentence.append(line.split(split_tag)[-2])
            ner_sentence.append(line.split(split_tag)[-1].strip())

    return sentences, pos_sentences, ner_sentences


def _tag_idx(ner_sentences, save=True, file_name=None):
    tag_vocab = set()
    for tag in ner_sentences:
        tag_vocab.update(tag)

    # Sort the sets to ensure consistent ordering
    tag_vocab = sorted(list(tag_vocab))
    # Add a special character for vocab
    tag_vocab.insert(PAD_token, PAD)

    # Create dictionaries to map tag to integers
    tag2idx = {tag: idx for idx, tag in enumerate(tag_vocab)}
    idx2tag = {idx: tag for idx, tag in enumerate(tag_vocab)}

    if save and file_name:
        with open(file_name, 'w') as f:
            json.dump(tag2idx, f)
            f.close()

    return tag2idx, idx2tag


def pad_sequences(sequences, maxlen, vocab, padding=True, truncation=True):
    pad_seqs = []
    for seq in sequences:
        tokenized_seq = [vocab.get(w) for w in seq]
        len_seq = len(tokenized_seq)
        if padding and len_seq < maxlen:
            pad_seq = tokenized_seq + [vocab.get(PAD)] * (maxlen - len_seq)
        if truncation and len_seq > maxlen:
            pad_seq = tokenized_seq[:maxlen]
        pad_seqs.append(pad_seq)
    return pad_seqs


def load_from_path(path):
    train_inputs = np.load(f"{path}/train_inputs.npy")
    val_inputs = np.load(f"{path}/val_inputs.npy")
    train_masks = np.load(f"{path}/train_masks.npy")
    val_masks = np.load(f"{path}/val_masks.npy")
    train_poss = np.load(f"{path}/train_poss.npy")
    val_poss = np.load(f"{path}/val_poss.npy")
    train_ners = np.load(f"{path}/train_ners.npy")
    val_ners = np.load(f"{path}/val_ners.npy")
    return train_inputs, val_inputs, train_masks, val_masks, train_poss, val_poss, train_ners, val_ners


def get_number_vocab(filename):
    """filename is vocabulary including keys and values."""
    with open(filename, "r") as json_file:
        vocab_dict = json.load(json_file)
    return len(vocab_dict)

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, default="pos-ner",
                        help='Model name')

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir

    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams
