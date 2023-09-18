import numpy as np

from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from utils import read_file, _tag_idx, pad_sequences, get_hparams

hps = get_hparams()
maxlen = hps.preprocessing.maxlen

# Get pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained(hps.base_model_name)
sentences, pos_sentences, ner_sentences = read_file(hps.preprocessing.data_path)

vocab_ner_path = f"{hps.preprocessing.out_path}/{hps.preprocessing.vocab_ner}"
vocab_pos_path = f"{hps.preprocessing.out_path}/{hps.preprocessing.vocab_pos}"
ner2idx, idx2ner = _tag_idx(ner_sentences, file_name=vocab_ner_path)
pos2idx, idx2pos = _tag_idx(pos_sentences, file_name=vocab_pos_path)

# update the number of ner tags and pos tags in parameters
hps.n_pos = len(pos2idx)
hps.n_ner = len(ner2idx)

tokenized_sentences = [
    tokenizer.encode(sen,
                     max_length=maxlen,
                     padding='max_length',
                     truncation=True,
                     add_special_tokens=True)
    for sen in sentences
]
attention_masks = [[int(idx > 0) for idx in seq] for seq in tokenized_sentences]
tokenized_poss = pad_sequences(pos_sentences, maxlen, pos2idx)
tokenized_ners = pad_sequences(ner_sentences, maxlen, ner2idx)

# Test with first 100 examples
train_inputs, val_inputs, train_masks, val_masks, train_poss, val_poss, train_ners, val_ners = train_test_split(
    tokenized_sentences[:100],
    attention_masks[:100],
    tokenized_poss[:100],
    tokenized_ners[:100],
    random_state=hps.preprocessing.random_state,
    test_size=hps.preprocessing.test_size
)

if __name__ == "__main__":
    out_path = hps.preprocessing.out_path
    np.save(f"{out_path}/train_inputs.npy", train_inputs)
    np.save(f"{out_path}/val_inputs.npy", val_inputs)

    np.save(f"{out_path}/train_masks.npy", train_masks)
    np.save(f"{out_path}/val_masks.npy", val_masks)

    np.save(f"{out_path}/train_poss.npy", train_poss)
    np.save(f"{out_path}/val_poss.npy", val_poss)

    np.save(f"{out_path}/train_ners.npy", train_ners)
    np.save(f"{out_path}/val_ners.npy", val_ners)



