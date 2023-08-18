from torch.utils.data import Dataset

# Set up the data loader
class MultiDataset(Dataset):
    def __init__(
            self,
            input_ids,
            input_masks,
            pos_sequences,
            ner_sequences,
    ):
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.pos_sequences = pos_sequences
        self.ner_sequences = ner_sequences

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ = self.input_ids[idx]
        mask = self.input_masks[idx]
        pos_seq = self.pos_sequences[idx]
        ner_seq = self.ner_sequences[idx]
        return input_, mask, pos_seq, ner_seq
