import torch

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, NLLLoss
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from model import NeurondBERTMultitask
from dataloader import MultiDataset
from utils import load_from_path, get_hparams, get_number_vocab
from metrics import flat_accuracy, f1_score_seq


def train(model, train_dataloader, optimizer, scheduler, criterion, clip):
    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss, total_loss_ner, total_loss_pos = 0.0, 0.0, 0.0

    train_accuracy_ner, train_accuracy_pos = 0.0, 0.0
    train_f1_ner, train_f1_pos = 0.0, 0.0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_pos_labels, b_ner_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        output_ner, output_pos = model(b_input_ids, b_input_mask)
        loss_ner = criterion(output_ner.view(-1, n_ner), b_ner_labels.view(-1))
        loss_pos = criterion(output_pos.view(-1, n_pos), b_pos_labels.view(-1))

        # get the loss
        w_ner = n_ner
        w_pos = n_pos
        loss = w_ner / (w_ner + w_pos) * loss_ner + w_pos / (w_ner + w_pos) * loss_pos
        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # track train loss
        total_loss += loss.item()
        total_loss_ner += loss_ner.item()
        total_loss_pos += loss_pos.item()

        # Clip the norm of the gradient
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=clip)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

        # Metric
        train_accuracy_ner += flat_accuracy(output_ner, b_ner_labels)
        train_accuracy_pos += flat_accuracy(output_pos, b_pos_labels)

        train_f1_ner += f1_score_seq(output_ner, b_ner_labels)
        train_f1_pos += f1_score_seq(output_pos, b_pos_labels)

    # Calculate the average loss over the training data.
    nb_train = len(train_dataloader)
    avg_train_loss = total_loss / nb_train
    avg_train_loss_ner = total_loss_ner / nb_train
    avg_train_loss_pos = total_loss_pos / nb_train
    avg_train_accuracy_ner = train_accuracy_ner / nb_train
    avg_train_accuracy_pos = train_accuracy_pos / nb_train
    avg_train_f1_ner = train_f1_ner / nb_train
    avg_train_f1_pos = train_f1_pos / nb_train

    return avg_train_loss, avg_train_loss_ner, avg_train_loss_pos, \
           avg_train_accuracy_ner, avg_train_accuracy_pos, \
           avg_train_f1_ner, avg_train_f1_pos


def evaluate(model, valid_dataloader, criterion):
    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch
    eval_loss, eval_loss_ner, eval_loss_pos = 0.0, 0.0, 0.0
    eval_accuracy_ner, eval_accuracy_pos = 0.0, 0.0
    eval_f1_ner, eval_f1_pos = 0.0, 0.0
    for batch in valid_dataloader:
        b_input_ids, b_input_mask, b_pos_labels, b_ner_labels = batch
        with torch.no_grad():
            output_ner, output_pos = model(b_input_ids, b_input_mask)
        loss_ner = criterion(output_ner.view(-1, n_ner), b_ner_labels.view(-1))
        loss_pos = criterion(output_pos.view(-1, n_pos), b_pos_labels.view(-1))

        # get the loss
        loss = loss_ner + loss_pos
        eval_loss_ner += loss_ner.mean().item()
        eval_loss_pos += loss_pos.mean().item()
        eval_loss += loss.mean().item()

        # Metric
        eval_accuracy_ner += flat_accuracy(output_ner, b_ner_labels)
        eval_accuracy_pos += flat_accuracy(output_pos, b_pos_labels)

        eval_f1_ner += f1_score_seq(output_ner, b_ner_labels)
        eval_f1_pos += f1_score_seq(output_pos, b_pos_labels)

    # Calculate the average loss over the eval data.
    nb_eval = len(valid_dataloader)
    avg_eval_loss = eval_loss / nb_eval
    avg_eval_loss_ner = eval_loss_ner / nb_eval
    avg_eval_loss_pos = eval_loss_pos / nb_eval
    avg_eval_accuracy_ner = eval_accuracy_ner / nb_eval
    avg_eval_accuracy_pos = eval_accuracy_pos / nb_eval
    avg_eval_f1_ner = eval_f1_ner / nb_eval
    avg_eval_f1_pos = eval_f1_pos / nb_eval

    return avg_eval_loss, avg_eval_loss_ner, avg_eval_loss_pos, \
           avg_eval_accuracy_ner, avg_eval_accuracy_pos, \
           avg_eval_f1_ner, avg_eval_f1_pos


hps = get_hparams()

# Setup hyperparameters:
total_steps = hps.train.total_steps
epochs = hps.train.epochs
FULL_FINETUNING = hps.train.FULL_FINETUNING
lr = hps.train.lr
max_grad_norm = hps.train.max_grad_norm
is_save_metric = hps.train.is_save_metric
batch_size = hps.train.batch_size
eps = hps.train.eps
vocab_pos_path = f"{hps.preprocessing.out_path}/{hps.preprocessing.vocab_pos}"
vocab_ner_path = f"{hps.preprocessing.out_path}/{hps.preprocessing.vocab_ner}"
n_pos = get_number_vocab(vocab_pos_path)
n_ner = get_number_vocab(vocab_ner_path)

# Step 1: Initialize model
model = NeurondBERTMultitask(n_pos=n_pos, n_ner=n_ner, model_name=hps.base_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = CrossEntropyLoss()

# Initialize Optimizer
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=lr,
    eps=eps
)

# Step 2: Load data loader
train_inputs, val_inputs, train_masks, val_masks, train_poss, val_poss, train_ners, val_ners = load_from_path(
    hps.preprocessing.out_path
)

train_data = MultiDataset(train_inputs, train_masks, train_poss, train_ners)
valid_data = MultiDataset(val_inputs, val_masks, val_poss, val_ners)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size)

total_steps = len(train_dataloader) * epochs
num_warmup_steps = int(len(train_dataloader) * 0.05)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps
)

for _ in trange(epochs, desc="Epoch"):
    avg_train_loss, avg_train_loss_ner, avg_train_loss_pos, \
    avg_train_accuracy_ner, avg_train_accuracy_pos, \
    avg_train_f1_ner, avg_train_f1_pos = train(
        model, train_dataloader, optimizer, scheduler, criterion, max_grad_norm
    )

    avg_eval_loss, avg_eval_loss_ner, avg_eval_loss_pos, \
    avg_eval_accuracy_ner, avg_eval_accuracy_pos, \
    avg_eval_f1_ner, avg_eval_f1_pos = evaluate(model, valid_dataloader, criterion)

    print(f"""
    Averages of train loss: {avg_train_loss}, loss ner: {avg_train_loss_ner}, and loss_pos: {avg_train_loss_pos}
    Averages of valid loss: {avg_eval_loss}, loss ner: {avg_eval_loss_ner}, and loss_pos: {avg_eval_loss_pos}
    
    Accuracy of train NER: {avg_train_accuracy_ner} - Accuracy of train POS: {avg_train_accuracy_pos}
    Accuracy of valid NER: {avg_eval_accuracy_ner} - Accuracy of train POS: {avg_eval_accuracy_pos}
    
    F1-score of train NER: {avg_train_f1_ner} - F1-score of train POS: {avg_train_f1_pos}
    F1-score of valid NER: {avg_eval_f1_ner} - F1-score of train POS: {avg_eval_f1_pos}
    """)

