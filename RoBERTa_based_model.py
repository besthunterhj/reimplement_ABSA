from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from utils import ABSADataset, create_label_mapping, collect_fc_pretrained


DEVICE = 'cpu'


def train_vanilla_roberta(model, criterion, optimizer, training_loader: DataLoader):

    # sign to begin the training
    model.train()

    # store the training losses
    n_correct, n_total, loss_total = 0, 0, 0

    # training loop
    for batch in tqdm(training_loader):
        contexts = batch["contexts"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        # get the prediction of this loop
        current_prediction = model(contexts)["logits"]

        # calculate the current loss
        current_loss = criterion(current_prediction, labels)

        # back propagation
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        # calculate and print the training loss for this epoch
        n_correct += (torch.argmax(current_prediction, -1) == labels).sum().item()
        n_total += len(current_prediction)
        loss_total += current_loss.item() * len(current_prediction)

        train_acc = n_correct / n_total
        train_loss = loss_total / n_total
        print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))



def evaluate_acc(model, criterion, eval_dataloader):

    # sign to begin the evaluation
    model.eval()

    # store all labels from test dataset
    test_labels = []
    test_prediction = []

    # calculate the loss of evaluation
    losses = []

    # we don't need to calculate the gradient at the evaluation
    with torch.no_grad():
        for batch in eval_dataloader:
            contexts = batch["contexts"]
            labels = batch["labels"]

            current_prediction_ = model(contexts)["logits"]


def main(batch_size: int, label_smoothing: float, lr: float, weight_decay: float, epoch_num: int):

    # Construct the training and testing dataset by ABSADataset
    training_dataset = ABSADataset(path="dataset/Laptops_Train.xml.seg")
    testing_dataset = ABSADataset(path="dataset/Laptops_Test_Gold.xml.seg")

    # load the tokenizer (RobertaTokenizer class support the tokenization and pre-trained vocabulary) and tokenization
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # load the label mapping, which consists of label(int) -> textual_label(str)
    label_mapping = create_label_mapping(dataset=training_dataset)

    # load the model
    # model = RobertaModel.from_pretrained('roberta-base', num_labels=3).to(DEVICE)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(DEVICE)
    _params = filter(lambda p: p.requires_grad, model.parameters())

    # Init the training dataloader
    collate_fc = lambda samples: collect_fc_pretrained(
        samples=samples,
        tokenizer=tokenizer
    )

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fc
    )

    # set the optimizer and loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(params=_params, lr=lr, weight_decay=weight_decay)

    # training
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch + 1}")
        train_vanilla_roberta(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            training_loader=training_dataloader,
        )


if __name__ == '__main__':

    main(batch_size=16, label_smoothing=0.1, lr=5e-5, weight_decay=0.01, epoch_num=3)

    # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

    # test_tokenizer = tokenizer(input_data, padding=True, truncation=True)["input_ids"]



