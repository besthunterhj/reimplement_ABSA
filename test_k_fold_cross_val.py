import sys
import os

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from utils import ABSADataset, collect_fc_pretrained, create_label_mapping
from sklearn.metrics import f1_score, classification_report, accuracy_score

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate_acc_f1(model, eval_data_loader):

    n_correct, n_total = 0, 0
    targets_all, outputs_all = None, None

    # sign the model to eval mode
    # model.eval()

    # validation stage doesn't need to change the params of the model
    with torch.no_grad():
        for i_batch, batch in enumerate(eval_data_loader):
            eval_inputs = batch["contexts"]
            eval_targets = batch["labels"]
            eval_predictions = model(eval_inputs)["logits"]

            # calculate the metrics of the validation model
            n_correct += (torch.argmax(eval_predictions, -1) == eval_targets).sum().item()
            n_total += len(eval_predictions)

            if targets_all is None:
                targets_all = eval_targets
                outputs_all = eval_predictions
            else:
                targets_all = torch.cat((targets_all, eval_targets), dim=0)
                outputs_all = torch.cat((outputs_all, eval_predictions), dim=0)

        # print the accuracy and f1_score of the model, which constructed by this epoch
        accuracy = n_correct / n_total
        f1 = f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average="macro")
        return accuracy, f1


def train(model, num_epoch: int, criterion, optimizer, train_data_loader, eval_data_loader, patience: int):

    # init the path to save the trained models
    path = None

    # init the variable "global_steps" which is constructed by "epoch_num * batch_size"
    global_steps = 0

    # init the accuracy, f1-score and corresponding epoch of evaluation stage
    max_val_acc, max_val_f1, max_val_epoch = 0, 0, 0

    for i_epoch in range(num_epoch):
        print("-" * 100)
        print("Epoch: {}".format(i_epoch))
        n_correct, n_total, loss_total = 0, 0, 0

        # sign the model to training mode
        model.train()
        for batch in tqdm(train_data_loader):
            global_steps += 1

            # clear the gradient accumulators
            optimizer.zero_grad()

            inputs = batch["contexts"].to(DEVICE)
            targets = batch["labels"].to(DEVICE)
            predictions = model(inputs)["logits"]

            # calculate the loss
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            # correction statistics for the training stage
            n_correct += (torch.argmax(predictions, -1) == targets).sum().item()
            n_total += len(predictions)
            loss_total += loss.item() * len(predictions)

            # print the statistics each 100 steps
            if global_steps % 100 == 0:
                train_acc = n_correct / n_total
                train_loss = loss_total / n_total
                print('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

        # validate the model
        val_acc, val_f1 = evaluate_acc_f1(model=model, eval_data_loader=eval_data_loader)
        print('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))

        # store and select the trained model
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_val_epoch = i_epoch

            # store the model of specific path
            if not os.path.exists("state_dict"):
                os.mkdir("state_dict")

            path = "state_dict/checkpoint_val_acc_{0}".format(round(val_acc, 4))
            torch.save(model.state_dict(), path)
            print(('>> saved: {}'.format(path)))

        if val_f1 > max_val_f1:
            max_val_f1 = val_f1
        if i_epoch - max_val_epoch >= patience:
            print(">> early stop.")
            break

    return path


def test(model, test_data_loader):

    # init "all_prediction" and "all_labels" to store the all results of prediction and labels in testset data
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_data_loader:
            inputs = batch["contexts"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            predictions = model(inputs)["logits"]

            all_labels.append(labels)
            # predictions are the shape as [batch_size, 4]
            all_predictions.append(predictions.argmax(dim=-1))

    all_labels = torch.cat(all_labels).detach().cpu().numpy()
    all_predictions = torch.cat(all_predictions).detach().cpu().numpy()

    # return the results of testing
    test_acc = accuracy_score(
        y_true=all_labels,
        y_pred=all_predictions,
    )
    f1 = f1_score(y_true=all_labels, y_pred=all_predictions, average='macro')

    return test_acc, f1


def main(cross_k_fold: int, batch_size: int, lr: float, l2reg: float, num_epoch: int, patience: int):

    # Construct the training and testing dataset by ABSADataset
    training_dataset = ABSADataset(path="dataset/Laptops_Train.xml.seg")
    testing_dataset = ABSADataset(path="dataset/Laptops_Test_Gold.xml.seg")

    # load the tokenizer (RobertaTokenizer class support the tokenization and pre-trained vocabulary) and tokenization
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    # load the model
    # model = RobertaModel.from_pretrained('roberta-base', num_labels=3).to(DEVICE)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(DEVICE)
    pretrained_state_dict = model.state_dict()
    _params = filter(lambda p: p.requires_grad, model.parameters())

    # Init the training dataloader
    collate_fc = lambda samples: collect_fc_pretrained(
        samples=samples,
        tokenizer=tokenizer
    )

    # set the optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=_params, lr=lr, weight_decay=l2reg)

    # init the data loader
    test_data_loader = DataLoader(
        dataset=testing_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fc
    )

    # k-fold cross validation
    valset_len = len(training_dataset) // cross_k_fold
    splitted_datasets = random_split(
        training_dataset,
        tuple([valset_len] * (cross_k_fold - 1) + [len(training_dataset) - valset_len * (cross_k_fold - 1)])
    )

    # store the statistics of each fold
    all_test_acc, all_test_f1 = [], []
    for fold_index in range(cross_k_fold):
        print("fold : {}".format(fold_index))
        print(">" * 100)
        trainset = ConcatDataset([x for i, x in enumerate(splitted_datasets) if i != fold_index])
        valset = splitted_datasets[fold_index]

        train_data_loader = DataLoader(
            dataset=trainset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fc
        )
        eval_data_loader = DataLoader(
            dataset=valset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fc
        )

        # before the training, the model should load the original parameters
        model.load_state_dict(pretrained_state_dict)

        # train for the best model
        best_model_path = train(
            model=model,
            num_epoch=num_epoch,
            criterion=criterion,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            eval_data_loader=eval_data_loader,
            patience=patience
        )

        model.load_state_dict(torch.load(best_model_path))

        # test for the best model
        test_acc, test_f1 = test(model=model, test_data_loader=test_data_loader)
        all_test_acc.append(test_acc)
        all_test_f1.append(test_f1)
        print('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

    mean_test_acc, mean_test_f1 = np.mean(all_test_acc), np.mean(all_test_f1)
    print('>' * 100)
    print('>>> mean_test_acc: {:.4f}, mean_test_f1: {:.4f}'.format(mean_test_acc, mean_test_f1))


if __name__ == '__main__':

    main(10, 32, 1e-5, 0.01, 20, 5)