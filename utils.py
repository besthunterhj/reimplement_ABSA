from typing import List, Tuple, Dict, Callable

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab
from transformers import RobertaTokenizer


class ABSADataset(Dataset):

    def __init__(self, path: str):
        super(ABSADataset, self).__init__()
        self.context_aspect_dicts, self.polarities = self.read_corpus(path=path)

    def __getitem__(self, index):
        return self.context_aspect_dicts[index], self.polarities[index]

    def __len__(self):
        return len(self.polarities)

    @classmethod
    def read_corpus(cls, path: str) -> Tuple[List[Dict], List[int]]:
        """
        Read the data of ABSA dataset (Restaurant, Laptop)
        :param path: the path of dataset
        :return: A tuple which contains the list of dictionary and the list of polarities
        """

        context_aspect_dicts = []
        polarities = []

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

            for i in range(0, len(lines), 3):
                # get the context, which consists of two parts: the left context and right context of the aspect term
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                polarity = lines[i + 2].strip()

                # revert the whole sentence by text_left, text_right and aspect term
                context = text_left + " " + aspect + " " + text_right

                # construct the input data dictionary
                context_and_aspect = {
                    "context": context,
                    "aspect": aspect
                }

                # change the type of polarity to integer
                polarity = int(polarity) + 1

                context_aspect_dicts.append(context_and_aspect)
                polarities.append(polarity)

        return context_aspect_dicts, polarities


def make_inputs_for_roberta(contexts_and_aspect: Dict) -> str:
    """
    Making the input data for roberta, each item should be like "context + </s> aspect"
    :param contexts_and_aspect: the dictionary contains contexts and aspects
    :return: the input data which is formatted by the corresponding format
    """

    context = contexts_and_aspect["context"] + " </s> " + contexts_and_aspect["aspect"]

    return context


def create_label_mapping(dataset: ABSADataset) -> Dict[int, str]:

    label_types = list(set(dataset.polarities))
    textual_labels = ["negative", "neutral", "positive"]

    # construct the dictionary for storing the mapping for labels(integers) to textual labels
    integer2text = {}
    for i in range(len(label_types)):
        integer2text[i] = textual_labels[i]

    return integer2text


def collect_fc_pretrained(samples: List[Tuple[str, int]], tokenizer: RobertaTokenizer) -> dict:

    context_aspect_dicts, labels = zip(*samples)

    contexts = torch.tensor(
        list(
            tokenizer(
                list(
                    map(
                        lambda current_context_aspect_dict: make_inputs_for_roberta(current_context_aspect_dict),
                        context_aspect_dicts
                    )
                ), padding=True, truncation=True
            )["input_ids"]
        )
    )

    labels = torch.tensor(
        list(labels)
    )

    return {
        "contexts": contexts,
        "labels": labels
    }
