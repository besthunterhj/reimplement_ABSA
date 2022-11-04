from typing import List, Tuple, Dict

import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import Dataset


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


def make_inputs_for_roberta(contexts_and_aspects: List[Dict]) -> List[str]:
    """
    Making the input data for roberta, each item should be like "context + </s> aspect"
    :param contexts_and_aspects: the dictionary contains contexts and aspects
    :return: the input data which is formatted by the corresponding format
    """

    inputs = []
    for i in range(len(contexts_and_aspects)):
        item = contexts_and_aspects[i]["context"] + " </s> " + contexts_and_aspects[i]["aspect"]
        inputs.append(item)

    return inputs


if __name__ == '__main__':

    # test for ABSADataset
    absa_dataset = ABSADataset(path="dataset/Laptops_Train.xml.seg")
    input_data = make_inputs_for_roberta(contexts_and_aspects=absa_dataset[:][0])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

    test_tokenizer = tokenizer(input_data, padding=True)["input_ids"]

    print(test_tokenizer[:5])

