import torch
import random
import os
import math
from argparse import Namespace
import pandas as pd
import numpy as np
from typing import Tuple, Union

class Dataset(torch.utils.data.Dataset):
    """
    This class implements the question rank dataset.
    """

    def __init__(self, opts: Namespace, split: str = "train"):
        super(Dataset, self).__init__()
        self.data_root = opts.data_root
        self.split = split
        self.len_orig = None

        # Read data
        if self.split == "train":
            queries_questions = self.read_tsv(opts.train_file_name)
            # Randomly shuffle for train and val
            np.random.shuffle(queries_questions)
            self.queries, self.questions = self.split_data(queries_questions)
            self.len_orig = self.queries.shape[0]
        elif self.split == "test":
            self.queries = self.read_tsv(opts.test_file_name, header=None).squeeze(1)

        # Add if 0 or 1 (0 = don't belong to each other, 1= belong to each other)
        self.labels = [torch.tensor(opts.true_label)] * len(self)

        # Create negative examples as much as the future train dataset will be (50 % positive, 50 % negative)
        if self.split == "train":
            self.create_negative_examples(val_split=opts.val_split, false_label=opts.false_label)

    def __len__(self) -> int:
        return self.queries.shape[0]

    def __getitem__(self, idx: int) -> Union[str, Tuple[str, str, torch.Tensor]]:
        query = self.queries[idx]
        if self.split == "test":
            return query
        elif self.split == "train":
            question = self.questions[idx]
            label = self.labels[idx]
            return query, question, label

    def read_tsv(self, file_name: str, header: Union[str, None] = "infer") -> np.ndarray:
        arr = pd.read_csv(os.path.join(self.data_root, file_name), sep="\t", header=header).to_numpy()
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr[i, j] = arr[i, j].rstrip().replace("  ", " ")
        return arr

    def split_data(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr1, arr2 = np.split(arr, indices_or_sections=2, axis=1)
        arr1 = arr1.squeeze(1)
        arr2 = arr2.squeeze(1)
        return arr1, arr2

    def create_negative_examples(self, val_split: int, false_label: int):
        # Take the queries and combine them with clarifying questions which are not the correct one
        num_samples = math.floor((1 - val_split) * len(self))
        queries_neg = self.queries[len(self) - num_samples:]
        questions_neg = []
        for q_idx, query in enumerate(queries_neg):
            wrong_questions = [question for question in self.questions if question != self.questions[q_idx]]
            questions_neg.append(random.choice(wrong_questions))

        # Add negative examples to original examples
        self.questions = np.concatenate((self.questions, np.array(questions_neg)))
        self.queries = np.concatenate((self.queries, queries_neg))
        self.labels += [torch.tensor(false_label)] * num_samples

class QuestionBankDataset(Dataset):
    """
    This class implements the question bank dataset.
    """
    def __init__(self, opts: Namespace):
        super(Dataset, self).__init__()
        self.data_root = opts.data_root
        questions_bank = self.read_tsv(opts.question_bank_name)
        self.qids_bank, self.questions_bank = self.split_data(questions_bank)

    def __len__(self) -> int:
        return len(self.qids_bank)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.qids_bank[idx], self.questions_bank[idx]

    def get_qid(self, question: str) -> str:
        idx = np.where(self.questions_bank == question)[0][0]
        return self.qids_bank[idx]
