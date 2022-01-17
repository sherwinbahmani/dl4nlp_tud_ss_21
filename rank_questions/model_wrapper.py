import torch
import math
from argparse import Namespace
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import pathlib

from rank_questions.model import Model
from rank_questions.dataset import Dataset, QuestionBankDataset
from rank_questions.metric import TopKAccuracy

class ModelWrapper(object):
    """
    This class wraps model, datasets, optimizer etc. and implements training and testing.
    """
    def __init__(self, opts: Namespace):
        self.num_epochs = opts.num_epochs
        self.top_k_accuracy = opts.top_k_accuracy
        self.true_label = opts.true_label
        self.val_start = opts.val_start
        self.val_step = opts.val_step
        self.val = opts.val
        self.val_split = opts.val_split
        self.checkpoint_path = os.path.join(opts.checkpoints_root, f'{opts.time_stamp}.pth')
        self.txt_path = os.path.join(opts.txt_root, f'{opts.time_stamp}.txt')
        pathlib.Path(opts.checkpoints_root).mkdir(parents=True, exist_ok=True)
        pathlib.Path(opts.txt_root).mkdir(parents=True, exist_ok=True)

        # Set up tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(opts.runs_root, opts.time_stamp))

        # Set up model, optimizer and loss
        self.device = torch.device(opts.device)
        self.model = Model(device=self.device)
        if opts.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=opts.lr, betas=opts.betas, weight_decay=opts.weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Metrics
        self.val_metric = TopKAccuracy(k=opts.top_k_accuracy)

        # Datasets
        self.train_dataset = Dataset(opts, split="train")
        self.test_dataset = Dataset(opts, split="test")
        self.question_bank_dataset = QuestionBankDataset(opts)

        # Create val dataset from train dataset
        self.split_train_val()

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers
        )
        self.question_bank_loader = DataLoader(
            self.question_bank_dataset, batch_size=1, shuffle=False, num_workers=opts.num_workers
        )

    def split_train_val(self):
        # Split first part into val and second part into train where negative example are included
        split_size_val = math.floor(self.val_split * self.train_dataset.len_orig)
        split_size_train = len(self.train_dataset) - split_size_val
        split_indices_val = list(range(split_size_val))
        split_indices_train = list(range(split_size_val, split_size_train + split_size_val))
        # Create subsets from train dataset for val and train
        self.val_dataset = torch.utils.data.Subset(self.train_dataset, split_indices_val)
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, split_indices_train)

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            # Training
            self.model.train()
            for train_idx, (query_train, question_train, label_train) in enumerate(self.train_loader):
                # Forward pass
                train_out_logits, _ = self.model(query_train, question_train)

                # Loss calculation
                label_train = label_train.to(self.device)
                train_loss = self.criterion(train_out_logits, label_train)
                self.writer.add_scalar("Training/Loss_steps", train_loss, train_idx + len(self.train_loader) * epoch)

                # Backprop and weight updates
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

            # Validation
            if self.val and epoch >= self.val_start and epoch % self.val_step == 0:
                self.model.eval()
                val_scores = torch.empty(len(self.val_loader), len(self.question_bank_loader))
                qids = []
                for bank_idx, (qid, question_bank) in enumerate(self.question_bank_loader):
                    questions_vals = []
                    for val_idx, (query_val, question_val, label_val) in enumerate(self.val_loader):
                        # Forward pass
                        with torch.no_grad():
                            _, val_out_soft = self.model(query_val, question_bank)

                        # Get probability of this query matching this question from the question bank
                        val_out_soft_true = val_out_soft[:, self.true_label]

                        # For every sample in batch
                        for batch_idx in range(len(query_val)):
                            val_scores[val_idx + batch_idx, bank_idx] = val_out_soft_true[batch_idx].detach().item()
                            questions_vals.append(question_val[batch_idx])
                    qids.append(*qid)

                # Get indices of highest scores to get QIDs with highest scores
                val_scores_ind = val_scores.sort(descending=True)[1].cpu().numpy()
                qids_scores = np.array(qids)[val_scores_ind]
                qids_labels = np.array([self.question_bank_dataset.get_qid(question) for question in questions_vals])

                # Compute metric
                acc = self.val_metric.compute(qids_scores, qids_labels)
                self.writer.add_scalar("Validation/Acc", acc, epoch)

                # Save checkpoint if accuracy is higher than in best epoch
                if acc > self.val_metric.best:
                    self.val_metric.best = acc
                    if os.path.isfile(self.checkpoint_path):
                        os.remove(self.checkpoint_path)
                    torch.save(self.model.state_dict(), self.checkpoint_path)
                    print(f"New best accuracy = {self.val_metric.best}")
        print(f"Best accuracy = {self.val_metric.best}")

    def test(self):
        self.model.eval()
        test_scores = torch.empty(len(self.test_loader), len(self.question_bank_loader))
        qids = []
        for bank_idx, (qid, question_bank) in enumerate(self.question_bank_loader):
            queries_test = []
            for test_idx, query_test in enumerate(self.test_loader):
                # Forward pass
                with torch.no_grad():
                    _, test_out_soft = self.model(query_test, question_bank)

                # Get probability of this query matching this question from the question bank
                test_out_soft_true = test_out_soft[:, self.true_label]

                # For every sample in batch
                for batch_idx in range(len(query_test)):
                    test_scores[test_idx + batch_idx, bank_idx] = test_out_soft_true[batch_idx].item()
                queries_test.append(*query_test)
            qids.append(*qid)

        # Get indices of highest scores to get QIDs with highest scores
        test_scores_ind = test_scores.sort(descending=True)[1].numpy()
        qids_scores = np.array(qids)[test_scores_ind]

        # Only use top k highest predictions
        qids_scores = qids_scores[:, :self.top_k_accuracy]

        # Create output txt from QID
        with open(self.txt_path, 'w') as f:
            for query_test, qids in zip(queries_test, qids_scores):
                f.write(f"{query_test}\t" + ",".join(qids) + "\n")