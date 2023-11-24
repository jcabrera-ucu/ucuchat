import re
import os
import numpy as np
import json
import torch


class Trainer:
    """Main class for model training"""

    def __init__(
            self,
            model,
            epochs,
            data_partition,
            train_dataloader,
            train_steps,
            val_dataloader,
            val_steps,
            checkpoint_frequency,
            criterion,
            optimizer,
            lr_scheduler,
            device,
            model_dir,
            model_name,
    ):
        self.model = model
        self.epochs = epochs
        self.data_partition = data_partition
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)

    def train(self):
        for i in range(self.data_partition):
            print(f'Training data in partition {i + 1}.')
            for epoch in range(self.epochs):
                self._train_epoch(self.train_dataloader[i])
                self._validate_epoch(self.val_dataloader[i])
                print(
                    "Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                        epoch + 1,
                        self.epochs,
                        self.loss["train"][-1],
                        self.loss["val"][-1],
                    )
                )

                self.lr_scheduler.step(self.loss["train"][-1])

                if self.checkpoint_frequency:
                    self._save_checkpoint(epoch)

            torch.cuda.empty_cache()

    def _train_epoch(self, train_dataloader):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(train_dataloader):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.train_steps:
                break

        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self, val_dataloader):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        """Save train/val loss as json file to `self.model_dir` directory"""
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)
