"""Train models on a given dataset."""
import os
import json
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from common import OUTPUT_DIR, CHECKPOINT_DIR


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclass
class LoggingParameters:
    """Data class holding parameters for logging."""
    model_name: str
    dataset_name: str
    optimizer_name: str
    optimizer_params: dict

# pylint: disable=R0902, R0913, R0914
class Trainer:
    """Abstract model trainer on a binary classification task."""
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion,
                 batch_size: int,
                 train_dataset: Dataset,
                 validation_dataset: Dataset,
                 test_dataset: Dataset):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.epoch = 0

    def train_one_epoch(self) -> tuple[float, float]:
        """Train the model for a single epoch on the training dataset.
        Returns:
            (avg_loss, accuracy): tuple containing the average loss and
            accuracy across all dataset samples.
        """
        self.model.train()
        total_loss = 0
        avg_loss = 0
        accuracy = 0
        nof_samples = 0
        correct_labeled_samples = 0
        correct_predictions = 0
        total_samples = 0

        train_dataloader = DataLoader(self.train_dataset,
                                      self.batch_size,
                                      shuffle=True)
        print_every = int(len(train_dataloader) / 10)

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to the appropriate device

            # Zero the gradients before running the backward pass.
            self.optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = self.model(inputs)

            # Calculate the loss
            loss = self.criterion(outputs, targets)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            self.optimizer.step()

            # Update total loss for the epoch
            total_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

            if batch_idx % print_every == 0 or \
                    batch_idx == len(train_dataloader) - 1:
                print(f'Epoch [{self.epoch:03d}] | Loss: {avg_loss:.3f} | '
                      f'Acc: {accuracy:.2f}[%] '
                      f'({correct_labeled_samples}/{nof_samples})')

        # Calculate average loss and accuracy over the epoch
        avg_loss = total_loss / total_samples
        accuracy = (correct_predictions / total_samples) * 100

        return avg_loss, accuracy

    def evaluate_model_on_dataloader(
            self, dataset: torch.utils.data.Dataset) -> tuple[float, float]:
        """Evaluate model loss and accuracy for dataset.

        Args:
            dataset: the dataset to evaluate the model on.

        Returns:
            (avg_loss, accuracy): tuple containing the average loss and
            accuracy across all dataset samples.
        """
        self.model.eval()
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False)
        total_loss = 0
        avg_loss = 0
        accuracy = 0
        nof_samples = 0
        correct_labeled_samples = 0
        print_every = max(int(len(dataloader) / 10), 1)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            """INSERT YOUR CODE HERE."""
            inputs, targets = inputs.to(device), targets.to(device)

            # Disable gradient computation for evaluation
            with torch.no_grad():
                # Forward pass
                outputs = self.model(inputs)

                # Compute the loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)

                #compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_labeled_samples += (predicted == targets).sum().item()
                nof_samples += targets.size(0)

                #calculate avg loss and accuracy for now
                avg_loss = total_loss / nof_samples
                accuracy = 100.0 * correct_labeled_samples / nof_samples

            if batch_idx % print_every == 0 or batch_idx == len(dataloader) - 1:
                print(f'Epoch [{self.epoch:03d}] | Loss: {avg_loss:.3f} | '
                      f'Acc: {accuracy:.2f}[%] '
                      f'({correct_labeled_samples}/{nof_samples})')

        return avg_loss, accuracy

    def validate(self):
        """Evaluate the model performance."""
        return self.evaluate_model_on_dataloader(self.validation_dataset)

    def test(self):
        """Test the model performance."""
        return self.evaluate_model_on_dataloader(self.test_dataset)

    @staticmethod
    def write_output(logging_parameters: LoggingParameters, data: dict):
        """Write logs to json.

        Args:
            logging_parameters: LoggingParameters. Some parameters to log.
            data: dict. Holding a dictionary to dump to the output json.
        """
        if not os.path.isdir(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        output_filename = f"{logging_parameters.dataset_name}_" \
                          f"{logging_parameters.model_name}_" \
                          f"{logging_parameters.optimizer_name}.json"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        print(f"Writing output to {output_filepath}")
        # Load output file
        if os.path.exists(output_filepath):
            # pylint: disable=C0103
            with open(output_filepath, 'r', encoding='utf-8') as f:
                all_output_data = json.load(f)
        else:
            all_output_data = []

        # Add new data and write to file
        all_output_data.append(data)
        # pylint: disable=C0103
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, indent=4)

    def run(self, epochs, logging_parameters: LoggingParameters):
        """Train, evaluate and test model on dataset, finally log results."""
        output_data = {
            "model": logging_parameters.model_name,
            "dataset": logging_parameters.dataset_name,
            "optimizer": {
                "name": logging_parameters.optimizer_name,
                "params": logging_parameters.optimizer_params,
            },
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "test_loss": [],
            "test_acc": [],
        }
        best_acc = 0
        model_filename = f"{logging_parameters.dataset_name}_" \
                         f"{logging_parameters.model_name}_" \
                         f"{logging_parameters.optimizer_name}.pt"
        checkpoint_filename = os.path.join(CHECKPOINT_DIR, model_filename)
        for self.epoch in range(1, epochs + 1):
            print(f'Epoch {self.epoch}/{epochs}')

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            test_loss, test_acc = self.test()

            output_data["train_loss"].append(train_loss)
            output_data["train_acc"].append(train_acc)
            output_data["val_loss"].append(val_loss)
            output_data["val_acc"].append(val_acc)
            output_data["test_loss"].append(test_loss)
            output_data["test_acc"].append(test_acc)

            # Save checkpoint
            if val_acc > best_acc:
                print(f'Saving checkpoint {checkpoint_filename}')
                state = {
                    'model': self.model.state_dict(),
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'epoch': self.epoch,
                }
                torch.save(state, checkpoint_filename)
                best_acc = val_acc
        self.write_output(logging_parameters, output_data)
