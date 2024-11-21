import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from torch import device
from tqdm import tqdm
from pathlib import Path

from .dataset.condition_map import MAPPING_WEATHER


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader = None,
        eval_loader: DataLoader = None,
        test_loader: DataLoader = None,
        criterion: nn.Module = None,
        optimizer: optim = None,
        device: device = torch.device("cpu"),
        ckpt_path: Path = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.ckpt_path = ckpt_path

        if self.ckpt_path is not None:
            self.int_ckpt_path = ckpt_path / "intermediate"

            self.int_ckpt_path.mkdir(parents=True, exist_ok=True)

    def train(self, num_epochs: int) -> dict:
        assert self.train_loader is not None, "The train loader must be set"
        assert self.eval_loader is not None, "The eval loader must be set"

        pbar = tqdm(range(num_epochs), total=num_epochs, desc="Training epochs")

        train_losses = np.zeros(num_epochs)
        eval_losses = np.zeros(num_epochs)
        train_accuracies = np.zeros((self.model.get_num_classes(), num_epochs))
        eval_accuracies = np.zeros((self.model.get_num_classes(), num_epochs))

        best_accuracies = np.zeros(self.model.get_num_classes())
        best_mean_accuracy = 0
        best_epoch = 0
        best_cf_matrix = None

        for epoch in pbar:
            # Train the model
            train_epoch_losses, train_c_matrix = self.__train_loop()

            # Eval the model
            eval_epoch_losses, eval_c_matrix = self.__eval_loop()

            # Update the epoch metrics
            train_losses[epoch] = np.mean(train_epoch_losses)
            train_accuracies[:, epoch] = np.nan_to_num(
                np.diag(train_c_matrix) / np.sum(train_c_matrix, axis=1), 0
            )
            train_label_accuracies = {
                k: round(v * 100, 2)
                for k, v in zip(MAPPING_WEATHER.keys(), train_accuracies[:, epoch])
            }

            eval_losses[epoch] = np.mean(eval_epoch_losses)
            eval_accuracies[:, epoch] = np.nan_to_num(
                np.diag(eval_c_matrix) / np.sum(eval_c_matrix, axis=1), 0
            )
            eval_label_accuracies = {
                k: round(v * 100, 2)
                for k, v in zip(MAPPING_WEATHER.keys(), eval_accuracies[:, epoch])
            }
            eval_mean_accuracy = np.mean(eval_accuracies[:, epoch])

            # Check if the current model is the best one
            if eval_mean_accuracy > best_mean_accuracy:
                best_mean_accuracy = eval_mean_accuracy
                best_epoch = epoch + 1
                best_cf_matrix = eval_c_matrix
                best_accuracies = eval_accuracies[:, epoch]

                # Save the model
                if self.ckpt_path is not None:
                    torch.save(self.model.state_dict(), self.ckpt_path / "model.pth")

            # Save the intermediate model
            torch.save(
                self.model.state_dict(),
                self.int_ckpt_path / f"model_epoch_{epoch + 1}.pth",
            )

            if self.ckpt_path is not None:
                with open(self.ckpt_path / "train_log.txt", "a") as f:
                    f.write(
                        f"Epoch {epoch + 1} -> train_loss: {round(train_losses[epoch].item(), 4)} | train_accuracy: {train_label_accuracies}%\n"
                    )
                    f.write(
                        f"           eval_loss: {round(eval_losses[epoch].item(), 4)} | eval_accuracy: {eval_label_accuracies}%\n"
                    )

        output = {
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "eval_losses": eval_losses,
            "eval_accuracies": eval_accuracies,
            "best_epoch": best_epoch,
            "best_accuracies": best_accuracies,
            "best_cf_matrix": best_cf_matrix,
        }

        return output

    def test(self) -> dict:
        assert self.test_loader is not None, "The test loader must be set"

        pred, gt, confusion_matrix = self.__test_loop()

        acuracy = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

        output = {
            "prediction": pred,
            "gt": gt,
            "c_matrix": confusion_matrix,
            "accuracy": acuracy,
        }

        return output

    def set_train_loader(self, train_loader: DataLoader) -> None:
        self.train_loader = train_loader

    def set_test_loader(self, test_loader: DataLoader) -> None:
        self.test_loader = test_loader

    def set_eval_loader(self, eval_loader: DataLoader) -> None:
        self.eval_loader = eval_loader

    def set_criterion(self, criterion: nn.Module) -> None:
        self.criterion = criterion

    def set_optimizer(self, optimizer: optim) -> None:
        self.optimizer = optimizer

    def __train_loop(self):
        self.model.train()

        inner_pbar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc="Batches",
            leave=False,
        )

        confusion_matrix = np.zeros(
            (self.model.get_num_classes(), self.model.get_num_classes())
        )
        epoch_losses = np.zeros(len(self.train_loader))

        for i, data in enumerate(inner_pbar):
            inputs, labels = [elm.to(self.device) for elm in data]

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _outputs = torch.argmax(nn.functional.softmax(outputs, dim=1), dim=1)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Update the epoch metrics
            epoch_losses[i] = loss.item()
            confusion_matrix[labels.cpu().numpy(), _outputs.cpu().numpy()] += 1

        return epoch_losses, confusion_matrix

    def __eval_loop(self):
        self.model.eval()

        inner_pbar = tqdm(
            self.eval_loader,
            total=len(self.eval_loader),
            desc="Eval samples",
            leave=False,
        )

        confusion_matrix = np.zeros(
            (self.model.get_num_classes(), self.model.get_num_classes())
        )
        losses = np.zeros(len(self.eval_loader))

        with torch.no_grad():
            for i, data in enumerate(inner_pbar):
                inputs, labels = [elm.to(self.device) for elm in data]

                outputs = self.model(inputs)
                _outputs = torch.argmax(nn.functional.softmax(outputs, dim=1), dim=1)

                loss = self.criterion(outputs, labels)

                # Update the epoch metrics
                losses[i] = loss.item()
                confusion_matrix[labels, _outputs] += 1

        return losses, confusion_matrix

    def __test_loop(self):
        self.model.eval()

        pbar = (
            tqdm(
                self.test_loader,
                total=len(self.test_loader),
                desc="Test samples",
            )
            if len(self.test_loader) > 1
            else self.test_loader
        )

        confusion_matrix = np.zeros(
            (self.model.get_num_classes(), self.model.get_num_classes())
        )

        outputs = np.zeros(len(self.test_loader))
        gt = np.zeros(len(self.test_loader))

        with torch.no_grad():
            for i, data in enumerate(pbar):
                input, label = [elm.to(self.device) for elm in data]

                output = self.model(input)
                _output = torch.argmax(nn.functional.softmax(output, dim=1), dim=1)

                # Update the epoch metrics
                confusion_matrix[label, _output] += 1

                # Save the output
                outputs[i] = _output.item()

                # Save the ground truth
                gt[i] = label.item()
        
        return outputs, gt, confusion_matrix
