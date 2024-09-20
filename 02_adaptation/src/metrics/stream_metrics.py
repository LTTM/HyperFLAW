import numpy as np
import torch
from utils.condition_map import REVERSE_MAPPING_WEATHER


class StreamSegMetrics:
    def __init__(self, n_classes, name):
        super().__init__()
        self.n_classes = n_classes
        self.name = name
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.confusion_matrix_weather = None
        self.total_samples = 0
        self.results = {}

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds, conditions=[0]):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        self.total_samples += len(label_trues)

        if conditions[0] != 0:
            self.confusion_matrix_weather = (
                self.confusion_matrix_weather
                if self.confusion_matrix_weather is not None
                else {
                    k: np.zeros((self.confusion_matrix.shape))
                    for k in list(REVERSE_MAPPING_WEATHER.keys())[1:]
                }
            )
            for lt, lp in zip(label_trues, label_preds):
                self.confusion_matrix_weather[conditions[0]] += self._fast_hist(
                    lt.flatten(), lp.flatten()
                )

    @staticmethod
    def compute_stats(confusion_matrix):
        EPS = 1e-6
        hist = confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = gt_sum != 0
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        precision_cls_c = diag / (hist.sum(axis=0) + EPS)
        precision_cls = np.mean(precision_cls_c)
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        return {
            "mask": mask,
            "acc": acc,
            "acc_cls": acc_cls,
            "acc_cls_c": acc_cls_c,
            "precision_cls_c": precision_cls_c,
            "precision_cls": precision_cls,
            "iu": iu,
            "mean_iu": mean_iu,
            "fwavacc": fwavacc,
        }

    def get_results(self):
        stats = self.compute_stats(self.confusion_matrix)
        if self.confusion_matrix_weather is not None:
            weather_iu = {
                k: self.compute_stats(v)["mean_iu"] for k, v in self.confusion_matrix_weather.items()
            }
        else:
            weather_iu = None

        cls_iu = dict(
            zip(
                range(self.n_classes),
                [stats["iu"][i] if m else "X" for i, m in enumerate(stats["mask"])],
            )
        )
        cls_acc = dict(
            zip(
                range(self.n_classes),
                [
                    stats["acc_cls_c"][i] if m else "X"
                    for i, m in enumerate(stats["mask"])
                ],
            )
        )
        cls_prec = dict(
            zip(
                range(self.n_classes),
                [
                    stats["precision_cls_c"][i] if m else "X"
                    for i, m in enumerate(stats["mask"])
                ],
            )
        )

        self.results = {
            "Total samples": self.total_samples,
            "Overall Acc": stats["acc"],
            "Mean Acc": stats["acc_cls"],
            "Mean Precision": stats["precision_cls"],
            "FreqW Acc": stats["fwavacc"],
            "Mean IoU": stats["mean_iu"],
            "Class IoU": cls_iu,
            "Class Acc": cls_acc,
            "Class Prec": cls_prec,
            "Weather IoU": weather_iu,
        }

        return self.results

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        if self.confusion_matrix_weather is not None:
            self.confusion_matrix_weather = {
                k: np.zeros((self.n_classes, self.n_classes))
                for k in list(REVERSE_MAPPING_WEATHER.keys())[1:]
            }
        self.total_samples = 0

    def synch(self, device):
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        if self.confusion_matrix_weather is not None:
            confusion_matrix_weather = {
                k: torch.tensor(v).to(device)
                for k, v in self.confusion_matrix_weather.items()
            }
        samples = torch.tensor(self.total_samples).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        if self.confusion_matrix_weather is not None:
            for v in confusion_matrix_weather.values():
                torch.distributed.reduce(v, dst=0)
        torch.distributed.reduce(samples, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            if self.confusion_matrix_weather is not None:
                self.confusion_matrix_weather = {
                    k: v.cpu().numpy() for k, v in confusion_matrix_weather.items()
                }
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_text(self):
        string = []
        for i in range(self.n_classes):
            string.append(f"{i} : {self.confusion_matrix[i].tolist()}")
        return "\n" + "\n".join(string)

    def __str__(self):
        string = "\n"
        ignore = [
            "Class IoU",
            "Class Acc",
            "Class Prec",
            "Confusion Matrix Pred",
            "Confusion Matrix",
            "Confusion Matrix Text",
        ]
        for k, v in self.results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)

        string += "Class IoU:\n"
        for k, v in self.results["Class IoU"].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += "Class Acc:\n"
        for k, v in self.results["Class Acc"].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += "Class Prec:\n"
        for k, v in self.results["Class Prec"].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string
