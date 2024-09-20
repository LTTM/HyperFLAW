import time

import wandb
import os
import numpy as np
import pytorch_lightning
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Iterable
from numbers import Number
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from .data_utils import Label2Color, color_map, Denormalize


class Writer:

    def __init__(self, local_rank):
        self.local_rank = local_rank
        self.valid_target_ds_names = (['cityscapes'], ['acdc'], ['flyawarer'], ['flyawarerxl'], ['acdc', 'flyawarer'],
                                      ['flyawarer', 'acdc'], ['acdc', 'flyawarerxl'], ['flyawarerxl', 'acdc'])
        self.source_label2color = self.source_denorm = self.target_label2color = self.target_denorm = None
        self.wandb = None

    @staticmethod
    def __set_img_utils(valid_ds_names, dataset_name, mean, std):
        if dataset_name in valid_ds_names:
            label2color = Label2Color(cmap=color_map(dataset_name))
            denorm = Denormalize(mean=mean, std=std)
            return label2color, denorm
        return None, None

    def set_source_img_utils(self, dataset_name, mean, std):
        self.source_label2color, self.source_denorm = \
            self.__set_img_utils(self.valid_target_ds_names, dataset_name, mean, std)

    def set_target_img_utils(self, dataset_name, mean, std):
        self.target_label2color, self.target_denorm = \
            self.__set_img_utils(self.valid_target_ds_names, dataset_name, mean, std)

    def set_wandb(self, wandb_logger):
        self.wandb = wandb_logger

    def write(self, msg):
        if self.local_rank == 0:
            if 'CLIENT' in msg and self.wandb.args.save_clients_order:
                with open(os.path.join(os.getcwd(), self.wandb.job_name) + "_clients.txt", "a") as f:
                    f.write(msg + "\n")
            print(msg)

    @staticmethod
    def print_model_chunk(string, model, precision=10):
        for _, v in model.named_parameters():
            print(f"{string}: [", end='')
            chunk = v[0][0][0].tolist()
            for e in chunk[:-1]:
                print(f"{round(e, precision)}, ", end='')
            print(f"{round(chunk[-1], precision)}]")
            break

    def plot_step_loss(self, metric_name, step, losses):
        if self.local_rank == 0:
            for name, l in losses.items():
                if name == "source_test_loss":
                    name = "test_loss"
                self.wandb.log_metrics({f"{metric_name}_{name}": l}, step=step + 1)
                # self.wandb.log_metrics({f"{name}": l}, step=step + 1)

    def print_step_loss(self, client_name, metric_name, step, len_loader, losses):
        if self.local_rank == 0:
            str_to_print = f"{client_name}, {step + 1}/{len_loader}: "
            for name, l in losses.items():
                str_to_print += f"{metric_name}_{name} = {round(l.item(), 3)}, "
            str_to_print = str_to_print[:-2]
            self.write(str_to_print)

    def plot_step_lr(self, metric_name, step, lr):
        if self.local_rank == 0:
            self.wandb.log_metrics({f"{metric_name}_lr": lr}, step=step + 1)

    def plot_metric(self, step, metric, cl_name, ret_score, classes=None):
        if self.local_rank == 0:
            train_score = metric.get_results()
            # self.wandb.log_metrics({f'{metric.name}_{cl_name}_{ret_score.lower()}': train_score[ret_score]},
            #                        step=step + 1)
            if "test_user" in cl_name:
                name = "target_" + cl_name
            elif "add" in cl_name:
                name = cl_name.split("add_")[-1]
            else:
                name = cl_name
            if classes is None:
                self.wandb.log_metrics({f'{name}_{ret_score.lower()}': train_score[ret_score]},
                                       step=step + 1)
            else:
                train_score = train_score[ret_score]
                keys = [classes[k] for k in train_score.keys()]
                values = list(train_score.values())
                values = [float(np.round(v, 4)) if type(v) is not str else float(0.0) for v in values]  # Replace X with 0.0

                data = {f'{name}_{ret_score.lower()}/{k}': v for k, v in zip(keys, values)}
                self.wandb.log_metrics(data, step=step + 1)

    def plot_samples(self, metric_name, plot_sample, source=False, prepend=''):
        if self.local_rank == 0:
            denorm = self.source_denorm if source else self.target_denorm
            label2color = self.source_label2color if source else self.target_label2color

            def process(x, name):
                if x.ndim == 2:
                    return label2color(x).transpose(2, 0, 1).astype(np.uint8)
                elif x.ndim == 3 and 'Confidence' not in name:
                    return (denorm(x) * 255).astype(np.uint8)
                else:
                    return x

            for cl_name, sample in plot_sample.items():
                sid = sample['sid']
                data = [v for k, v in sample.items() if k != 'sid']
                convert = {'img': "RGB", 'label': "Target", 'pred': "Prediction", 'pseudo_lab': "Pseudo Label"}
                caption = [convert[k] if k in convert else k for k, v in sample.items()]
                to_concatenate = [process(v, k) for k, v in sample.items() if k != 'sid']
                if len(to_concatenate) in (4, 5):
                    if len(to_concatenate) == 5:
                        to_concatenate.append(np.zeros_like(data[-1]))
                    w = len(to_concatenate) // 2
                    concat_img_top = np.concatenate(to_concatenate[:w], axis=2)
                    concat_img_bottom = np.concatenate(to_concatenate[w:], axis=2)
                    concat_img = np.concatenate([concat_img_top, concat_img_bottom], axis=1)
                else:
                    concat_img = np.concatenate(to_concatenate, axis=2)

                prt_im = concat_img.transpose((1, 2, 0))
                # self.wandb.log_image(f'{prepend}{metric_name}_{cl_name}_sid{sid}', [prt_im],
                #                      caption=[", ".join(caption[1:])])
                if "add_test" in cl_name:
                    name = cl_name.split("add_test_")[-1]
                else:
                    name = cl_name
                self.wandb.log_image(f'{prepend}{name}_sid{sid}', [prt_im],
                                     caption=[", ".join(caption[1:])])


class CustomWandbLogger(pytorch_lightning.loggers.WandbLogger):

    def __init__(self, args):
        self.args = args
        self.job_name = None

        if args.load:
            ids = args.wandb_id
        else:
            args.wandb_id = ids = wandb.util.generate_id()

        root_dir = Path(__file__).parent.parent.parent

        self.id = ids
        while True:
            try:
                super().__init__(
                    name=self.get_job_name(args),
                    project=self.get_project_name(args.framework, args.source_dataset, args.target_dataset),
                    group=self.get_group_name(args),
                    entity=args.wandb_entity,
                    offline=args.wandb_offline,
                    resume="allow",
                    id=ids,
                    dir=root_dir
                )
                break
            except wandb.errors.error.UsageError:
                print("Retrying")
                time.sleep(5)

    def get_job_name(self, args):

        if self.job_name is not None:
            return self.job_name

        job_name = f"{args.name}_"

        steps_dict = {
            'sr': args.num_source_rounds,
            'se': args.num_source_epochs,
            'r': args.num_rounds,
            'e': args.num_epochs
        }
        for k, v in steps_dict.items():
            if v is not None:
                job_name += f"{k}{v}_"

        if args.framework == 'federated':
            job_name += f"cl{args.clients_per_round}_{args.clients_type}_"

        job_name += f"lr{args.lr}_bs{args.total_batch_size}_rs{args.random_seed}"
        self.job_name = job_name

        return job_name

    def get_dc_pretrained_model(self, run_id, best):
        wandb_path = f'{self._wandb_init["entity"]}/{self._wandb_init["project"]}'

        if self.args.load_dc_target_dataset:
            target_dts_name = self.args.load_dc_target_dataset
        else:
            target_dts_name = self.args.target_dataset[0]
            try:
                for name in self.args.target_dataset[1:]:
                    target_dts_name += f"_{name}"
            except IndexError:
                pass

        try:
            ckpt_path = str(Path('checkpoints')/self.args.framework/self.args.source_dataset/target_dts_name/ \
                 f'pretrained_server_{run_id}{"__best" if best else ""}.ckpt')
            wandb.restore(ckpt_path, run_path=f"{wandb_path}/{run_id}", replace=True, root=os.getcwd())
            if best:
                print(f"Restored best model")
            return torch.load(ckpt_path)
        except ValueError:
            pass


        # trick per gestire il caso precedente dove il modello aveva la path senza le directory
        try:
            ckpt_path = f'pretrained_server_{run_id}_{"_best" if best else ""}.ckpt'
            wandb.restore(ckpt_path, run_path=f"{wandb_path}/{run_id}", replace=True, root=os.getcwd())
            if best:
                print(f"Restored best model")
            return torch.load(ckpt_path)
        except ValueError:
            print(f"!!! Warning: Ckpt missing !!!")


    def get_fda_pretrained_model(self):
        group_name = self._wandb_init["group"]
        wandb_path = f'{self._wandb_init["entity"]}/{self._wandb_init["project"]}'
        keys_to_ignore = ["local_rank", "device_ids", "root", "double_dataset", "quadruple_dataset", "cv2",
                          "server_opt", "algorithm", "server_lr", "server_momentum",
                          "num_rounds", "num_source_rounds", "num_epochs", "num_epochs_phase2", "num_epochs_phase3",
                          "clients_per_round",
                          "test_batch_size", "eval_interval", "test_interval", "server_eval_interval",
                          "server_test_interval",
                          "name", "print_interval", "plot_interval", "save_samples", "load", "print_interval",
                          "wandb_id",
                          "ignore_warnings", "profiler_folder", "ignore_train_metrics",
                          "client_loss", "teacher_step", "load_FDA", "load_FDA_id",
                          "lr_fed", "num_source_epochs_factor_retrain", "lr_factor_server_retrain",
                          "train_source_round_interval", ]
        pretrain_keys = list(set(vars(self.args)) - set(keys_to_ignore))
        new_run_params = {key: value for key, value in vars(self.args).items() if key in pretrain_keys}

        def normalize(v):
            if isinstance(v, str):
                return str(float(v)) if v.isnumeric() else v
            elif isinstance(v, Number):
                return str(float(v))
            elif isinstance(v, dict):
                return [normalize(el) for el in v.items()]
            elif isinstance(v, Iterable):
                return [normalize(el) for el in v]
            else:
                return str(v)

        def dict_eq(d1, d2):
            for k, v in d1.items():
                if not normalize(v) == normalize(d2[k]):
                    return False
            return True

        def restore(run_id, best: bool):
            try:
                ckpt_path = '/'.join(
                    ['checkpoints', self.args.framework, self.args.source_dataset, self.args.target_dataset,
                     f'pretrained_server{"_best" if best else ""}.ckpt'])
                wandb.restore(ckpt_path, run_path=f"{wandb_path}/{run_id}", replace=True, root=os.getcwd())
                if best:
                    print(f"Restored best model")
                return torch.load(ckpt_path)
            except ValueError:
                print(f"!!! Warning: Ckpt missing !!!")


        if self.args.load_FDA_id is not None:
            print(f"Loading previous model from {self.args.load_FDA_id}")
            ckpt = restore(self.args.load_FDA_id, self.args.load_FDA_best)
            if ckpt is not None:
                return ckpt

        for old_run in wandb.Api().runs(path=wandb_path):
            if old_run._attrs["group"] == group_name:
                old_run_params = {key: value for key, value in old_run.config.items() if key in pretrain_keys}
                diff = set(new_run_params.keys()) ^ set(old_run_params.keys())
                if not diff and dict_eq(new_run_params, old_run_params):
                    print(f"same FDA pretraining found, loading previous model from {old_run.id}")
                    ckpt = restore(old_run.id)
                    if ckpt is not None:
                        return ckpt

        print(f"No FDA pre-trained ckpt has been found. Starting pre-training from scratch...")

    @staticmethod
    def get_project_name(framework, source_dataset, target_dataset):
        if source_dataset:
            return f"{framework}_{source_dataset}_{target_dataset}"
        return f"{framework}_{target_dataset}"

    @staticmethod
    def get_group_name(args):
        return args.fw_task

    @rank_zero_only
    def save(self, obj):
        base_dir = str(Path(__file__).parent.parent.parent)
        return wandb.save(obj, base_path=base_dir)

    @staticmethod
    def restore(name, run_path, root):
        return wandb.restore(name=name, run_path=run_path, root=root)
