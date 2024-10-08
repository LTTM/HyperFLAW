import os
import torch
import random
import numpy as np
from torch.utils import data
from torch.cuda.amp import GradScaler
from torch import nn, distributed
from utils import HardNegativeMining, MeanReduction
from dataset import SingleBatchSampler, MixedBatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from utils import ConditionMap
from utils.condition_map import FULL2WEATHER


class Client:
    def __init__(
        self,
        args,
        client_id,
        dataset,
        model,
        writer,
        batch_size,
        world_size,
        rank,
        num_gpu,
        device=None,
        test_user=False,
    ):
        self.args = args
        self.id = client_id
        self.dataset = dataset
        self.model = model
        self.writer = writer
        self.batch_size = batch_size
        self.device = device
        self.test_user = test_user
        self.world_size = world_size
        self.rank = rank
        self.condition_map = ConditionMap(args.condition_map_type)

        if os.name == "nt":
            num_gpu = 0
        self.num_gpu = num_gpu

        # Set random seed
        if args.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(args.random_seed)
        else:
            g = None

        # Set data loader based on which dataset is used
        if not self.dataset.weather_aware:
            # b_sampler = BatchSampler(self.dataset, batch_size=self.batch_size, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
            # b_sampler_full = DistributedSampler(self.dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
            b_sampler = BatchSampler(sampler=RandomSampler(self.dataset, generator=g), batch_size=self.batch_size, drop_last=True)
            b_sampler_full = BatchSampler(sampler=SequentialSampler(self.dataset), batch_size=self.batch_size, drop_last=False)
        elif self.dataset.category in ["mixed"]:
            b_sampler = MixedBatchSampler(
                    classes=self.dataset.get_classes(),
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    generator=g,
                    fuse_strategy=self.args.dts_fuse_strategy,
                )
            b_sampler_full = MixedBatchSampler(
                    classes=self.dataset.get_classes(),
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                    fuse_strategy=self.args.dts_fuse_strategy,
                )
        elif self.dataset.category in ["car", "drone"]:
            b_sampler = SingleBatchSampler(
                    classes=self.dataset.get_classes(),
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    generator=g,
                    fuse_strategy="og" if self.args.dts_fuse_strategy == "og" else "alt",
                )
            b_sampler_full = SingleBatchSampler(
                    classes=self.dataset.get_classes(),
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                    fuse_strategy="og" if self.args.dts_fuse_strategy == "og" else "alt",
                )
        else:
            raise ValueError(
                f"Invalid dataset, the given one is {self.dataset}, the has no compatible dataloader"
            )

        self.loader = data.DataLoader(
            self.dataset,
            worker_init_fn=self.seed_worker,
            batch_sampler=b_sampler,
            num_workers=4 * num_gpu,
            pin_memory=True,
            generator=g,
        )
        self.loader_full = data.DataLoader(
            self.dataset,
            worker_init_fn=self.seed_worker,
            batch_sampler=b_sampler_full,
            num_workers=4 * num_gpu,
            pin_memory=True,
            generator=g,
        )

        self.criterion, self.reduction = self.__get_criterion_and_reduction_rules()

        if self.args.mixed_precision:
            self.scaler = GradScaler()

        self.profiler_path = (
            os.path.join("profiler", self.args.profiler_folder)
            if self.args.profiler_folder
            else None
        )

    @staticmethod
    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __get_criterion_and_reduction_rules(self):
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
        reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        return criterion, reduction

    def update_metric(self, metric, outputs, labels, conditions=[0], conditions_type=None, **kwargs):
        _, prediction = outputs.max(dim=1)
        if prediction.shape != labels.shape:
            prediction = (
                nn.functional.interpolate(
                    prediction.unsqueeze(0).double(), labels.shape[1:], mode="nearest"
                )
                .squeeze(0)
                .long()
            )
        labels = labels.cpu().numpy()

        # If the used dataset is FLYAWARER map the prediction
        if hasattr(self.dataset, "city2coarse"):
            # import matplotlib
            # matplotlib.use('webagg')
            # from matplotlib import pyplot as plt
            # fig, axs = plt.subplots(2,2)
            z = torch.zeros_like(prediction)
            for i, city in enumerate(self.dataset.city2coarse):
                for c in city:
                    z[prediction == c] = i
            # axs[0,0].imshow(prediction[0].cpu().numpy())
            # axs[0,1].imshow(z[0].cpu().numpy())
            prediction = z.clone()
            z *= 0
            for k, v in self.dataset.labels2train.items():
                z[prediction == k] = v
            prediction = z.clone()
            # axs[1,0].imshow(labels[0])
            # axs[1,1].imshow(labels2[0])
            # plt.show()
            # labels = labels2.copy()


        if conditions_type is not None and conditions_type == "full":
            conditions = [FULL2WEATHER[c] for c in conditions]

        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction, conditions=conditions)

    def calc_loss_and_output(self, images, labels, conditions):
        if self.args.model in ("deeplabv3",):
            out_dict = self.model(images, conditions)  # out: model output, enc_feats: encoder features, dec_feats: deeplab features
            outputs = out_dict["out"]
            loss_tot = self.reduction(self.criterion(outputs, labels), labels)
            dict_calc_losses = {"loss_tot": loss_tot}

        else:
            raise NotImplementedError

        return dict_calc_losses, outputs

    def get_test_output(self, images, conditions=None):
        if self.args.model == "deeplabv3":
            return self.model(images, conditions)

        return self.model(images)

    def calc_test_loss(self, outputs, labels):
        return self.reduction(self.criterion(outputs, labels), labels)

    def manage_tot_test_loss(self, tot_loss):
        tot_loss = torch.tensor(tot_loss).to(self.device)
        distributed.reduce(tot_loss, dst=0)
        return tot_loss / distributed.get_world_size() / len(self.loader)

    def run_epoch(self, cur_epoch, metrics, optimizer, scheduler=None):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.id

    @property
    def num_samples(self):
        return len(self.dataset)

    @property
    def len_loader(self):
        return len(self.loader)
