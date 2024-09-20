import copy
import math
import wandb
import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F
from collections import defaultdict
from matplotlib.pyplot import close as closefig

from tqdm import tqdm

from clients import OracleClient
from utils import (
    HardNegativeMining,
    MeanReduction,
    AdvEntLoss,
    IW_MaxSquareloss,
    SelfTrainingLoss,
    SelfTrainingLossEntropy,
    SelfTrainingLossLovaszEntropy,
    EntropyLoss,
    LovaszLoss,
    get_optimizer_and_scheduler,
    KnowledgeDistillationLoss,
    ProtoClustering,
    ProtoContrastive,
)
from utils.proto import Protos
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.optim import RiemannianAdam
from utils.condition_map import FULL2WEATHER, NUM_CONDITIONS, REVERSE_MAPPING_WEATHER, REVERSE_MAPPING_FULL
from utils.pca import ProtoPCA
from modules.conditional_classifier import ConditionalClassifier
from collections import OrderedDict


class DcClient(OracleClient):
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
        super().__init__(
            args,
            client_id,
            dataset,
            model,
            writer,
            batch_size,
            world_size,
            rank,
            num_gpu,
            device=device,
            test_user=test_user,
        )

        self.source_client = False
        self.criterion, self.reduction = self.__get_criterion_and_reduction_rules()
        (
            self.test_criterion,
            self.test_reduction,
        ) = self.__get_criterion_and_reduction_rules(use_labels=True)
        self.entropy_loss = EntropyLoss(
            lambda_entropy=self.args.lambda_entropy, num_classes=self.args.num_classes
        )
        self.kd_loss = KnowledgeDistillationLoss(
            reduction="mean", alpha=self.args.alpha_kd
        )
        self.teacher_kd_model = None
        self.lambda_kd = self.args.lambda_kd
        self.class_probs = None
        self.feat_channels = (
            1280 if self.args.proto_feats_type == "encoder" else 256
        )  # TODO: get channels based on model (now hardcoded) try self.model.features[-3]
        self.proto = Protos(
            self.args.num_classes,
            self.feat_channels,
            device=model.device,
            exp=self.args.hyperbolic_feats,
        )
        self.manifold = (
            PoincareBall(
                c=Curvature(value=self.args.curvature_c, requires_grad=True)
            ).to(self.model.device)
            if self.args.hyperbolic_feats
            else None
        )
        self.proto_loss = (
            ProtoClustering(
                manifold=self.manifold, distance=self.args.proto_loss_distance
            )
            if self.args.proto_loss_type == "clustering"
            else ProtoContrastive(
                manifold=self.manifold,
                type=self.args.proto_loss_type,
                distance=self.args.proto_loss_distance,
            )
        )
        self.pca = ProtoPCA()
        self.conditional_classifier = (
            ConditionalClassifier(
                num_classes=NUM_CONDITIONS["weather"]-1
            ).to(self.model.device)
            if self.args.conditional_classifier
            else None
        )
        self.optimizer_manifold = None

        # Set the category of the client based on its dataset
        self.category = self.dataset.get_category()
        self.bn_dict = OrderedDict()

    def load_bn_dict(self):
       self.model.load_state_dict(self.bn_dict, strict=False)

    def save_bn_dict(self):
        if self.args.bntype == "fedbn":
            for k, v in self.model.state_dict().items():
                if 'bn' in k:
                    self.bn_dict[k] = copy.deepcopy(v.float())

        elif self.args.bntype == "silobn":
            for k, v in self.model.state_dict().items():
                if 'bn' in k and (("running" in k) or ("num_batches_tracked" in k)):
                    self.bn_dict[k] = copy.deepcopy(v.float())

    def is_source_client(self):
        self.source_client = True
        self.criterion, self.reduction = self.__get_criterion_and_reduction_rules()

    def calc_loss_and_output(self, images, labels, conditions, cur_epoch=None):
        # verifica se e' il server or il client
        # se e' il server, si usa una loss diversa (supervised)
        if self.source_client:
            if (
                cur_epoch is not None
                and self.id == "source_train"
                and 0 <= cur_epoch < self.args.num_source_epochs_warmup
            ):
                conditions = [0 for _ in range(len(conditions))]
            elif (
                cur_epoch is not None
                and self.id == "source_train"
                and self.args.num_source_epochs_warmup == cur_epoch
                and self.args.num_source_epochs_warmup > 0
            ):
                self.model.module.copy_batch()

            out_dict = self.model(
                images, conditions
            )  # out: model output, enc_feats: encoder features, dec_feats: deeplab features
            outputs = out_dict["out"]
            ce_loss = self.reduction(self.criterion(outputs, labels), labels)
            loss_tot = ce_loss
            dict_calc_losses = {"ce_loss": ce_loss}

            if cur_epoch >= self.args.num_source_epochs_warmup:
                # Now the conditional classifier is laoded already trained sao it is not needed to train it
                # if self.args.conditional_classifier:
                #     self.conditional_classifier.train()
                #     _conditions = self.conditional_classifier(images)
                #     gt_conditions = torch.tensor(conditions).to(self.model.device)
                #     conditions_loss = torch.mean(
                #         self.criterion(_conditions, gt_conditions)
                #     )
                #     loss_tot += conditions_loss
                #     dict_calc_losses = {
                #         **dict_calc_losses,
                #         "condition_loss": conditions_loss,
                #     }

                if self.args.lambda_proto_loss_server:
                    feats = (
                        out_dict["enc_feats"]
                        if self.args.proto_feats_type == "encoder"
                        else out_dict["dec_feats"]
                    )
                    if self.args.hyperbolic_feats:
                        self.manifold.train()
                    bprots, bvecs = self.proto(feats, labels, manifold=self.manifold)
                    p_l = self.proto_loss(bvecs, self.proto.protos)

                    fig_pca = self.pca.get_figure(bvecs, self.proto, train=True)
                    if fig_pca is not None:
                        self.writer.wandb.log_image(
                            "prototypes", [wandb.Image(fig_pca)]
                        )
                        closefig(fig_pca)

                    dict_calc_losses = {
                        **dict_calc_losses,
                        "proto_loss": self.args.lambda_proto_loss_server * p_l,
                    }
                    loss_tot += self.args.lambda_proto_loss_server * p_l

            dict_calc_losses = {**dict_calc_losses, "loss_tot": loss_tot}

            return dict_calc_losses, outputs

        # Conditions handling for the client
        if self.args.conditional_classifier:
            with torch.no_grad():
                conditions = self.conditional_classifier(images)
                cond = torch.sum(conditions, dim=0)
                cond = torch.argmax(nn.functional.softmax(cond, dim=0), dim=0)
                conditions = [cond.item()] * len(conditions)
                if self.args.condition_map_type == "full":
                    conditions = [FULL2WEATHER[c] for c in conditions]
                # Sum 1 to each condition to avoid 0
                conditions = [c + 1 for c in conditions]
                device_type = self.dataset.get_category()
                conditions = [f"{device_type}_{REVERSE_MAPPING_WEATHER[c]}" for c in conditions]
                conditions = self.condition_map(conditions)

        def pseudo(outs):
            return outs.max(1)[1]

        kwargs = {}
        if self.args.teacher_step > 0:
            kwargs["imgs"] = images
            if self.args.count_classes_teacher_step != -1 and self.args.weights_lovasz:
                kwargs["weights"] = self.class_probs

        if self.args.model in ("deeplabv3",):
            loss_tot = 0

            if "div" in self.args.client_loss:
                out_dict = self.model(images, conditions)
                outputs = out_dict["out"]
                out_cond = out_dict["condition"]
                with torch.no_grad():
                    outputs_old = (
                        self.teacher_kd_model(images)["out"]
                        if self.args.lambda_kd
                        else None
                    )

                self_loss = self.reduction(
                    self.criterion(outputs, conditions=out_cond, **kwargs),
                    pseudo(outputs),
                ) if not self.args.clients_supervised else nn.CrossEntropyLoss(ignore_index=255)(outputs, labels)
                entropy_loss = self.entropy_loss(outputs)

                dict_calc_losses = {
                    "self_loss": self_loss,
                    "entropy_loss": entropy_loss,
                }

                if outputs_old is not None:
                    pseudo_labels = self.criterion.get_pseudo_lab(
                        outputs_old,
                        imgs=images,
                        conditions=conditions,
                        model=self.teacher_kd_model,
                    )
                    mask = torch.ones(pseudo_labels.shape).double().to(self.device)
                    mask = (
                        torch.where(pseudo_labels != 255, mask, 0.0)
                        if pseudo_labels is not None
                        else None
                    )
                    kd_loss = self.kd_loss(
                        outputs, outputs_old, pred_labels=labels, mask=mask
                    )
                    dict_calc_losses = {
                        **dict_calc_losses,
                        "kd_loss": self.lambda_kd * kd_loss,
                    }
                    loss_tot = (
                        loss_tot + self_loss + entropy_loss + self.lambda_kd * kd_loss
                    )
                else:
                    loss_tot = loss_tot + self_loss + entropy_loss

                if self.args.lambda_proto_loss_clients:
                    with torch.no_grad():
                        ps = self.criterion.get_pseudo_lab(
                            outputs, images, conditions=conditions
                        )
                    # labels = pseudo_labels if outputs_old is not None else pseudo(outputs)
                    feats = (
                        out_dict["enc_feats"]
                        if self.args.proto_feats_type == "encoder"
                        else out_dict["dec_feats"]
                    )
                    bprots, bvecs = self.proto(feats, ps, manifold=self.manifold)
                    fig_pca = self.pca.get_figure(bvecs, self.proto)
                    if fig_pca is not None:
                        self.writer.wandb.log_image(
                            "prototypes", [wandb.Image(fig_pca)]
                        )
                        closefig(fig_pca)
                    p_l = self.proto_loss(bvecs, self.proto.protos)
                    loss_tot = loss_tot + self.args.lambda_proto_loss_clients * p_l
                    dict_calc_losses = {
                        **dict_calc_losses,
                        "proto_loss": self.args.lambda_proto_loss_clients * p_l,
                    }

                dict_calc_losses = {**dict_calc_losses, "loss_tot": loss_tot}

            else:
                outputs = self.model(images, conditions)["out"]
                loss_tot = self.reduction(
                    self.criterion(outputs, conditions=conditions, **kwargs),
                    pseudo(outputs),
                )
                dict_calc_losses = {"loss_tot": loss_tot}

        else:
            raise NotImplementedError

        return dict_calc_losses, outputs

    def calc_test_loss(self, outputs, labels):
        if self.args.model in ("deeplabv3",) and "div" in self.args.client_loss:
            lovasz_loss = self.test_reduction(
                self.test_criterion(outputs, labels), labels
            )
            entropy_loss = self.entropy_loss(outputs)
            return lovasz_loss + entropy_loss
        else:
            return self.test_reduction(self.test_criterion(outputs, labels), labels)

    def plot_condition(self, cur_step):
        return (cur_step + 1) % self.args.plot_interval == 0

    def update_metric(
        self,
        metrics,
        outputs,
        labels,
        conditions=[0],
        conditions_type=None,
        is_test=False,
    ):
        if not self.source_client and not is_test:
            return
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
        if hasattr(self.dataset, "labels2coarse"):
            z = 255 * torch.ones_like(prediction)
            for k, v in self.dataset.labels2coarse.items():
                z[prediction == k] = v
            prediction = z.clone()

        if conditions_type is not None and conditions_type == "full":
            conditions = [FULL2WEATHER[c] for c in conditions]

        prediction = prediction.cpu().numpy()
        metrics.update(labels, prediction, conditions=conditions)

    def count_classes(self):
        class_freqs = defaultdict(lambda: 0)
        class_by_image = defaultdict(lambda: [])
        self.model.eval()
        self.dataset.test = True
        for i, sample in enumerate(
            tqdm(self.loader_full, maxinterval=len(self.dataset))
        ):
            image, _ = self.process_samples(self.loader_full, sample[0][0])
            with torch.no_grad():
                output = self.model(image)["out"]
                pseudo = self.criterion.get_pseudo_lab(output, image)
                np_pseudo = pseudo.cpu().detach().numpy()
                unique, counts = np.unique(np_pseudo, return_counts=True)
                for cl, count in zip(unique, counts):
                    if cl == 255:
                        continue
                    class_freqs[cl] += count
                    class_by_image[cl].append(i)
        class_freqs = {
            k: v / sum(class_freqs.values()) * 100 for k, v in class_freqs.items()
        }
        class_probs = {
            k: math.exp(1 - class_freqs[k]) / self.args.temperature
            for k in class_freqs.keys()
        }
        class_probs = {k: v / sum(class_probs.values()) for k, v in class_probs.items()}
        self.class_probs = class_probs
        labels = [
            "road",
            "sidewalk",
            "building",
            "traffic light",
            "traffic sign",
            "vegetation",
            "sky",
            "person",
            "rider",
            "car",
            "bus",
            "motorcycle",
            "bicycle",
        ]
        cprob_to_print = {labels[k]: round(v, 3) for k, v in class_probs.items()}
        self.writer.write(f"Extracted class probs for client {self}: {cprob_to_print}")
        self.dataset.test = False
        self.model.train()

        return class_probs, class_by_image

    def __exec_epoch(
        self,
        optimizer,
        cur_epoch,
        metric,
        scheduler,
        plot_lr,
        dict_all_iters_losses,
        profiler=None,
        stop_at_step=None,
        r=None,
    ):
        self.model.train()

        if self.args.stop_epoch_at_step != -1:
            stop_at_step = self.args.stop_epoch_at_step

        for cur_step, samples in enumerate(self.loader):
            torch.cuda.empty_cache()

            if stop_at_step is not None and cur_step >= stop_at_step:
                break

            if (
                self.args.teacher_step > 0
                and self.args.teacher_upd_step
                and self.args.centr_fda_ft_uda
            ):
                teacher_model = copy.deepcopy(self.model)
                teacher_model.eval()
                self.set_client_teacher(cur_step, teacher_model)

            if (
                self.args.teacher_kd_step > 0
                and self.args.teacher_kd_upd_step
                and self.args.centr_fda_ft_uda
            ):
                if (
                    cur_step % self.args.teacher_kd_step == 0
                    and self.args.centr_fda_ft_uda
                ):
                    self.writer.write("Setting kd teacher...")
                    teacher_kd_model = copy.deepcopy(self.model)
                    teacher_kd_model.eval()
                    self.teacher_kd_model = teacher_kd_model
                    self.writer.write("Done.")

            images, labels, conditions = self.process_samples(self.loader, samples)

            # Define and map the conditions of the current batch
            str_conditions = conditions
            conditions = self.condition_map(conditions)

            optimizer.zero_grad()

            if self.args.batch_norm_round_0 and r == 0:
                with torch.no_grad():
                    if self.args.mixed_precision:
                        with torch.autocast():
                            dict_calc_losses, outputs = self.calc_loss_and_output(
                                images, labels, conditions, cur_epoch
                            )
                    else:
                        dict_calc_losses, outputs = self.calc_loss_and_output(
                            images, labels, conditions, cur_epoch
                        )
                optimizer.zero_grad()
            else:
                if self.args.mixed_precision:
                    with torch.autocast():
                        dict_calc_losses, outputs = self.calc_loss_and_output(
                            images, labels, conditions, cur_epoch
                        )
                    self.scaler.scale(dict_calc_losses["loss_tot"]).backward()
                else:
                    dict_calc_losses, outputs = self.calc_loss_and_output(
                        images, labels, conditions, cur_epoch
                    )
                    dict_calc_losses["loss_tot"].backward()

            if self.args.fedprox:
                self.handle_grad()
            self.handle_logs(
                cur_step, cur_epoch, dict_calc_losses, metric, scheduler, plot_lr
            )
            self.scaler.step(
                optimizer
            ) if self.args.mixed_precision else optimizer.step()

            if profiler is not None:
                profiler.step()
            if scheduler is not None:
                scheduler.step()

            if self.optimizer_manifold is not None:
                self.optimizer_manifold.step()

            if (
                self.update_metric_condition(cur_epoch)
                and not self.args.ignore_train_metrics
            ):
                if self.condition_map.type == "none":
                    self.condition_map.type = "weather"
                    non_zero_conditions = self.condition_map(str_conditions)
                    self.condition_map.type = "none"
                else:
                    non_zero_conditions = conditions
                self.update_metric(
                    metric,
                    outputs,
                    labels,
                    conditions=non_zero_conditions,
                    conditions_type=self.args.condition_map_type,
                )

            if self.args.mixed_precision:
                self.scaler.update()

            self.update_all_iters_losses(dict_all_iters_losses, dict_calc_losses)

    def run_epoch(
        self,
        cur_epoch,
        optimizer,
        metric=None,
        scheduler=None,
        e_name="EPOCH",
        plot_lr=True,
        stop_at_step=None,
        r=None,
    ):
        dict_all_iters_losses = defaultdict(lambda: 0)
        if hasattr(self.loader.batch_sampler, "set_epoch"):
            self.loader.batch_sampler.set_epoch(cur_epoch)
        else:
            try:
                self.loader.sampler.set_epoch(cur_epoch)
            except AttributeError:
                pass

        if self.profiler_path:
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.profiler_path
                ),
                with_stack=True,
            ) as profiler:
                self.__exec_epoch(
                    optimizer,
                    cur_epoch,
                    metric,
                    scheduler,
                    plot_lr,
                    dict_all_iters_losses,
                    profiler=profiler,
                    stop_at_step=stop_at_step,
                    r=r,
                )
        else:
            self.__exec_epoch(
                optimizer,
                cur_epoch,
                metric,
                scheduler,
                plot_lr,
                dict_all_iters_losses,
                stop_at_step=stop_at_step,
                r=r,
            )

        self.mean_all_iters_losses(dict_all_iters_losses)
        self.writer.write(f"{e_name} {cur_epoch + 1}: ended.")

        return dict_all_iters_losses

    # CLIENT TRAINING
    def train(self, partial_metric, r=None):
        plist = [p for p in self.model.parameters()]
        if self.args.hyperbolic_feats:
            self.manifold.train()
            self.optimizer_manifold = RiemannianAdam(self.manifold.parameters(),
                                                     lr=self.args.manifold_lr, weight_decay=self.args.manifold_wd)
            # plist += [p for p in self.manifold.parameters() if p.requires_grad]
        if self.args.fedprox:
            self.server_model = copy.deepcopy(self.model)
        optimizer, _ = get_optimizer_and_scheduler(
            self.args,
            plist,
            self.max_iter(),
            lr=self.args.lr_fed if not self.source_client else None,
        )

        dict_losses_list = defaultdict(lambda: [])
        self.model.train()

        for epoch in range(self.args.num_epochs):
            dict_all_iters_losses = self.run_epoch(
                epoch, optimizer, metric=partial_metric, r=r
            )
            self._OracleClient__sync_all_iters_losses(
                dict_losses_list, dict_all_iters_losses
            )

        partial_metric.synch(self.device)

        if self.args.fedprox:
            del self.server_model
        if self.args.local_rank == 0:
            return (
                len(self.dataset),
                copy.deepcopy(self.model.state_dict()),
                dict_losses_list,
            )
        return len(self.dataset), copy.deepcopy(self.model.state_dict())

    def __get_criterion_and_reduction_rules(self, use_labels=False):
        loss_choices = {
            "advent": AdvEntLoss,
            "maxsquares": IW_MaxSquareloss,
            "selftrain": SelfTrainingLoss,
            "selftrainentropy": SelfTrainingLossEntropy,
            "lovasz_entropy_joint": SelfTrainingLossLovaszEntropy,
            "lovasz_entropy_div": LovaszLoss,
            "selftrain_div": SelfTrainingLoss,
        }
        loss_fn = (
            nn.CrossEntropyLoss
            if (use_labels or self.source_client)
            else loss_choices[self.args.client_loss]
        )

        shared_kwargs = {"ignore_index": 255, "reduction": "none"}
        if not (use_labels or self.source_client):
            if self.args.client_loss == "lovasz_entropy_div":
                criterion = loss_fn = LovaszLoss(
                    lambda_selftrain=self.args.lambda_selftrain, **shared_kwargs
                )
            elif self.args.client_loss == "selftrain_div":
                criterion = loss_fn = SelfTrainingLoss(
                    lambda_selftrain=self.args.lambda_selftrain, **shared_kwargs
                )
        else:
            criterion = loss_fn(**shared_kwargs)
        if hasattr(loss_fn, "requires_reduction") and not loss_fn.requires_reduction:
            reduction = lambda x, y: x
        else:
            reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        return criterion, reduction

    def test(self, metric, swa=False):
        self.model.eval()

        if swa:
            self.switch_bn_stats_to_test()

        self.dataset.test = True

        tot_loss = 0.0

        conditions_list_predicted = []
        conditions_list_gt = []

        tot_features = None
        tot_labels = None

        with torch.no_grad():
            for i, (images, labels, conditions) in enumerate(self.loader):
                if self.args.stop_epoch_at_step != -1 and i >= self.args.stop_epoch_at_step:
                    break

                if (i + 1) % self.args.print_interval == 0:
                    self.writer.write(f'{self}: {i + 1}/{self.len_loader}, '
                                      f'{round((i + 1) / self.len_loader * 100, 2)}%')

                if self.args.hp_filtered:
                    original_images, images, images_hpf = images
                else:
                    original_images, images = images

                if self.args.hp_filtered:
                    images = self.add_4th_layer(images, images_hpf)

                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                # Perform a center crop to the images and labels of shape 1024x1920
                if self.args.crop_dc_test is True:
                    images = F.center_crop(images, [1024, 1920])
                    labels = F.center_crop(labels, [1024, 1920])

                # Load oracle conditions (to log gt)
                if self.condition_map.type == 'none':
                    self.condition_map.type = 'weather'
                    non_zero_conditions = self.condition_map(conditions)
                    self.condition_map.type = 'none'
                else:
                    non_zero_conditions = self.condition_map(conditions)
                    conditions_list_gt.append(non_zero_conditions)

                # Compute conditions predictions
                if self.args.conditional_classifier:
                    device_type = conditions[0].split("_")[0]
                    conditions = self.conditional_classifier(images)
                    cond = torch.sum(conditions, dim=0)
                    cond = torch.argmax(nn.functional.softmax(cond, dim=0), dim=0)
                    conditions = [cond.item()] * len(conditions)
                    if self.args.condition_map_type == "full":
                        conditions = [FULL2WEATHER[c] for c in conditions]
                    # Sum 1 to each condition to avoid 0
                    conditions = [c + 1 for c in conditions]
                    conditions = [f"{device_type}_{REVERSE_MAPPING_WEATHER[c]}" for c in conditions]
                    conditions = self.condition_map(conditions)
                    conditions_list_predicted.append(conditions)
                else:
                    str_conditions = conditions
                    conditions = self.condition_map(conditions)

                out = self.get_test_output(images, conditions)
                features = out['enc_feats']
                outputs = out['out']

                # Stack the features to the output and the labels
                if self.args.crop_dc_test is True:
                    if tot_features is None:
                        tot_features = features.cpu()
                        tot_labels = labels.cpu()
                    else:
                        tot_features = torch.cat((tot_features, features.cpu()), dim=0)
                        tot_labels = torch.cat((tot_labels, labels.cpu()), dim=0)

                self.update_metric(metric, outputs, labels, conditions=non_zero_conditions, conditions_type=self.args.condition_map_type, is_test=True)

                if outputs.shape != labels.shape:
                    outputs = torch.nn.functional.interpolate(outputs, labels.shape[1:], mode='nearest')
                loss = self.calc_test_loss(outputs, labels)
                tot_loss += loss.item()

                torch.cuda.empty_cache()

            metric.synch(self.device)
            mean_loss = self.manage_tot_test_loss(tot_loss)

        self.dataset.test = False

        if self.args.conditional_classifier:
            conditions_tensor_predicted = torch.Tensor(conditions_list_predicted)
            conditions_tensor_gt = torch.Tensor(conditions_list_gt)
            correct = torch.sum(conditions_tensor_predicted == conditions_tensor_gt)
            acc = correct / conditions_tensor_predicted.shape[0]
            print(f"Conditional classifier accuracies: {acc}")

        return {f'{self}_loss': mean_loss, 'features': tot_features, 'labels': tot_labels}
