import copy
import torch
import torch.nn as nn
import os
import numpy as np
from collections import OrderedDict

from .oracle_server import OracleServer
from torch.utils import data
from utils import StyleAugment, DistributedRCSSampler
from pathlib import Path

from utils.proto import Protos
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import ManifoldTensor
from utils.condition_map import NUM_CONDITIONS
from modules.conditional_classifier import ConditionalClassifier
from collections import deque


class DcServer(OracleServer):

    def __init__(self, args, model, writer, local_rank, lr, momentum, optimizer=None, source_dataset=None):
        super().__init__(model, writer, local_rank, lr, momentum, optimizer, source_dataset=source_dataset)

        self.args = args
        self.feat_channels = 1280 if self.args.proto_feats_type == "encoder" else 256  # TODO: get channels based on model (now hardcoded) try self.model.features[-3]
        self.proto = Protos(self.args.num_classes, self.feat_channels, device=self.model.device, exp=self.args.hyperbolic_feats)
        self.manifold = PoincareBall(c=Curvature(value=self.args.curvature_c, requires_grad=True))\
            .to(self.model.device) if self.args.hyperbolic_feats else None
        self.conditional_classifier = ConditionalClassifier(num_classes=NUM_CONDITIONS[args.condition_map_type])\
            .to(self.model.device) if self.args.conditional_classifier else None

        if self.args.queue_aggregation != -1:
            self.server_models_queue = deque([], self.args.queue_aggregation)

        if self.args.aggr_layers_across_rounds == "enc":
            self.aggr_layer = "backbone"
        elif self.args.aggr_layers_across_rounds == "dec":
            self.aggr_layer = "classifier"
        elif self.args.aggr_layers_across_rounds == "bn":
            self.aggr_layer = "bns"
        else:
            self.aggr_layer = "all"

        self.bn_dict = OrderedDict()

    def save_pretrained_server(self, model_state, best: bool, protos, manifold, conditional_classifier):
        print("Saving best server pretrained ckpt...")
        state = {
            "model_state": model_state,
            "proto": protos,
            "manifold": manifold,
            "conditional_classifier": conditional_classifier,
                 }
        
        target_dts_name = self.args.target_dataset[0]
        try:
            for name in self.args.target_dataset[1:]:
                target_dts_name += f"_{name}"
        except IndexError:
            pass

        ckpt_path = Path(__file__).parent.parent.parent.parent / 'checkpoints' / self.args.framework / \
                    self.args.source_dataset / target_dts_name / \
                    f'pretrained_server_{self.writer.wandb.id}{"__best" if best else ""}.ckpt'
        torch.save(state, str(ckpt_path))
        self.writer.wandb.save(str(ckpt_path))
        print("Done.")

    @staticmethod
    def get_fake_max_scores(improvement, len_cl):
        if improvement:
            return [0] * len_cl
        return [1] * len_cl


    def train_source(self, train_clients, test_clients, train_metric, test_metric, optimizer, scheduler,
                     ret_score, device, test_fn, num_epochs=None, num_steps=None):

        train_clients[0].model.load_state_dict(self.model_params_dict)

        max_scores = [0] * len(test_clients)

        num_source_epochs = self.args.num_source_epochs if num_epochs is None else num_epochs
        if num_steps is not None:
            stop_at_step = num_steps % len(train_clients[0].loader)
            num_source_epochs = num_steps // len(train_clients[0].loader) + min(1, stop_at_step)

        for e in range(0, num_source_epochs):

            self.writer.write(f"EPOCH: {e + 1}/{num_source_epochs}")
            train_clients[0].model.train()

            # TO USE if we want to define a different scheduler:
            # param_groups = optimizer.param_groups
            # param_config = {k: v for k, v in param_groups[0].items() if k != 'params'}
            # if self.args.conditional_classifier:
            #     param_groups.append(
            #         {'params': [p for p in train_clients[0].conditional_classifier.parameters() if p.requires_grad],
            #          **param_config})
            # if self.args.hyperbolic_feats:
            #     param_groups.append(
            #         {'params': [p for p in train_clients[0].manifold.parameters() if p.requires_grad], **param_config})
            # optimizer.param_groups = param_groups

            if self.args.conditional_classifier:
                optimizer.param_groups[0]['params'].extend([p for p in train_clients[0].conditional_classifier.parameters() if p.requires_grad])
            if self.args.hyperbolic_feats:
                optimizer.param_groups[0]['params'].extend([p for p in train_clients[0].manifold.parameters() if p.requires_grad])
            _ = train_clients[0].run_epoch(e, optimizer, train_metric, scheduler, e_name='EPOCH',
                                           stop_at_step=num_steps if (e + 1) == num_source_epochs and num_steps is
                                           not None and num_steps > 0 else None)
            train_metric.synch(device)
            self.writer.plot_metric(e, train_metric, str(train_clients[0]), ret_score)
            train_metric.reset()

            # PERFORM TEST EVERY server_test_interval on: target dataset + list of datasets in self.args.additional_test_datasets
            if (e + 1) % self.args.server_test_interval == 0 or (e + 1) == num_source_epochs:
                max_scores, found_new_max = test_fn(test_clients["target"], test_metric["target_test"], e, 'EPOCH', max_scores, cl_type='target', prepend="target_")

                if self.args.additional_test_datasets:
                    datasets_list = [
                        self.args.additional_test_datasets] if "," not in self.args.additional_test_datasets \
                        else self.args.additional_test_datasets.split(",")
                    for i_dataset in datasets_list:
                        metric_name = f"test_{i_dataset}"
                        _, _ = test_fn(test_clients[i_dataset], test_metric[metric_name], e, 'EPOCH', max_scores, cl_type=i_dataset)

                protos = train_clients[0].proto.protos
                manifold = train_clients[0].manifold
                conditional_classifier = train_clients[0].conditional_classifier
                if found_new_max:
                    self.save_pretrained_server(train_clients[0].model.state_dict(), best=True,
                                                protos=protos, manifold=manifold,
                                                conditional_classifier=conditional_classifier)


        # TEST ON SOURCE at the end of pretraining
        # _, _ = test_fn(test_clients["source"], test_metric["source_test"], e, 'EPOCH', max_scores,
        #                                     cl_type='source')

        self.model_params_dict = copy.deepcopy(train_clients[0].model.state_dict())
        self.proto.protos = copy.deepcopy(train_clients[0].proto.protos)
        self.manifold = copy.deepcopy(train_clients[0].manifold)
        self.conditional_classifier = copy.deepcopy(train_clients[0].conditional_classifier)
        self.pretrain_model_params_dict = copy.deepcopy(train_clients[0].model.state_dict())
        return max_scores

    def count_classes(self):
        self.writer.write("Extracting pseudo labels stats...")
        for i, c in enumerate(self.selected_clients):
            self.writer.write(f"client {i + 1}/{len(self.selected_clients)}: {c}...")
            class_probs, class_by_image = c.count_classes()
            c.loader = data.DataLoader(c.dataset, batch_size=c.batch_size, worker_init_fn=c.seed_worker,
                                       sampler=DistributedRCSSampler(c.dataset, num_replicas=c.world_size,
                                                                     rank=c.rank, class_probs=class_probs,
                                                                     class_by_image=class_by_image,
                                                                     seed=self.args.random_seed),
                                       num_workers=4*c.num_gpu, drop_last=True, pin_memory=True)
        self.writer.write("Done.")

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

    def train_clients(self, partial_metric=None, r=None, metrics=None, target_test_client=None, test_interval=None,
                      ret_score='Mean IoU'):

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        clients = self.selected_clients
        losses = {}

        for i, c in enumerate(clients):

            self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c}")

            c.model.load_state_dict(self.model_params_dict)

            if self.args.bntype:
                if c.bn_dict:
                    c.load_bn_dict()

            out = c.train(partial_metric, r=r)

            if self.args.bntype:
                c.save_bn_dict()

            if self.local_rank == 0:
                num_samples, update, dict_losses_list = out
                losses[c.id] = {'loss': dict_losses_list, 'num_samples': num_samples}
            else:
                num_samples, update = out

            if self.optimizer is not None:
                update = self._compute_client_delta(update)

            self.updates.append((num_samples, update))

        if self.local_rank == 0:
            return losses
        return None


    def update_clients_lr(self, lr):
        for c in self.selected_clients:
            c.lr_fed = lr

    def update_manifold(self):
        total_weight = 0.
        base = OrderedDict()
        for (client_samples, _), client in zip(self.updates, self.selected_clients):
            for key, p in client.manifold.named_parameters():
                total_weight += client_samples
                if key in base:
                    base[key] += client_samples * p.data.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * p.data.type(torch.FloatTensor)
        for key, value in base.items():
            if total_weight != 0:
                base[key] = value.to(self.local_rank) / total_weight
        self.manifold.load_state_dict(base)


    def update_proto(self):
        total_weight = {c: 0. for c in range(self.args.num_classes)} #0.
        if not self.args.hyperbolic_feats:
            global_proto = {c: torch.zeros(self.selected_clients[0].feat_channels, 1, device=self.model.device) for c in range(self.args.num_classes)}
        else:
            global_proto = {c: torch.empty(self.selected_clients[0].feat_channels, 0, device=self.model.device) for c in
                            range(self.args.num_classes)}
            clients_samples = {c: torch.empty(0, device=self.model.device) for c in range(self.args.num_classes)}
        for client in self.selected_clients:
            # total_weights += client_samples
            local_proto = client.proto.protos
            valid = [c for c, v in client.proto.num_samples.items() if v > 0]
            for c in valid:
                client_samples = client.proto.num_samples[c]
                total_weight[c] += client_samples
                if not self.args.hyperbolic_feats:
                    global_proto[c] += local_proto[c] * client_samples
                else:
                    global_proto[c] = torch.cat([global_proto[c], local_proto[c]], dim=1)
                    clients_samples[c] = torch.cat([clients_samples[c], torch.tensor([client_samples], device=self.model.device)], dim=0)

        if not self.args.hyperbolic_feats:
            global_proto = {c: v / total_weight[c] for c, v in global_proto.items() if total_weight[c] > 0}
        else:
            clients_samples = {c: v / total_weight[c] for c, v in clients_samples.items() if total_weight[c] > 0}
            #self.manifold = self.manifold if self.manifold is not None else self.selected_clients[0].manifold
            global_proto = {c: self.manifold.midpoint(ManifoldTensor(torch.t(global_proto[c]), manifold=self.manifold, man_dim=1), keepdim=True, w=clients_samples[c].unsqueeze(1))
            .tensor.view(-1, 1).detach() for c, v in global_proto.items() if total_weight[c] > 0}

        if self.args.proto_update_ema == 0:
                self.proto.protos = global_proto
        else:
            for c in global_proto:
                if total_weight[c] != 0: # update only if clients modified the local prototypes
                    if self.proto.protos[c].shape[1] == 0:
                        self.proto.protos[c] = global_proto[c]
                    else:
                        self.proto.protos[c] = self.args.proto_update_ema * self.proto.protos[c] + (1 - self.args.proto_update_ema) * global_proto[c]


    def update_model(self):

        averaged_sol_n = self._aggregation()

        if self.args.bntype:
            averaged_sol_n.update(self.bn_dict)

        if self.args.queue_aggregation != -1:
            self.server_models_queue.append(self.model_params_dict)
            n_items = len(self.server_models_queue)
            for k in averaged_sol_n.keys():
                if self.aggr_layer == "all" or self.aggr_layer not in k:
                    for j in range(n_items):
                        averaged_sol_n[k] += self.server_models_queue[j][k]
                    averaged_sol_n[k] /= n_items + 1

        elif self.args.ema_aggregation:
            for k in averaged_sol_n.keys():
                averaged_sol_n[k] = self.model_params_dict[k] * self.args.ema_aggregation + \
                                    averaged_sol_n[k] * (1 - self.args.ema_aggregation)

        if self.optimizer is not None:
            self._server_opt(averaged_sol_n)
            self.total_grad = self._get_model_total_grad()
        else:
            self.model.load_state_dict(averaged_sol_n)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.updates = []
