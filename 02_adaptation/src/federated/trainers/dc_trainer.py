import torch
import copy
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import numpy as np
import wandb
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE

from metrics import StreamSegMetrics
from federated.trainers.oracle_trainer import OracleTrainer as OracleTrainerFed
from utils import dynamic_import, get_optimizer_and_scheduler, schedule_cycling_lr
from centralized.trainers.oracle_trainer import OracleTrainer as OracleTrainerCent
from modules import ConditionalClassifier

from utils.condition_map import NUM_CONDITIONS
from modules import ConditionalClassifier
from torch.nn import functional as F
import pandas as pd

class DcTrainer(OracleTrainerFed, OracleTrainerCent):

    def __init__(self, args, writer, device, rank, world_size):

        super().__init__(args, writer, device, rank, world_size)

        self.teacher_kd_model = None
        self.swa_teacher_model = None

        if self.args.swa_teacher_start != -1:
            self.swa_teacher_n = 0

        for clients in (self.source_train_clients, self.source_test_clients):
            for c in clients:
                c.is_source_client()

    def server_setup(self):
        server_class = dynamic_import(self.args.framework, self.args.fw_task, 'server')
        server = server_class(self.args, self.model, self.writer, self.args.local_rank, self.args.server_lr,
                              self.args.server_momentum, self.args.server_opt,
                              self.source_train_clients[0].dataset)
        return server

    def set_metrics(self, writer, num_classes):
        writer.write('Setting up metrics...')
        metrics = {
            'source_train': StreamSegMetrics(num_classes, 'source_train'),
            'target_train': StreamSegMetrics(num_classes, 'target_train'),
            'source_test': StreamSegMetrics(num_classes, 'source_test'),
            'target_test': StreamSegMetrics(num_classes, 'target_test'),
            'target_eval': StreamSegMetrics(num_classes, 'target_eval')
        }
        if self.args.additional_test_datasets:
            for i_dataset in self.additional_test_datasets_list:
                metrics = {**metrics, f"test_{i_dataset}": StreamSegMetrics(num_classes, f"test_{i_dataset}")}

        writer.write('Done.')
        return metrics

    def get_optimizer_and_scheduler(self, lr=None):
        return get_optimizer_and_scheduler(self.args, self.model.parameters(), self.max_iter(), lr=lr)

    def max_iter(self):
        return self.args.num_source_epochs * self.source_train_clients[0].len_loader

    def update_swa_teacher_model(self, alpha, model, swa_teacher_model=None):
        swa_teacher_model = self.swa_teacher_model if swa_teacher_model is None else swa_teacher_model
        for param1, param2 in zip(swa_teacher_model.parameters(), model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha
        if swa_teacher_model is not None:
            return swa_teacher_model

    def set_client_teacher(self, r, model):

        if r % self.args.teacher_step == 0 and not self.args.teacher_upd_step:

            self.writer.write(f"round {r}, setting new teacher...")

            if self.args.teacher_kd_step == -1 and self.args.lambda_kd > 0:
                self.writer.write(f"Setting kd teacher too...")

            if self.args.swa_teacher_start != -1 and r + 1 > self.args.swa_teacher_start and \
                    ((r - self.args.swa_teacher_start) // self.args.teacher_step) % self.args.swa_teacher_c == 0:
                self.writer.write(f"Number of models: {self.swa_teacher_n}")
                self.update_swa_teacher_model(1.0 / (self.swa_teacher_n + 1), model)
                self.swa_teacher_n += 1

            if self.swa_teacher_model is not None:
                model = self.swa_teacher_model

            for c in self.target_train_clients:
                if self.args.teacher_kd_step == -1:
                    c.teacher_kd_model = model
                if hasattr(c.criterion, 'set_teacher'):
                    if self.args.fw_task == "ladd" and self.server.clusters_models:
                        model.load_state_dict(self.server.clusters_models[c.cluster_id])
                        c.criterion.set_teacher(model)
                    else:
                        c.criterion.set_teacher(model)
                else:
                    break

            if self.args.count_classes_teacher_step != -1:
                if (r // self.args.teacher_step) % self.args.count_classes_teacher_step == 0:
                    self.writer.write("Updating sampling probs...")
                    self.server.count_classes()
                    self.writer.write("Done.")

            self.writer.write(f"Done.")

    def set_client_kd_teacher(self, r, model):
        if r % self.args.teacher_kd_step == 0:
            self.writer.write(f"Setting kd teacher...")
            for c in self.target_train_clients:
                c.teacher_kd_model = model
            self.writer.write(f"Done.")

    def save_pretrained_server(self):
        self.writer.write("Saving server pretrained ckpt...")
        state = {
            "model_state": self.server.model_params_dict,
            "proto": self.server.proto.protos,
            "manifold": self.server.manifold,
            "conditional_classifier": self.server.conditional_classifier,
        }

        target_dts_name = self.args.target_dataset[0]
        try:
            for name in self.args.target_dataset[1:]:
                target_dts_name += f"_{name}"
        except IndexError:
            pass

        ckpt_path = Path(__file__).parent.parent.parent.parent / 'checkpoints' / self.args.framework / \
                    self.args.source_dataset / target_dts_name / \
                    f'pretrained_server_{self.writer.wandb.id}.ckpt'
        torch.save(state, str(ckpt_path))
        self.writer.wandb.save(str(ckpt_path))
        print("Done")

    def save_model(self, step, optimizer=None, scheduler=None):

        state = {
            "step": step,
            "model_state": self.server.model_params_dict,
            "proto": self.server.proto.protos,
            "manifold": self.server.manifold,
        }

        if self.server.optimizer is not None:
            state["server_optimizer_state"] = self.server.optimizer.state_dict()
        torch.save(state, self.ckpt_path)
        self.writer.wandb.save(self.ckpt_path)

    def load_from_checkpoint(self):
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.server.model_params_dict = deepcopy(self.model.state_dict())
        self.writer.write(f"[!] Model restored from step {self.checkpoint_step}.")
        if "server_optimizer_state" in self.checkpoint.keys():
            self.server.optimizer.load_state_dict(self.checkpoint["server_optimizer_state"])
            self.writer.write(f"[!] Server optimizer restored.")

    def setup_swa_teacher_model(self, swa_ckpt=None):
        self.swa_teacher_model = deepcopy(self.model)
        if swa_ckpt is not None:
            self.swa_teacher_model.load_state_dict(swa_ckpt)

    @staticmethod
    def extract_subtensor_with_occurrences(tensor, num_occurrences):
        unique_values = torch.unique(tensor)
        subtensor_list = []
        index_list = []

        for value in unique_values:
            indices = torch.nonzero(tensor == value).flatten()
            selected_indices = indices[:num_occurrences]
            subtensor = tensor[selected_indices]
            subtensor_list.append(subtensor)
            index_list.append(selected_indices)

        result = torch.cat(subtensor_list)
        indices = torch.cat(index_list)
        return result, indices
    
    @staticmethod
    def filter_data(features, labels):
        filtered_bottleneck = []
        labels_bottleneck = []
        for i in range(features.size(0)):  # Iterate over the batch dimension
            for j in range(features.size(2)):  # Iterate over the spatial dimensions
                for k in range(features.size(3)):
                    # Calculate the corresponding indices in the ground truth segmentation
                    label_row_start = j * 32
                    label_row_end = (j + 1) * 32
                    label_col_start = k * 32
                    label_col_end = (k + 1) * 32

                    # Get the corresponding region in the ground truth segmentation
                    region_labels = labels[i, label_row_start:label_row_end, label_col_start:label_col_end]

                    # Step 4: Check for mixed class labels
                    unique_labels = torch.unique(region_labels)
                    if len(unique_labels) == 1:
                        # Only one unique label, meaning consistent class within the region
                        filtered_bottleneck.append(features[i, :, j, k])
                        labels_bottleneck.append(unique_labels)

        # Convert the filtered bottleneck features to a tensor
        return torch.stack(filtered_bottleneck), torch.stack(labels_bottleneck)

    def compute_tsne(self, data, dataset_names, perplexity=30, init="pca", learning_rate='auto', n_iter=1000, random_state=42, num_occurrences=500, legend=False):
        tsne = TSNE(perplexity=perplexity,
                    init=init,
                    learning_rate=learning_rate,
                    n_iter=n_iter,
                    random_state=random_state)
        
        colors = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                 (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                 (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                 (0, 0, 0)]) / 255
        
        colors_dict = {k: v for k,v in enumerate(colors)}
    
        tsnne_dict = {'X': None, 'Y': None, 'label': None, 'dts': None}
        for i, d in enumerate(data.values()):
            dataset_name = dataset_names[i]

            features = d['features'].detach().cpu()
            labels = d['labels'].detach().cpu()
            # labels = F.interpolate(d['labels'].unsqueeze(1).float(), features.shape[2:], mode='nearest').squeeze(1).long()

            # Filter the data
            features, labels = self.filter_data(features, labels)

            features = features.view(-1, features.shape[1]).cpu()
            labels = labels.view(-1)
            features_nobgr = features[labels != -1, :]
            labels = labels[labels != -1]

            features_nobgr = F.normalize(features_nobgr, p=2, dim=0)

            labels, indices = self.extract_subtensor_with_occurrences(labels, num_occurrences)
            features_nobgr = features_nobgr[indices, :]
            tsne_data = tsne.fit_transform(features_nobgr.cpu())

            # Concatenate the data to the tsne_dict
            if tsnne_dict['X'] is None:
                tsnne_dict['X'] = tsne_data[:, 0]
                tsnne_dict['Y'] = tsne_data[:, 1]
                tsnne_dict['label'] = labels.cpu()
                tsnne_dict['dts'] = [dataset_name] * len(labels)
            else:
                tsnne_dict['X'] = np.concatenate((tsnne_dict['X'], tsne_data[:, 0]))
                tsnne_dict['Y'] = np.concatenate((tsnne_dict['Y'], tsne_data[:, 1]))
                tsnne_dict['label'] = np.concatenate((tsnne_dict['label'], labels.cpu()))
                tsnne_dict['dts'] = np.concatenate((tsnne_dict['dts'], [dataset_name] * len(labels)))

        df = pd.DataFrame(tsnne_dict)
        df = df[df.label != 255] # remove the void label

        out_path = Path(__file__).parent.parent.parent.parent / 'tsne.pdf'

        matplotlib.rcParams['lines.markersize'] = 4  # <---- set markersize here
        plt.figure()
        ax = sns.scatterplot(data=df,
                            x='X', 
                            y='Y',
                            legend=legend,
                            palette=colors_dict,
                            hue='label',
                            style='dts',
                            )
        ax.set(xticklabels=[], yticklabels=[])  # remove the tick labels
        ax.tick_params(bottom=False)  # remove the ticks
        ax.tick_params(labelleft=False, left=False)
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def test(self, test_clients, metric, step, step_type, max_scores, cl_type='target', prepend=''):

        cl_type_cat = cl_type
        prepend_cat = prepend
        mean_max_score = sum(max_scores) / len(max_scores)

        if self.server is not None:
            if self.server.swa_model is not None:
                tmp_model = copy.deepcopy(test_clients[0].model.state_dict())

        scores, outs  = self.perform_test(metric, test_clients, step)

        if self.args.evaluation and self.args.crop_dc_test:
            self.compute_tsne(outs, ['acdc', 'flyawarer', 'flyawarerxl'], perplexity=20, num_occurrences=150, legend=False)

        # metric = metric[0] if isinstance(metric, list) else metric

        ref_scores = [s[self.ret_score] for s in scores]
        mean_score = sum(ref_scores) / len(scores)

        if mean_score > mean_max_score:
            self.writer.write(f"New best result found at {step_type.lower()} {step + 1}")

        for i, score in enumerate(scores):
            # ref_client = test_clients[i]
            self.writer.write(f"Test {self.ret_score.lower()} at {step_type.lower()} {step + 1}: "
                              f"{round(score[self.ret_score] * 100, 3)}%")

        for ref_client in test_clients:  # # when we have 2 target dataset, instead of overriding the same image
            if (self.test_plot_counter >= 0 and self.test_plot_counter % 2 == 0) or mean_score > mean_max_score:
                if self.args.save_samples > 0:
                    if len(test_clients) > 1:
                        cl_type_cat = cl_type + "_" + ref_client.dataset.category
                        prepend_cat = prepend + ref_client.dataset.category + "_"
                    plot_samples = self.get_plot_samples(ref_client, cl_type=cl_type_cat)
                    for plot_sample in plot_samples:
                        self.writer.plot_samples("", plot_sample, source=cl_type_cat == 'source', prepend=prepend_cat)
                    self.test_plot_counter = 1
            elif self.test_plot_counter >= 0:
                self.test_plot_counter += 1

        if self.server is not None:
            if self.server.swa_model is not None:
                test_clients[0].model.load_state_dict(tmp_model)

        if mean_score > mean_max_score:
            return ref_scores, True
        return max_scores, False

    def bar_plot_selected_clients(self):
        w_conditions = ["rain", "clear", "night", "fog"]
        clients_weather_dict = {}
        client_type_list = []
        for ic, client in enumerate(self.server.selected_clients):
            client_conditions = []
            for im, (_, _, conditions) in enumerate(client.loader):
                client_conditions.append([c.split("_")[-1] for c in conditions])
            weather_client = list(itertools.chain.from_iterable(client_conditions))
            clients_weather_dict[ic] = dict(Counter(weather_client))
            for key_b in clients_weather_dict[ic].keys():
                clients_weather_dict[ic][key_b] /= len(weather_client)
            for con in w_conditions:
                if con not in clients_weather_dict[ic].keys():
                    clients_weather_dict[ic][con] = 0
            client_type_list.append(client.dataset.category[0] + " " + client.id.split("_")[0][1:])
            print(f"len dataset {len(client.loader.dataset)}")
        clients_ids = list(clients_weather_dict.keys())
        data_array = np.array(
            [[clients_weather_dict[element][con] for element in
             clients_ids] for con in w_conditions])
        # Plotting the stacked bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [(0.937, 0.486, 0.556),
                  (0.980, 0.909, 0.878),
                  (0.713, 0.886, 0.827),
                  (0.847, 0.654, 0.694)]
        bars = []
        bottom = np.zeros(len(clients_ids))
        for i in range(len(data_array)):
            bar = ax.bar(clients_ids, data_array[i], bottom=bottom, color=colors[i])
            bars.append(bar)
            bottom += data_array[i]
        ax.set_xlabel('clients')
        ax.set_ylabel('Weather fraction')
        ax.set_title('Weather Distribution for the Randomly Selected Clients')
        ax.set_xticks(clients_ids)
        ax.set_xticklabels(client_type_list)
        ax.legend(bars, w_conditions)
        self.writer.wandb.log_image("weather_fraction", [wandb.Image(plt)])
        plt.close(fig)

    def convert_from_ladd(self, ckpt):
        old_dict = self.model.state_dict()
        for k, v in ckpt.items():
            k = k.replace("module.classifier.0.", "module.classifier.aspp.")
            k = k.replace("module.classifier.1.", "module.classifier.conv1.")
            k = k.replace("module.classifier.2.", "module.classifier.bn.")
            k = k.replace("module.classifier.4.", "module.classifier.conv2.")

            if k in old_dict:
                old_dict[k] = v
            else:
                pieces = k.split(".")
                for i in range(NUM_CONDITIONS[self.args.condition_map_type]):
                    nk = ".".join(pieces[:-1] + ["bns", str(i)] + pieces[-1:])
                    if nk in old_dict:
                        old_dict[nk] = v
                    else:
                        raise ValueError(f"Unknown Key {k}")

        return old_dict

    def convert_from_noweather(self, ckpt):
        old_dict = self.model.state_dict()
        for k, v in ckpt.items():
            if 'bns' in k:
                pieces = k.split("bns.")
                for i in range(NUM_CONDITIONS[self.args.condition_map_type]):
                    nk = pieces[0] + "bns." + str(i) + pieces[1][1:]
                    if k in old_dict:
                        old_dict[nk] = v
                    else:
                        raise ValueError(f"Unknown Key {k}")
            else:
                old_dict[k] = v

        return old_dict
    
    def train(self):

        max_scores = [0] * len(self.target_test_clients)

        test_clients = {
            "source": self.source_test_clients,
            "target": self.target_test_clients,
        }
        for add_client in self.add_test_clients:
            d_name = add_client.id.split("add_test_")[-1]
            test_clients = {**test_clients, d_name: [add_client]}

        pretrained_ckpt = None
        if self.ckpt_round == 0:

            if self.args.save_samples > 0:
                plot_samples = self.get_plot_samples(self.source_train_clients[0], cl_type="source_train")
                for plot_sample in plot_samples:
                    self.writer.plot_samples("", plot_sample, source=True)

            if self.args.load_dc_id is not None:
                self.writer.write(f"Loading ckpt from wandb id: {self.args.load_dc_id}")
                pretrained_ckpt = self.writer.wandb.get_dc_pretrained_model(run_id=self.args.load_dc_id, best=self.args.load_dc_best)
                self.server.model_params_dict = deepcopy(pretrained_ckpt['model_state'])
                if 'proto' in pretrained_ckpt:
                    self.server.proto.protos = pretrained_ckpt['proto']  # TODO: verify it is correct
                if 'manifold' in pretrained_ckpt and pretrained_ckpt['manifold'] is not None:
                    self.server.manifold = deepcopy(pretrained_ckpt['manifold'])
                # if 'conditional_classifier' in pretrained_ckpt and pretrained_ckpt['conditional_classifier'] is not None:
                #     self.server.conditional_classifier = deepcopy(pretrained_ckpt['conditional_classifier'])
                if self.args.conditional_classifier:
                    try:
                        self.server.conditional_classifier.load_state_dict(torch.load(Path(__file__).parent.parent.parent.parent / Path('data/classifier_model/model.pth')))
                    except RuntimeError:
                        self.server.conditional_classifier = ConditionalClassifier(num_classes=4).to(self.device)
                        self.server.conditional_classifier.load_state_dict(torch.load(Path(__file__).parent.parent.parent.parent / Path('data/classifier_model/model.pth')))
                self.target_test_clients[0].model.load_state_dict(self.server.model_params_dict)
                self.writer.write(f'Done')
            elif self.args.pretrained_ckpt_cr is not None:
                self.writer.write(f'Ckpt found, loading...')
                pretrained_ckpt = torch.load(self.args.pretrained_ckpt_cr)
                if self.args.ladd_ckpt:
                    pretrained_ckpt['model_state'] = self.convert_from_ladd(pretrained_ckpt['model_state'])
                elif self.args.load_from_noweather:
                    pretrained_ckpt['model_state'] = self.convert_from_noweather(pretrained_ckpt['model_state'])

                self.server.model_params_dict = pretrained_ckpt['model_state']
                if 'proto' in pretrained_ckpt:
                    self.server.proto.protos = pretrained_ckpt['proto']
                if 'manifold' in pretrained_ckpt and pretrained_ckpt['manifold'] is not None:
                    self.server.manifold = deepcopy(pretrained_ckpt['manifold'])
                # if 'conditional_classifier' in pretrained_ckpt and pretrained_ckpt['conditional_classifier'] is not None:
                #     self.server.conditional_classifier = deepcopy(pretrained_ckpt['conditional_classifier'])
                if self.args.conditional_classifier:
                    try:
                        self.server.conditional_classifier.load_state_dict(torch.load(Path(__file__).parent.parent.parent.parent / Path('data/classifier_model/model.pth')))
                    except RuntimeError:
                        self.server.conditional_classifier = ConditionalClassifier(num_classes=4).to(self.device)
                        self.server.conditional_classifier.load_state_dict(torch.load(Path(__file__).parent.parent.parent.parent / Path('data/classifier_model/model.pth')))
                self.target_test_clients[0].model.load_state_dict(self.server.model_params_dict)
                self.writer.write(f'Done')
            else:
                if not self.args.skip_pretraining:
                    self.writer.write('Traning on server data...')
                    max_scores = self.server.train_source(train_clients=self.source_train_clients,
                                                          test_clients=test_clients,
                                                          train_metric=self.metrics['source_train'],
                                                          test_metric=self.metrics,
                                                          optimizer=self.optimizer, scheduler=self.scheduler,
                                                          ret_score=self.ret_score,
                                                          device=self.device, test_fn=self.test,
                                                          num_epochs=self.args.num_source_epochs)
                    self.save_pretrained_server()
                    self.writer.write('Done')
            if self.args.save_samples > 0:
                plot_samples = self.get_plot_samples(self.source_test_clients[0], cl_type="source")
                for plot_sample in plot_samples:
                    self.writer.plot_samples("", plot_sample, source=True)
            self.server.model.load_state_dict(self.server.model_params_dict)

        if self.args.num_rounds == 0:
            return max_scores

        max_scores = [0] * len(self.target_test_clients)

        if self.args.conditional_classifier is not None and self.args.conditional_classifier:
            self.server.conditional_classifier.training = False
            self.server.conditional_classifier.eval()
            for client in self.target_test_clients:
                client.conditional_classifier = self.server.conditional_classifier

        if self.args.evaluation and pretrained_ckpt:
            max_scores, _ = self.test(self.target_test_clients, self.metrics['target_test'], 0, 'ROUND', max_scores,
                                          cl_type='target', prepend="target_")

            return max_scores

        if self.args.pretrain:
            return max_scores

        if self.args.bntype:
            self.server.save_bn_dict()

        for r in range(self.ckpt_round, self.args.num_rounds):

            torch.cuda.empty_cache()

            self.writer.write(f'ROUND {r + 1}/{self.args.num_rounds}: '
                              f'Training {self.args.clients_per_round} Clients...')
            self.server.select_clients(r, self.target_train_clients, num_clients=self.args.clients_per_round)

            if self.args.weather_barplot:
                self.writer.write('Computing the weather barplot')
                self.bar_plot_selected_clients()

            # Init prototypes and manifold
            if self.args.lambda_proto_loss_clients:
                for client in self.server.selected_clients:
                    client.proto.protos = copy.deepcopy(self.server.proto.protos)
                    client.proto.samples_init()
                    if self.args.hyperbolic_feats:
                        if self.server.manifold is not None:
                            client.manifold = copy.deepcopy(self.server.manifold)
            # Init conditional classifier
            if self.args.conditional_classifier:
                for client in self.server.selected_clients:
                    client.conditional_classifier = self.server.conditional_classifier
                for client in self.target_test_clients:
                    client.conditional_classifier = self.server.conditional_classifier
                for client in self.add_test_clients:
                    client.conditional_classifier = self.server.conditional_classifier

            if self.args.swa_start != -1 and r + 1 >= self.args.swa_start:
                if r + 1 == self.args.swa_start:
                    self.writer.write("Setting up SWA...")
                    self.server.setup_swa_model()
                if self.args.swa_c > 1:
                    lr = schedule_cycling_lr(r, self.args.swa_c, self.args.lr_fed, self.args.swa_lr)
                    self.server.update_clients_lr(lr)

            if self.args.swa_teacher_start != -1 and r + 1 >= self.args.swa_teacher_start:
                if r + 1 == self.args.swa_teacher_start:
                    self.writer.write("Setting up SWA teacher...")
                    self.setup_swa_teacher_model()

            if self.args.teacher_step > 0 and not self.args.teacher_upd_step:
                self.model.load_state_dict(self.server.model_params_dict)
                teacher_model = deepcopy(self.model)
                teacher_model.eval()
                self.set_client_teacher(r, teacher_model)
            if self.args.teacher_kd_step > 0 and not self.args.teacher_kd_upd_step:
                teacher_kd_model = deepcopy(self.model)
                teacher_kd_model.eval()
                self.set_client_kd_teacher(r, teacher_kd_model)
            if self.args.teacher_kd_step > 0 and self.args.teacher_kd_mult_factor != -1 and \
                    r % self.args.teacher_kd_mult_step == 0 and r != 0:
                self.writer.write(f"Updating lambda_kd={self.target_train_clients[0].lambda_kd} at epoch {r}...")
                self.target_train_clients[0].lambda_kd *= self.args.teacher_kd_mult_factor
                self.writer.write(f"Done. New lambda_kd={self.target_train_clients[0].lambda_kd}")

            losses = self.server.train_clients(partial_metric=self.metrics['target_train'], r=r)

            self.plot_train_metric(r, self.metrics['target_train'], losses, plot_metric=False)
            self.metrics['target_train'].reset()

            if self.args.lambda_proto_loss_clients:
                if self.args.hyperbolic_feats:
                    self.server.update_manifold()
                self.server.update_proto()
            self.server.update_model()
            self.model.load_state_dict(self.server.model_params_dict)
            if self.args.algorithm != "FedAvg":
                self.target_test_clients[0].clusters_models = self.server.clusters_models

            self.save_model(r + 1, optimizer=self.server.optimizer)

            if self.args.swa_start != -1 and r + 1 > self.args.swa_start and \
                    (r - self.args.swa_start) % self.args.swa_c == 0:
                self.writer.write(f"Number of models: {self.swa_n}")
                self.server.update_swa_model(1.0 / (self.swa_n + 1))
                self.swa_n += 1

            if (r + 1) % self.args.eval_interval == 0 and self.all_target_client.loader.dataset.ds_type not in (
                    'unsupervised',):
                self.test([self.all_target_client], self.metrics['target_eval'], r, 'ROUND',
                          self.get_fake_max_scores(False, 1), cl_type='all_target_train_data')

            if (r + 1) % self.args.test_interval == 0 or (r + 1) == self.args.num_rounds:
                max_scores, _ = self.test(self.target_test_clients, self.metrics['target_test'], r, 'ROUND', max_scores,
                                          cl_type='target', prepend="target_")

            if self.args.additional_test_datasets and r % self.args.additional_test_interval == 0:
                for i_dataset in self.additional_test_datasets_list:
                    metric_name = f"test_{i_dataset}"
                    _, _ = self.test(test_clients[i_dataset], self.metrics[metric_name], r, 'ROUND', self.get_fake_max_scores(False, 1), cl_type=i_dataset)


            if self.args.save_cluster_models and self.args.fw_task == "ladd" and (r + 1) % 100 == 0:
                self.server.save_clusters_models(r + 1)

        return max_scores
