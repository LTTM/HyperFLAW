import torch
from clients import FdaInvClient
from utils import StyleAugment
from tqdm import tqdm
from torch import nn
import wandb
from PIL import Image
import torchvision
import os
import numpy as np

class LADDClient(FdaInvClient):

    def __init__(self, args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                 device=None, test_user=False):
        super().__init__(args, client_id, dataset, model, writer, batch_size, world_size, rank,
                         num_gpu, device=device, test_user=test_user)

        self.cluster_id = None
        self.clusters_models = []
        self.styleaug_test = None
        self.test_images_clusters = []

        if args.test_only_with_global_model:
            self.test = super().test

    def find_test_images_cluster(self, k_means_model):
        self.dataset.return_unprocessed_image = True
        self.styleaug_test = StyleAugment(1, self.args.fda_L, self.args.fda_size, b=self.args.fda_b)

        for sample in tqdm(self.dataset):
            img_processed = self.styleaug_test.preprocess(sample)
            style = self.styleaug_test._extract_style(img_processed)
            cluster = k_means_model.predict(style.reshape(1, -1))[0]
            self.test_images_clusters.append(cluster)

        self.dataset.return_unprocessed_image = False

    def test(self, metric, swa=False):

        tot_loss = 0.0

        self.model.eval()

        if swa:
            self.switch_bn_stats_to_test()

        self.dataset.test = True

        global_model_dict = self.model.state_dict()

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.loader):

                if (i + 1) % self.args.print_interval == 0:
                    self.writer.write(f'{self}: {i + 1}/{self.len_loader}, '
                                      f'{round((i + 1) / self.len_loader * 100, 2)}%')

                original_images, images = images
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                self.model.load_state_dict(global_model_dict)
                outputs_global = self.get_test_output(images)
                try:
                    self.update_metric(metric[0], outputs_global, labels, is_test=True)
                except:
                    self.update_metric(metric, outputs_global, labels, is_test=True)

                if self.args.algorithm != "FedAvg" and self.clusters_models:
                    self.model.load_state_dict(self.clusters_models[self.test_images_clusters[i]])
                    outputs = self.get_test_output(images)
                    try:
                        self.update_metric(metric[1], outputs, labels, is_test=True)
                    except:
                        self.update_metric(metric, outputs, labels, is_test=True)

                if outputs_global.shape != labels.shape:
                    outputs_global = torch.nn.functional.interpolate(outputs_global, labels.shape[1:], mode='nearest')
                loss = self.calc_test_loss(outputs_global, labels)
                tot_loss += loss.item()

                if self.args.qualitative:
                    _, prediction = outputs_global.max(dim=1)
                if self.args.qualitative == "wandb":
                    self.writer.wandb.log_image(f"qualitative/RGB/{i}_", [images])
                    gt = self.writer.target_label2color(labels.cpu().squeeze())
                    self.writer.wandb.log_image(f"qualitative/GT/{i}", [wandb.Image(gt)])
                    pred = self.writer.target_label2color(prediction.cpu().squeeze())
                    self.writer.wandb.log_image(f"qualitative/PRED/{i}", [wandb.Image(pred)])
                elif self.args.qualitative == "file":
                    normalized_tensor = (images - images.min()) / (images.max() - images.min())
                    path = os.path.join(self.args.qualitative_dir, f"{i}_")  #
                    torchvision.utils.save_image(normalized_tensor, fp=path + "RGB.jpg")
                    gt = Image.fromarray(np.array(self.writer.target_label2color(labels.cpu().squeeze())))
                    gt.save(fp=path + "GT.png")
                    pred = Image.fromarray(np.array(self.writer.target_label2color(prediction.cpu().squeeze())))
                    pred.save(fp=path + "PRED.jpg")

                torch.cuda.empty_cache()

            try:
                metric[0].synch(self.device)
                metric[1].synch(self.device)
            except:
                metric.synch(self.device)

            mean_loss = self.manage_tot_test_loss(tot_loss)

        self.dataset.test = False

        return {f'{self}_loss': mean_loss}

    def update_metric(self, metric, outputs, labels, **kwargs):
        _, prediction = outputs.max(dim=1)
        if prediction.shape != labels.shape:
            prediction = nn.functional.interpolate(
                prediction.unsqueeze(0).double(), labels.shape[1:], mode='nearest').squeeze(0).long()
        labels = labels.cpu().numpy()

        # If the used dataset is FLYAWARER map the prediction
        if hasattr(self.dataset, "labels2coarse"):
            z = 255 * torch.ones_like(prediction)
            for k, v in self.dataset.labels2coarse.items():
                z[prediction == k] = v
            prediction = z.clone()

        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)