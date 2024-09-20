import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .loss_utils import lovasz_softmax
from hypll.manifolds.poincare_ball.math.diffgeom import mobius_add, dist


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()


class HardNegativeMining(nn.Module):

    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, _):
        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss


def weight_train_loss(losses):
    fin_losses = {}
    c = list(losses.keys())[0]
    loss_names = list(losses[c]['loss'].keys())
    for l_name in loss_names:
        tot_loss = 0
        weights = 0
        for _, d in losses.items():
            try:
                tot_loss += d['loss'][l_name][-1] * d['num_samples']
                weights += d['num_samples']
            except:
                pass
        fin_losses[l_name] = tot_loss / weights
    return fin_losses


def weight_test_loss(losses):
    tot_loss = 0
    weights = 0
    for k, v in losses.items():
        tot_loss = tot_loss + v['loss'] * v['num_samples']
        weights = weights + v['num_samples']
    return tot_loss / weights


class IW_MaxSquareloss(nn.Module):
    requires_reduction = False

    def __init__(self, ignore_index=255, ratio=0.2, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.ratio = ratio

    def forward(self, pred, **kwargs):
        prob = F.softmax(pred, dim=1)
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long) * self.ignore_index)
        label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(),
                               bins=C + 1, min=-1,
                               max=C - 1).float()
            hist = hist[1:]
            weight = \
            (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1))).to(
                argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0).unsqueeze(1)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2) * weights)[mask]) / (batch_size * C)
        return loss


class AdvEntLoss:
    requires_reduction = False

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, pred, **kwargs):
        prob = F.softmax(pred, dim=1)
        assert prob.dim() == 4
        n, c, h, w = prob.size()
        return -torch.sum(torch.mul(prob, torch.log2(prob + 1e-30))) / (n * h * w * np.log2(c))


class SelfTrainingLoss(nn.Module):
    requires_reduction = False

    def __init__(self, conf_th=0.9, fraction=0.66, ignore_index=255, lambda_selftrain=1, **kwargs):
        super().__init__()
        self.conf_th = conf_th
        self.fraction = fraction
        self.ignore_index = ignore_index
        self.teacher = None
        self.lambda_selftrain = lambda_selftrain

    def set_teacher(self, model):
        self.teacher = model

    def get_image_mask(self, prob, pseudo_lab):
        max_prob = prob.detach().clone().max(0)[0]
        mask_prob = max_prob > self.conf_th if 0. < self.conf_th < 1. else torch.zeros(max_prob.size(),
                                                                                       dtype=torch.bool).to(
            max_prob.device)
        mask_topk = torch.zeros(max_prob.size(), dtype=torch.bool).to(max_prob.device)
        if 0. < self.fraction < 1.:
            for c in pseudo_lab.unique():
                mask_c = pseudo_lab == c
                max_prob_c = max_prob.clone()
                max_prob_c[~mask_c] = 0
                _, idx_c = torch.topk(max_prob_c.flatten(), k=int(mask_c.sum() * self.fraction))
                mask_topk_c = torch.zeros_like(max_prob_c.flatten(), dtype=torch.bool)
                mask_topk_c[idx_c] = 1
                mask_c &= mask_topk_c.unflatten(dim=0, sizes=max_prob_c.size())
                mask_topk |= mask_c
        return mask_prob | mask_topk

    def get_batch_mask(self, pred, pseudo_lab):
        b, _, _, _ = pred.size()
        mask = torch.stack([self.get_image_mask(pb, pl) for pb, pl in zip(F.softmax(pred, dim=1), pseudo_lab)], dim=0)
        return mask

    def get_pseudo_lab(self, pred, imgs=None, conditions=None, return_mask_fract=False, model=None):
        teacher = self.teacher if model is None else model

        try:
            pred = pred['out']
        except:
            pass

        if teacher is not None:
            with torch.no_grad():
                pred = teacher(imgs, condition=conditions)
                try:
                    pred = pred['out']
                except:
                    pass
                pseudo_lab = pred.detach().max(1)[1]
        else:
            pseudo_lab = pred.detach().max(1)[1]
        mask = self.get_batch_mask(pred, pseudo_lab)
        pseudo_lab[~mask] = self.ignore_index
        if return_mask_fract:
            return pseudo_lab, F.softmax(pred, dim=1), mask.sum() / mask.numel()
        return pseudo_lab

    def forward(self, pred, conditions=None, imgs=None):
        pseudo_lab = self.get_pseudo_lab(pred, imgs=imgs, conditions=conditions)
        loss = F.cross_entropy(input=pred, target=pseudo_lab, ignore_index=self.ignore_index, reduction='none')
        return loss.mean() * self.lambda_selftrain


class SelfTrainingLossEntropy(SelfTrainingLoss):
    def __init__(self, lambda_entropy=0.005, **kwargs):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy

    def cross_entropy(self, pred, conditions, imgs=None):
        pseudo_lab = self.get_pseudo_lab(pred, imgs=imgs, conditions=conditions)
        loss = F.cross_entropy(input=pred, target=pseudo_lab, ignore_index=self.ignore_index, reduction='none')
        return loss.mean()

    @staticmethod
    def entropy_loss(pred):
        p = F.softmax(pred, dim=1)
        logp = F.log_softmax(pred, dim=1)
        plogp = p * logp
        ent = -1.0 * plogp.sum(dim=1)
        ent = ent / 2.9444
        ent = ent ** 2.0 + 1e-8
        ent = ent ** 2.0
        return ent.mean()

    def forward(self, pred, imgs=None):
        ce_loss = self.cross_entropy(pred, imgs)
        entropy_loss = self.entropy_loss(pred)*self.lambda_entropy
        loss = ce_loss + entropy_loss
        return loss


class SelfTrainingLossLovaszEntropy(SelfTrainingLossEntropy):
    def __init__(self, lambda_entropy=0.005, **kwargs):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.teacher = None

    def forward(self, pred, conditions, imgs=None):
        pseudo_lab = self.get_pseudo_lab(pred, imgs=imgs, conditions=conditions)
        output_sm = F.softmax(pred, dim=1)
        loss_seg = lovasz_softmax(output_sm, pseudo_lab, ignore=19)
        entropy_loss = self.entropy_loss(pred) * self.lambda_entropy
        loss = loss_seg + entropy_loss
        return loss


class LovaszLoss(SelfTrainingLoss):

    def __init__(self, lambda_selftrain=1, **kwargs):
        super().__init__(**kwargs)
        self.teacher = None
        self.lambda_selftrain = lambda_selftrain

    def forward(self, pred, conditions, imgs=None, weights=None):
        pseudo_lab = self.get_pseudo_lab(pred, imgs=imgs, conditions=conditions)
        output_sm = F.softmax(pred, dim=1)
        loss_seg = lovasz_softmax(output_sm, pseudo_lab, ignore=255, weights=weights)
        return loss_seg * self.lambda_selftrain


class EntropyLoss(nn.Module):

    def __init__(self, lambda_entropy=0.005, num_classes=13, **kwargs):
        super().__init__(**kwargs)
        self.lambda_entropy = lambda_entropy
        self.normalization_factor = self.__get_normalization_factor(num_classes)

    def __get_normalization_factor(self, num_classes):
        a = torch.ones((1, num_classes, 1, 1))
        a = 1 / num_classes * a
        p = F.softmax(a, dim=1)
        logp = F.log_softmax(a, dim=1)
        plogp = p * logp
        ent = -1.0 * plogp.sum(dim=1)
        return ent.item()

    def entropy_loss(self, pred):
        p = F.softmax(pred, dim=1)
        logp = F.log_softmax(pred, dim=1)
        plogp = p * logp
        ent = -1.0 * plogp.sum(dim=1)
        ent = ent / self.normalization_factor
        ent = ent ** 2.0 + 1e-8
        ent = ent ** 2.0
        return ent.mean()

    def forward(self, pred):
        loss = self.entropy_loss(pred)*self.lambda_entropy
        return loss


class Diff2d(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))


class Symkl2d(nn.Module):

    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.n_target_ch = 20
        self.prob1 = self.prob2 = self.log_prob1 = self.log_prob2 = None

    def forward(self, inputs1, inputs2):
        self.prob1 = F.softmax(inputs1)
        self.prob2 = F.softmax(inputs2)
        self.log_prob1 = F.log_softmax(self.prob1)
        self.log_prob2 = F.log_softmax(self.prob2)

        loss = 0.5 * (F.kl_div(self.log_prob1, self.prob2, size_average=self.size_average)
                      + F.kl_div(self.log_prob2, self.prob1, size_average=self.size_average))

        return loss


class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, pred_labels=None, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])
        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)
        loss = (outputs * labels).mean(dim=1)

        if pred_labels is not None:
            loss = loss * pred_labels.float()
        if mask is not None:
            loss = loss * mask
        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss
        return outputs


class HyperMSE(nn.Module):
    def __init__(self, manifold):
        super(HyperMSE, self).__init__()
        self.manifold = manifold

    def forward(self, x, y):
        return (mobius_add(-x.T, y.T, self.manifold.c(), dim=-1) ** 2).mean()


class HyperCosineDistance(nn.Module):
    def __init__(self, manifold):
        super(HyperCosineDistance, self).__init__()
        self.manifold = manifold

    def dist(self,
            x: torch.Tensor,
            y: torch.Tensor,
            c: torch.Tensor,
            dim: int = -1,
            keepdim: bool = False,
    ) -> torch.Tensor:

        c_sqrt = c.sqrt()
        add = mobius_add(-x, y, c, dim=dim).norm(dim=dim, keepdim=keepdim)
        dist = 2 / c_sqrt * (c_sqrt * add).clamp(min=-1 + 1e-5, max=1 - 1e-5).atanh()
        return dist

    def hyper_cossim(self, x, y, c):
        # Distance-bounded in between [0, c?] #TODO: da verificare
        # dist = c.sqrt() * mobius_add(-x, y, c, dim=-1).norm(dim=-1, keepdim=False)
        return self.dist(x, y, c).mean()

    def forward(self, x, y):
        return self.hyper_cossim(x.T, y.T, self.manifold.c())


class CosineDistanceLoss(nn.Module):
    def __init__(self, dim=0, eps=1e-08):
        super(CosineDistanceLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)
        self.eps = eps

    def forward(self, x, y):
        scores = self.cos(x, y)
        return (1. - scores).mean()


class ProtoClustering(nn.Module):
    def __init__(self, manifold=None, distance='mse'):
        super().__init__()
        self.manifold = manifold
        self.mse = nn.MSELoss() if not self.manifold else HyperMSE(manifold=self.manifold)
        self.cos = CosineDistanceLoss() if not self.manifold else HyperCosineDistance(manifold=self.manifold)
        self.distance = self.mse if distance == 'mse' else self.cos

    def forward(self, feature_vecs, protos):
        l = 0.
        active_pclasses = [c for c in feature_vecs.keys() if feature_vecs[c].numel() > 0]
        for c in active_pclasses:
            if protos[c].numel() > 0:
                l += self.distance(feature_vecs[c], protos[c].expand_as(feature_vecs[c]))
        l /= len(active_pclasses)
        return l if l else torch.zeros(1).to(protos[0].device)


class ProtoContrastive(nn.Module):
    def __init__(self, lambda_neg=.1, manifold=None, type='similarity', margin=0., temperature=.5, distance='cos'):
        super().__init__()
        self.manifold = manifold
        self.loss_type = type
        self.dist_type = distance
        self.lambda_neg = lambda_neg if self.loss_type != 'similarity' else 0.
        self.margin = margin
        self.temperature = temperature

    def dist(self,
            x: torch.Tensor,
            y: torch.Tensor,
            c: torch.Tensor,
            dim: int = -1,
            keepdim: bool = False,
    ) -> torch.Tensor:

        c_sqrt = c.sqrt()
        add = mobius_add(-x, y, c, dim=dim).norm(dim=dim, keepdim=keepdim)
        dist = 2 / c_sqrt * (c_sqrt * add).clamp(min=-1 + 1e-5, max=1 - 1e-5).atanh()
        return dist

    def hyper_dist(self, x, y, c):
        dist0 = self.dist(x.T, y.T, c)
        # dist0 = dist0 / dist0.norm(dim=-1, keepdim=False)
        return dist0.mean()  # (dist**2)?

    # def hyper_cossim(self, x, y, c):
    #     dist0 = self.dist(x.T, y.T, c)
    #     return dist0.cosh().mean()

    def hyper_cossim(self, x, y, c):
        # Distance-bounded in between [0, c?] #TODO: da verificare
        dist = c.sqrt() * mobius_add(-x.T, y.T, c, dim=-1).norm(dim=-1, keepdim=False)
        return dist.mean()

    def cossim(self, x, y):
        return torch.mean(F.cosine_similarity(x, y, dim=0))

    def d_cosine(self, x, y):
        if not self.manifold:
            return 1 - self.cossim(x, y)
        else:
            return self.hyper_cossim(x, y, self.manifold.c())

    def d_l2(self, x, y):
        if not self.manifold:
            return torch.mean(torch.sqrt(torch.sum(torch.pow(torch.subtract(x, y), 2), dim=0)))
        else:
            return self.hyper_dist(x, y, self.manifold.c())

    def distance(self, x, y):
        if self.dist_type == 'cos':
            return self.d_cosine(x, y)
        elif self.dist_type == 'l2':
            return self.d_l2(x, y)

    def forward(self, feature_vecs, protos):
        l = 0.
        dnegs = torch.empty(0, device=feature_vecs[0].device)
        active_pclasses = [c for c in feature_vecs.keys() if feature_vecs[c].numel() > 0]
        active_nclasses = [c for c in protos.keys() if protos[c].numel() > 0]
        Np = len(active_pclasses)
        for c_pos in active_pclasses:
            dpos = self.distance(feature_vecs[c_pos], protos[c_pos].expand_as(feature_vecs[c_pos]))
            for c_neg in active_nclasses:
                if self.lambda_neg > 0 and c_pos != c_neg and protos[c_pos].numel() > 0 and protos[c_neg].numel() > 0:
                    _dneg = self.distance(feature_vecs[c_pos], protos[c_neg].expand_as(feature_vecs[c_pos]))
                    dnegs = torch.cat([dnegs, _dneg.unsqueeze(0)])
            dneg = torch.mean(dnegs) if dnegs.numel() != 0 else 0.
            if self.loss_type == 'triplet':
                l += 1 / Np * torch.clamp((1 - self.lambda_neg) * dpos - self.lambda_neg * dneg, min=0)
            elif self.loss_type == 'similarity':
                l += 1 / Np * dpos
            elif self.loss_type == 'ntxent':
                dpos = dpos / self.temperature
                dnegs = dnegs / self.temperature
                max_value = torch.max(dpos, torch.max(dnegs))
                numerator = torch.exp(dpos - max_value)
                denominator = torch.sum(torch.exp(dnegs - max_value)) + numerator if dneg else numerator
                l += -1 / Np * torch.log(numerator / denominator) + 1e-9
            dnegs = torch.empty(0, device=feature_vecs[0].device)
        return l if l else torch.zeros(1).to(protos[0].device)
