import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import functional as F

from hypll.manifolds.poincare_ball import PoincareBall
from hypll.tensors import TangentTensor


#
# label_down = F.interpolate(labels.unsqueeze(1).float(), out_dict['enc_feats'].shape[2:], mode='nearest').squeeze(
#     1).long()
# import matplotlib
#
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
#
# fig, axs = plt.subplots(2, 2)
# axs[0, 0].imshow(images[0, 0].detach().cpu().numpy(), cmap='gray')
# axs[0, 1].imshow(images[0, 1].detach().cpu().numpy(), cmap='gray')
# axs[1, 0].imshow(labels[0].detach().cpu().numpy(), vmax=18)
# axs[1, 1].imshow(label_down[0].detach().cpu().numpy(), vmax=18)
# plt.show(block=True)


# enc_feats = out_dict['enc_feats']
# dec_feats = out_dict['dec_feats']
# outputs = self.model(images)['out']

# def prepare_feature_label(labels, features):
#     # Downsample the labels to match the features
#     labels = F.interpolate(labels.unsqueeze(1).float(), features.shape[2:], mode='nearest').squeeze(
#     1).long()
#
#     cl_present = torch.unique(input=labels)
#
#     feats = {c: torch.empty(features.shape[1], 0) for c in cl_present if c < 255}
#     for c in cl_present:
#         if c < 255:
#             for b in range(labels.shape[0]):
#                 mask = labels[b] == c # pixels of class c
#                 vecs = features[b,:,mask].flatten(-1,-2) # C x N
#                 feats[c] = torch.cat([feats[c], vecs], dim=1)
#
#     # feats: dict of class -> features of the class
#
#     # Flatten
#     features = features.detach().view(-1, features.shape[1]).cpu()
#     labels = labels.view(-1)
#
#     # Remove unlabeled
#     features = features[labels != 255, :]
#     labels = labels[labels != 255]
#
#     # Consider for update only present classes
#     cl_present = torch.unique(input=labels)
#
#     # Create the hyperbolic mapping
#     hyper_features = hyperbolic_mapping(features)
#
#     return

def hyperbolic_mapping(features):
    return

class Protos(nn.Module):
    def __init__(self, num_classes, channels, momentum=.85, device=None, exp=False):
        super().__init__()

        self.num_classes = num_classes
        self.channels = channels
        self.protos = {c: torch.empty(channels, 0, device=device) for c in range(num_classes)}
        self.num_samples = {c: 0. for c in range(num_classes)}
        self.batches_seen = 0
        self.momentum = momentum
        self.exp = exp
        self.manifold = None

    def samples_init(self):
        self.num_samples = {c: 0. for c in range(self.num_classes)}

    def forward(self, features, labels, manifold=None):
        # Downsample labels to features
        labels = F.interpolate(labels.unsqueeze(1).float(), features.shape[2:], mode='nearest').squeeze(
            1).long()

        # Create per-class vectors
        feat_vecs = {c: torch.empty(features.shape[1], 0, device=features.device) for c in range(self.num_classes)}
        for c in range(self.num_classes):
            for b in range(labels.shape[0]):
                mask = labels[b] == c  # pixels of class c
                vecs = features[b, :, mask]  # C x N
                feat_vecs[c] = torch.cat([feat_vecs[c], vecs], dim=1)

        if self.exp:
            valid = [c for c in range(self.num_classes) if feat_vecs[c].numel() > 0]
            for c in valid:
                self.manifold = manifold
                tangents = TangentTensor(data=torch.t(feat_vecs[c]), man_dim=1, manifold=manifold) # TODO: imposta come hyperparam
                feat_vecs[c] = manifold.expmap(tangents)
            protos = {c: manifold.midpoint(feat_vecs[c], keepdim=True).tensor.view(-1, 1) for c in valid}
            for c in valid:
                feat_vecs[c] = torch.t(feat_vecs[c].tensor)
            # return protos, feat_vecs  # .tensor to go back
        else:
            # Compute per-class prototypes
            protos = {c: vecs.mean(dim=-1, keepdims=True) for c, vecs in feat_vecs.items() if vecs.numel() > 0}

        # Samples seen over the training # TODO re-init a zero
        self.num_samples = {c: self.num_samples[c] + v.size(1) for c, v in feat_vecs.items()}

        # Smoothing
        for c in protos:
            if self.protos[c].shape[1] == 0:
                self.protos[c] = protos[c].detach()
            else:
                if self.num_samples[c] != 0:  # update only if proto was modified
                    self.protos[c] = self.momentum*self.protos[c] + (1-self.momentum)*protos[c].detach()
        self.batches_seen += 1

        return protos, feat_vecs

if __name__ == "__main__":

    labels = torch.randint(0,19, (8,512,1024), dtype=torch.long, device='cuda')
    features = torch.rand(8,1024,32,64, device='cuda', dtype=torch.float32)

    compute_proto = Protos(19, 1024)
    mse = nn.MSELoss()

    bprots, bvecs = compute_proto(features, labels)

    l = mse(bvecs[0], compute_proto.protos[0].expand_as(bvecs[0]))
    # loss = ProtoMSE()
    #l = loss(bvecs, compute_proto.protos)
    print(l)
