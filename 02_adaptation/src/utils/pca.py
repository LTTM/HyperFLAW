from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA


from .data_utils import color_map

class ProtoPCA():
    def __init__(self,
                 dataset="cityscapes",
                 n_components=2,
                 num_vecs=10,
                 log_frequency=500,
                 log_after_iterations=5000):
        self.pca_init = False
        self.pca = IncrementalPCA(n_components=n_components)
        self.norm_pca = IncrementalPCA(n_components=n_components)
        self.cmap = [(r/255, g/255, b/255) for r, g, b in color_map([dataset])]
        self.num_vecs = num_vecs
        self.log_frequency = log_frequency
        self.log_after_iterations = log_after_iterations

    def update(self, vecs, nvecs):
        if self.pca_init:
            self.pca.partial_fit(vecs)
            self.norm_pca.partial_fit(nvecs)
        else:
            self.pca.fit(vecs)
            self.norm_pca.fit(nvecs)
            self.pca_init = True

    def project(self, vecs, nvecs):
        return self.pca.transform(vecs), self.norm_pca.transform(nvecs)

    def get_figure(self, bvecs, proto, train=False):
        if proto.batches_seen < self.log_after_iterations:
            return None
        if not all(p.numel() > 0 for p in proto.protos.values()):
            return None

        pvecs = np.array([p.squeeze(-1).detach().cpu().numpy() for p in proto.protos.values()])
        npvecs = pvecs/np.linalg.norm(pvecs, axis=1, keepdims=True)

        if train or not self.pca_init: #TODO: togliere "not self.pca_init" quando implementiamo il trahsfer server->client della matrice
            self.update(pvecs, npvecs)

        if proto.batches_seen % self.log_frequency > 0:
            return None

        ps, nps = self.project(pvecs, npvecs)

        vecs, nvecs = {}, {}
        for k in bvecs:
            v = bvecs[k]
            if v.shape[1] > 0:
                per = np.random.permutation(v.shape[1])[:self.num_vecs]
                vs = v[:, per].detach().cpu().numpy().T
                nvs = vs / np.linalg.norm(vs, axis=1, keepdims=True)
                vecs[k], nvecs[k] = self.project(vs, nvs)

        fig, axs = plt.subplots(1,2, figsize=(20,10))

        axs[0].scatter(ps[:,0], ps[:,1], marker="*", c=self.cmap[:ps.shape[0]])
        axs[1].scatter(nps[:,0], nps[:,1], marker="*", c=self.cmap[:ps.shape[0]])
        
        for k in vecs:
            v, nv = vecs[k], nvecs[k]
            axs[0].scatter(v[:,0], v[:,1], marker=".", color=self.cmap[k])
            axs[1].scatter(nv[:,0], nv[:,1], marker=".", color=self.cmap[k])

        axs[0].set_title("PCA")
        axs[1].set_title("Normed PCA")
        fig.tight_layout()
        return fig