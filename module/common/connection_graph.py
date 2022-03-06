
import torch
from torch import nn
from torch_geometric.nn import knn_interpolate

from .knn import knn

class KGraph(nn.Module):
    def __init__(self, k, d=1):
        super().__init__()
        self.k = k
        self.d = d

    def forward(self, pos, fps=None, batch=None):
        k, d = self.k, self.d
        B, N, _ = pos.size()
        n = N if fps is None else int(len(fps) / B)
        dev = pos.device
        
        with torch.no_grad():
            pos_ = pos.view(B*N, -1)
            sid_euc, tid_euc = knn(pos, k, d, fps) # [B*n*k]
            euc_i, euc_j = pos_[tid_euc], pos_[sid_euc] # [B*n*k, C]
            euc_diff = euc_j - euc_i # [B*n*k, C]
            
            euc_diff_ = euc_diff.view(B, n, k, -1)
            cov = euc_diff_.transpose(2, 3).matmul(euc_diff_) # [B, n, C, C]
            #cov = cov.cpu().numpy()
            #eig = np.linalg.eig(cov)[0]
            #eig = torch.tensor(eig, device=dev) # [B, n, C]
            cov = cov.view(B, n, 9)
            eig, vec = eigen_values(cov, eigenvectors=True) # [B, n, C] [B, n, C, C]
            eig_ = eig.view(B*n, -1)
            # expand --------------------------------------------------
            if fps is not None:
                eig_ = knn_interpolate(eig_, pos_[fps], pos_, batch[fps], batch, 3) # [B*N, C]
                eig = eig_.view(B, N, -1)
            # ---------------------------------------------------------
            sid_eig, tid_eig = knn(eig, k, d, fps) # [B*n*k]
            if  self.prio == 2:
                return (sid_euc, tid_euc), (sid_eig, tid_eig), (eig, vec)
            elif self.prio ==1:
                eig_i, eig_j = eig_[tid_eig], eig_[sid_eig] # [B*n*k, C]
                eig_diff = eig_j - eig_i # [B*n*k, C]
                return (sid_euc, tid_euc), (sid_eig, tid_eig), eig_j, eig_diff
            else:
                return (sid_euc, tid_euc), (sid_eig, tid_eig)