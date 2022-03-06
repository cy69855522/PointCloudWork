import torch

def knn(x, k, d=1, fps=None): # x[B, N, C] fps[B*n]
    B, N, _ = x.size()
    n = N if fps is None else int(fps.size(0) / B)
    dev = x.device

    with torch.no_grad():
        inner = -2 * torch.matmul(x, x.transpose(2, 1)) # [B, N, N]
        xx = torch.sum(x**2, dim=-1, keepdim=True) # [B, N, 1]
        dis = -xx.transpose(2, 1) - inner - xx # [B, N, N]
        if fps is not None: dis = dis.view(B*N, N)[fps].view(B, n, N)
        sid = dis.topk(k=k*d-d+1, dim=-1)[1][..., ::d] # (B, n, k)
        sid += torch.arange(B, device=dev).view(B, 1, 1) * N
        sid = sid.reshape(B*n, k) # [B*n*k]
        tid = torch.arange(B * N, device=dev) if fps is None else fps # [B*n]
        tid = tid.view(-1, 1).repeat(1, k) # [B*n, k]
        return sid, tid # [B*n, k]