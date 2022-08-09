from collections import namedtuple
from typing import Union

import torch


_KNN = namedtuple("KNN", "dists idx knn")


def apply_knn(p1,p2,lengths1,lengths2,K,version,return_sorted):

    inner = -2. * torch.matmul(p1,p2.transpose(2,1))
    p1_2 = torch.sum((p1.transpose(2,1))**2,dim=1,keepdim=True)
    p2_2 = torch.sum((p2.transpose(2,1))**2,dim=1,keepdim=True)
    dist = p1_2 + inner + p2_2.transpose(2,1)
    value, pos = (-dist).topk(k=K,dim=-1)

    value = -value

    return value, pos

def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
) -> _KNN:

    if p1.shape[0] != p2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("pts1 and pts2 must have the same point dimension.")

    p1 = p1.contiguous()
    p2 = p2.contiguous()

    P1 = p1.shape[1]
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((p1.shape[0],), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((p1.shape[0],), P2, dtype=torch.int64, device=p1.device)


    p1_dists, p1_idx = apply_knn(p1,p2,lengths1,lengths2,K,version,return_sorted)
    p2_nn = None
    if return_nn:
        p2_nn = knn_gather(p2, p1_idx, lengths2)

    return _KNN(dists=p1_dists, idx=p1_idx, knn=p2_nn if return_nn else None)


def knn_gather(x: torch.Tensor, idx: torch.Tensor, lengths: Union[torch.Tensor, None] = None):

    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full((x.shape[0],), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    # idx_expanded has shape [N, L, K, U]

    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)
    # p2_nn has shape [N, L, K, U]
    """
    needs_mask = lengths.min() < K
    if needs_mask:
        # mask has shape [N, K], true where idx is irrelevant because
        # there is less number of points in p2 than K
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]

        # expand mask to shape [N, L, K, U]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0
    """
    return x_out
