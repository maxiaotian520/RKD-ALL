import torch
import numpy as np
from numba import jit
from numba.typed import List

__all__ = ['pdist', 'angle_equation', 'numba_list_conversion']


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


def recall(embeddings, labels, locater, K=[]):
    D = pdist(embeddings, squared=True)
    knn_inds = D.topk(1 + max(K), dim=1, largest=False, sorted=True)[1][:, 1:]

    """
    Check if, knn_inds contain index of query image.
    """
    assert ((knn_inds == torch.arange(0, len(labels), device=knn_inds.device).unsqueeze(1)).sum().item() == 0)
    selected_labels = labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)
    correct_labels = labels.unsqueeze(1) == selected_labels
    if (locater == 1):
        int_conversion = correct_labels.int()
        false_results = [i for i, value in enumerate(int_conversion) if value == 1]
        with open("true_results.txt", "w+") as f:
            for item in false_results:
                f.write('%s\n' %item)
        f.close()
    recall_k = []

    for k in K:
        correct_k = (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)
    return recall_k

@jit(nopython = True)
def angle_equation(list, alpha):   # len(list)   64
    output = []
    # 用三个循环完成了三个点随机的组成
    for i in range(len(list)): # 64
        for j in range(len(list)): # 64
            for k in range(len(list)):   # 64
                t_i = np.array(list[i])
                t_j = np.array(list[j])
                t_k = np.array(list[k])
                # 单位向量
                # np.linalg.norm 当输入是一个matrix的时候，在没有其他参数的情况下，np.linalg.norm的动作就是将此matrix的每一个参数的平方相加后开根号，
                # 相当于求所有column vector的长度的平方和，然后开根号。
                if i == j:
                    e_ij = np.zeros(len(t_i))
                else:
                    e_ij = (t_i - t_j)/np.linalg.norm(t_i - t_j)
                    e_ji = (t_j - t_i)/np.linalg.norm(t_j - t_i)

                if j == k:
                    e_jk = np.zeros(len(t_j))
                else:
                    e_jk = (t_j - t_k)/np.linalg.norm(t_j - t_k)
                    e_kj = (t_k - t_j)/np.linalg.norm(t_k - t_j)

                if i == k:
                    e_ik = np.zeros(len(t_k))
                else:
                    e_ik = (t_i - t_k)/np.linalg.norm(t_i - t_k)
                    e_ki = (t_k - t_i)/np.linalg.norm(t_k - t_i)

                # 原本的公式。点乘，计算角度cos<ijk> 
                dot_p = np.dot(e_ij, e_jk)
                
                # t_avg = (t_i + t_j + t_k) / 3

                # 添加到公式中的内容； 6个点两两相乘; 
                # 
                summation = np.linalg.norm(t_i - t_j)**2 + np.linalg.norm(t_j - t_i)**2 + np.linalg.norm(t_j - t_k)**2 + np.linalg.norm(t_k - t_j)**2 
                + np.linalg.norm(t_i - t_k)**2 + np.linalg.norm(t_k - t_i)**2
                # 公式
                angle = alpha * dot_p + (1- alpha) * summation
                output.append(angle)
    return output
# 为了便于@jit， 把所有list值赋值为numba.typed List
def numba_list_conversion(list):

    output = List()
    for lst in list:
        output.append(lst)

    return output
