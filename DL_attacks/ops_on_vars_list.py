
import torch
import copy
import numpy as np

""" Utility functions to perform operations on lists/dict of tensors (e.g., variables/gradients) """

def flat_tensor_list(par):
    f = []
    for l in par:
        f.append(l.detach().cpu().numpy().reshape(-1))
    return np.concatenate(f)

def deepCopyModel(model):
    _model = copy.deepcopy(model)
    return _model

def clone_list_tensors(A):
    n = len(A)
    B = [None] * n
    for i in range(n):
        B[i] = A[i].clone()
    return B

def assign_list_variables(A, B):
    """ A <- B """
    assert len(A) == len(B)
    n = len(A)
    for i in range(n):
        A[i].data.copy_(B[i].data)

def init_list_variables(A):
    B = [torch.zeros_like(p) for p in A.parameters()]
    return B

def agg_sum(A, B):
    assert(len(A) == len(B))
    n = len(A)
    C = [None] * n

    for i in range(n):
        C[i] = A[i] + B[i]
        
    return C

def agg_sum_param(A, B):
    A_param = [p for p in A.parameters()]
    B_param = [p for p in B.parameters()]
    
    C = [p1+p2 for p1, p2 in zip(A_param, B_param)]
    return C

def agg_sub_param(A, B):
    A_param = [p for p in A.parameters()]
    B_param = [p for p in B.parameters()]
    
    C = [p1-p2 for p1, p2 in zip(A_param, B_param)]
    return C

def agg_div_param(A, alpha):
    return [p/alpha for p in A]

def agg_sub(A, B):
    assert(len(A) == len(B))
    n = len(A)
    C = [None] * n

    for i in range(n):
        C[i] = A[i] - B[i]
        
    return C

def agg_div(A, alpha):
    n = len(A)
    C = [None] * n
    for i in range(n):
        C[i] = A[i] / alpha
    return C

def agg_mul(A, alpha):
    n = len(A)
    C = [None] * n
    for i in range(n):
        C[i] = A[i] * alpha
    return C

def agg_neg(A):
    n = len(A)
    C = [None] * n
    for i in range(n):
        C[i] = -A[i]
    return C

def agg_sumc(A, B):
    n = len(A)
    C = [None] * n
    for i in range(n):
        C[i] = A[i] + B[i]
    return C

def select_nn_mus(keys, buff):
    """ get only a subset of a dictionary """
    new_buff = {}
    for key in keys:
        name = key.name
        new_buff[name] = buff[name]
    return new_buff
