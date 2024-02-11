# import tools
from . import register

# import math
import torch
import numpy as np

def aggregate(gradients, f, aux=12, **kwargs):
    # print('in multi-TSM')
    # temp = [0.01, 0.1, 1, 10, 100, 1000]
    """ Resilient Aggregation using Auxiliary Gradients (RAAG) aggregation rule.
    Args:
      gradients Non-empty list of gradients to aggregate
      ...       Ignored keyword-arguments
    Returns:
      trimmed multi soft-medoid of the gradients
    """
    aux = len(gradients) - f
    start = 0.0001
    ratio = 10
    temp = [start * ratio ** i for i in range(aux)]
    n = len(gradients)
    k = n - f
    grads = torch.stack(gradients)
    cdist = torch.cdist(grads, grads, p=2)
    loss = cdist.sum(1)
    _, nbh = torch.topk(loss, k, largest=False)
    g_chosen = grads[nbh, :]

    # Calculate the losses of each gradient
    m = g_chosen.shape[0]
    cdist = torch.cdist(g_chosen, g_chosen, p=2)
    loss = cdist.sum(1)

    # calculate multiple soft-medoids of the chosen gradients
    sMedoids = []
    for T in temp:
        w_bar = np.zeros((m,))
        for i in range(m):
            w_bar[i] = torch.exp(-loss[i] / T)
        w_sum = np.sum(w_bar)
        if w_sum != 0:
            w = w_bar / w_sum
        else:
            w = w_bar
        g_dict = []
        for i in range(m):
            g_dict.append(g_chosen[i, :] * w[i])
        t = torch.stack(g_dict)
        sMedoids.append(torch.sum(t, 0))

    # print('type', (sum(sMedoids) / len(sMedoids)).shape)
    return sum(sMedoids) / len(sMedoids)


def check(gradients, f, m=None, **kwargs):
    """ Check parameter validity for Multi-Krum rule.
    Args:
      gradients Non-empty list of gradients to aggregate
      f         Number of Byzantine gradients to tolerate
      m         Optional number of averaged gradients for Multi-Krum
      ...       Ignored keyword-arguments
    Returns:
      None if valid, otherwise error message string
    """
    if not isinstance(gradients, list) or len(gradients) < 1:
        return f"Expected a list of at least one gradient to aggregate, got {gradients!r}"
    # if not isinstance(f, int) or f < 1 or len(gradients) < 2 * f:
    #     return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(len(gradients)) // 2}"


# ---------------------------------------------------------------------------- #
# GAR registering

#  Register aggregation rule (pytorch version)
method_name = "RAAG"
register(method_name, aggregate, check, upper_bound=None, influence=None)