# import tools
from . import register

# import math
import torch
import numpy as np


def aggregate(gradients, f, T,  **kwargs):
  # print('in TSM')
  """ TSM aggregation rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    trimmed soft-medoid of the gradients
  """
  n = len(gradients)
  k = n - f
  grads = torch.stack(gradients)
  cdist = torch.cdist(grads, grads, p=2)
  loss = cdist.sum(1)
  _, nbh = torch.topk(loss, k, largest=False)
  g_chosen = grads[nbh, :]

  # Calculate the weights assigned to each vector
  m = g_chosen.shape[0]
  w_bar = np.zeros((m,))
  cdist = torch.cdist(g_chosen, g_chosen, p=2)
  loss = cdist.sum(1)
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
  return torch.sum(t, 0)


def check(gradients, f, m=None, **kwargs):
  """ Check parameter validity for TSM rule.
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
  #   return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(len(gradients)) // 2}"

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "TSM"
register(method_name, aggregate, check, upper_bound=None, influence=None)