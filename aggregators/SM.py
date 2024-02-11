# import tools
from . import register

# import math
import torch
import numpy as np


def aggregate(gradients, T,  **kwargs):
  """ SM aggregation rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    soft-medoid of the gradients
  """
  # for key, value in kwargs.items():
  #   print("%s == %s" % (key, value))
  # print('fegfjdgfje' ,kwargs.items())

  grads = torch.stack(gradients)
  # Calculate the weights assigned to each vector
  n = len(gradients)
  w_bar = np.zeros((n,))
  cdist = torch.cdist(grads, grads, p=2)
  loss = cdist.sum(1)
  for i in range(n):
    w_bar[i] = torch.exp(-loss[i] / T)
  w_sum = np.sum(w_bar)
  if w_sum != 0:
    w = w_bar / w_sum
  else:
    w = w_bar
  g_dict = []
  for i in range(n):
    g_dict.append(grads[i, :] * w[i])
  t = torch.stack(g_dict)
  return torch.sum(t, 0)


def check(gradients, **kwargs):
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


method_name = "SM"
register(method_name, aggregate, check,)
