# import tools
from . import register

# import math
import torch
import numpy as np


def aggregate(gradients, f, T=10,  **kwargs):
  # print('in TSM')
  """ trimmed average aggregation rule: trim the higher energy gradients then average the chosen ones.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    trimmed soft-medoid of the gradients
  """
  n = len(gradients)
  k = n - f
  # print('k=', k)
  grads = torch.stack(gradients)
  cdist = torch.cdist(grads, grads, p=2)
  loss = cdist.sum(1)
  _, nbh = torch.topk(loss, k, largest=False)
  g_chosen = grads[nbh, :]

  return torch.mean(g_chosen, 0)


def check(gradients, f, m=None, **kwargs):
  """ Check parameter validity for trimmed average rule.
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
method_name = "trimAvg"
register(method_name, aggregate, check, upper_bound=None, influence=None)