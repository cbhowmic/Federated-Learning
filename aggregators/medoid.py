
import tools
from . import register

import math
import torch

# # medoid GAR

def aggregate(gradients, **kwargs):
  # print('in medoid')
  """ medoid aggregation rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    medoid of the gradients
  """
  grads = torch.stack(gradients)
  # print('type', type(grads), grads.shape)
  cdist = torch.cdist(grads, grads, p=2)
  i_star = torch.argmin(cdist.sum(1))
  # print('i star', i_star)
  return grads[i_star, :]


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
  #   return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(len(gradients)) // 2}"




# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "medoid"
register(method_name, aggregate, check, upper_bound=None, influence=None)
