# import tools
from . import register

# import math
import torch
import numpy as np


def clip(v, tau):
    v_norm = torch.norm(v)
    scale = min(1, tau / v_norm)
    return v * scale

def aggregate(gradients, momentum, tau=100, n_iter=1,  **kwargs):
  # print('in TSM')
  """ CC aggregation rule.
  Args:
    gradients Non-empty list of gradients to aggregate
    ...       Ignored keyword-arguments
  Returns:
    aggregated (centered clipped) gradient
  """
  if momentum is None:
      momentum = torch.zeros_like(gradients[0])
  for _ in range(n_iter):
      momentum = (sum(clip((v - momentum), tau) for v in gradients) / len(gradients) + momentum)

  return torch.clone(momentum).detach()


def check(gradients, f, m=None, **kwargs):
  """ Check parameter validity for CC rule.
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
method_name = "CC"
register(method_name, aggregate, check, upper_bound=None, influence=None)
