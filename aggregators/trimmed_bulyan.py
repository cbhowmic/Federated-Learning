import tools
from . import register

import math
import torch
import numpy as np

def aggregate(gradients, f, T=10,  **kwargs):
    # print('in trimmed bulyan')
    """ TSM aggregation rule.
    Args:
      gradients Non-empty list of gradients to aggregate
      ...       Ignored keyword-arguments
    Returns:
      trimmed bulyan of the gradients
    """

    n = len(gradients)
    k = n - f
    grads = torch.stack(gradients)
    cdist = torch.cdist(grads, grads, p=2)
    loss = cdist.sum(1)
    _, nbh = torch.topk(loss, k, largest=False)
    g_chosen = grads[nbh, :]

    n, d = g_chosen.shape
    param_med = torch.median(g_chosen, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(g_chosen - param_med), dim=0)
    sorted_params = g_chosen[sort_idx, torch.arange(d)[None, :]]

    return torch.mean(sorted_params[:n - 2 * f], dim=0)


def check(gradients, f, m=None, **kwargs):
  """ Check parameter validity for trimmed Bulyan rule.
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
  if not isinstance(f, int) or len(gradients) < 2 * f:
    return f"Invalid number of Byzantine gradients to tolerate, got f = {f!r}, expected 1 ≤ f ≤ {(len(gradients)) // 2}"

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rule (pytorch version)
method_name = "trimBulyan"
register(method_name, aggregate, check, upper_bound=None, influence=None)