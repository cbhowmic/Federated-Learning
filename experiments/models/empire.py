# coding: utf-8
###
 # @file   empire.py
 # @author Sébastien Rouault <sebastien.rouault@alumni.epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2019-2021 École Polytechnique Fédérale de Lausanne (EPFL).
 # See LICENSE file.
 #
 # @section DESCRIPTION
 #
 # The model from "Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation".
 # (The original paper did not include the CIFAR-100 variant.)
###

__all__ = ["cnn"]

import torch

# ---------------------------------------------------------------------------- #
# Simple convolutional model, for CIFAR-10/100 (3 input channels)

class _CNN(torch.nn.Module):
  """ Simple, small convolutional model.
  """

  def __init__(self, cifar100=False):
    """ Model parameter constructor.
    Args:
        cifar100 Build the CIFAR-100 variant (instead of the CIFAR-10)
    """
    super().__init__()
    # Build parameters
    self._c1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self._b1 = torch.nn.BatchNorm2d(self._c1.out_channels)
    self._c2 = torch.nn.Conv2d(self._c1.out_channels, 64, kernel_size=3, padding=1)
    self._b2 = torch.nn.BatchNorm2d(self._c2.out_channels)
    self._m1 = torch.nn.MaxPool2d(2)
    self._d1 = torch.nn.Dropout(p=0.25)
    self._c3 = torch.nn.Conv2d(self._c2.out_channels, 128, kernel_size=3, padding=1)
    self._b3 = torch.nn.BatchNorm2d(self._c3.out_channels)
    self._c4 = torch.nn.Conv2d(self._c3.out_channels, 128, kernel_size=3, padding=1)
    self._b4 = torch.nn.BatchNorm2d(self._c4.out_channels)
    self._m2 = torch.nn.MaxPool2d(2)
    self._d2 = torch.nn.Dropout(p=0.25)
    self._d3 = torch.nn.Dropout(p=0.25)
    if cifar100: # CIFAR-100
        self._f1 = torch.nn.Linear(8192, 256)
        self._f2 = torch.nn.Linear(self._f1.out_features, 100)
    else: # CIFAR-10
        self._f1 = torch.nn.Linear(8192, 128)
        self._f2 = torch.nn.Linear(self._f1.out_features, 10)

  def forward(self, x):
    """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
    activation = torch.nn.functional.relu
    flatten    = lambda x: x.view(x.shape[0], -1)
    logsoftmax = torch.nn.functional.log_softmax
    # Forward pass
    # print('input size', type(x), x.shape)
    x = self._c1(x)
    x = activation(x)
    x = self._b1(x)
    x = self._c2(x)
    x = activation(x)
    x = self._b2(x)
    x = self._m1(x)
    x = self._d1(x)
    x = self._c3(x)
    x = activation(x)
    x = self._b3(x)
    x = self._c4(x)
    x = activation(x)
    x = self._b4(x)
    x = self._m2(x)
    x = self._d2(x)
    x = flatten(x)
    x = self._f1(x)
    x = activation(x)
    x = self._d3(x)
    x = self._f2(x)
    x = logsoftmax(x, dim=1)
    return x


class _CNN_new(torch.nn.Module):
  """ Simple, small convolutional model.
  """

  def __init__(self, cifar100=False):
    """ Model parameter constructor.
    Args:
        cifar100 Build the CIFAR-100 variant (instead of the CIFAR-10)
    """
    super().__init__()
    # Build parameters
    self._c1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
    # self._b1 = torch.nn.BatchNorm2d(self._c1.out_channels)
    self._c2 = torch.nn.Conv2d(self._c1.out_channels, 64, kernel_size=4, stride=1, padding=0)
    # self._b2 = torch.nn.BatchNorm2d(self._c2.out_channels)
    self._m1 = torch.nn.MaxPool2d(kernel_size=3, stride=3)
    # self._d1 = torch.nn.Dropout(p=0.25)
    # self._c3 = torch.nn.Conv2d(self._c2.out_channels, 128, kernel_size=3, padding=1)
    # self._b3 = torch.nn.BatchNorm2d(self._c3.out_channels)
    # self._c4 = torch.nn.Conv2d(self._c3.out_channels, 128, kernel_size=3, padding=1)
    # self._b4 = torch.nn.BatchNorm2d(self._c4.out_channels)
    self._m2 = torch.nn.MaxPool2d(kernel_size=4, stride=4)
    # self._d2 = torch.nn.Dropout(p=0.25)
    # self._d3 = torch.nn.Dropout(p=0.25)
    self.relu = torch.nn.ReLU()
    if cifar100: # CIFAR-100
        self._f1 = torch.nn.Linear(8192, 256)
        self._f2 = torch.nn.Linear(self._f1.out_features, 100)
    else: # CIFAR-10
        self._f1 = torch.nn.Linear(64 * 4 * 4, 384)
        self._f2 = torch.nn.Linear(self._f1.out_features, 192)
        self.output = torch.nn.Linear(self._f2.out_features, 10)

  def forward(self, x):
    """ Model's forward pass.
    Args:
      x Input tensor
    Returns:
      Output tensor
    """
    # activation = torch.nn.functional.relu
    flatten    = lambda x: x.view(x.shape[0], -1)
    logsoftmax = torch.nn.functional.log_softmax
    # Forward pass
    # print('input size', type(x), x.shape)
    x = self._c1(x)
    x = self.relu(x)
    x = self._m1(x)
    # x = self._b1(x)
    x = self._c2(x)
    x = self.relu(x)
    # x = self._b2(x)
    x = self._m2(x)
    # x = self._d1(x)
    # x = self._c3(x)
    # x = activation(x)
    # x = self._b3(x)
    # x = self._c4(x)
    # x = activation(x)
    # x = self._b4(x)
    # x = self._m2(x)
    # x = self._d2(x)
    print(x.shape)
    x = flatten(x)
    # x = x.view(x.size(0), -1)
    print(x.shape)
    x = self._f1(x)
    x = self.relu(x)
    # x = self._d3(x)
    x = self._f2(x)
    x = self.relu(x)
    x = self.output(x)
    x = logsoftmax(x, dim=1)
    return x


class CIFAR10Net(torch.nn.Module):
    def __init__(self):
      super(CIFAR10Net, self).__init__()

      # Layer 1: Convolutional Layer
      self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
      self.relu1 = torch.nn.ReLU()
      self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=3)

      # Layer 2: Convolutional Layer
      self.conv2 = torch.nn.Conv2d(16, 64, kernel_size=4, stride=1, padding=1)
      self.relu2 = torch.nn.ReLU()
      self.maxpool2 = torch.nn.MaxPool2d(kernel_size=4, stride=4)

      # Layer 3: Fully Connected Layer
      self.fc1 = torch.nn.Linear(64 * 4, 384)
      self.relu3 = torch.nn.ReLU()

      # Layer 4: Fully Connected Layer
      self.fc2 = torch.nn.Linear(384, 192)
      self.relu4 = torch.nn.ReLU()

      # Output Layer with Softmax Activation
      self.output_layer = torch.nn.Linear(192, 10)  # 10 classes for CIFAR-10
      self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
      # print('in the forward pass')
      # Forward pass
      x = self.maxpool1(self.relu1(self.conv1(x)))
      x = self.maxpool2(self.relu2(self.conv2(x)))
      x = x.view(x.size(0), -1)  # Flatten the tensor
      # print('x size', x.shape)
      x = self.relu3(self.fc1(x))
      x = self.relu4(self.fc2(x))
      x = self.output_layer(x)
      x = self.softmax(x)
      return x

def cnn(*args, **kwargs):
  """ Build a new simple, convolutional model.
  Args:
    ... Forwarded (keyword-)arguments
  Returns:
    Convolutional model
  """
  global _CNN
  return _CNN(*args, **kwargs)
