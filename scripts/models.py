# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import json
import random
import numpy as np
import utils
from torch.autograd import grad

from basemodel import Model
from metadiag import MetaDiagModel
from projection_model import ProjectionModel

class ERM(Model):
    HPARAMS = dict()
    HPARAMS["lr"] = (1e-3, lambda: 10**random.uniform(-4, -2))
    HPARAMS['wd'] = (0., lambda: 10**random.uniform(-6, -2))

    def __init__(self, in_features, out_features, task, hparams):

        super().__init__(in_features, out_features, task, hparams)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def fit(self, envs, num_iterations, callback=False):
        x = torch.cat([xe for xe, ye in envs["train"]["envs"]])
        y = torch.cat([ye for xe, ye in envs["train"]["envs"]])

        for epoch in range(num_iterations):
            self.optimizer.zero_grad()
            self.loss(self.network(x), y).backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class IRM(Model):
  """
  Abstract class for IRM
  """
  HPARAMS = dict()
  HPARAMS["lr"] = (1e-3, lambda: 10**random.uniform(-4, -2))
  HPARAMS['wd'] = (0., lambda: 10**random.uniform(-6, -2))
  HPARAMS['irm_lambda'] = (0.9, lambda: 1 - 10**random.uniform(-3, -.3))

  def __init__(
          self, in_features, out_features, task, hparams, version=1):

      super().__init__(in_features, out_features, task, hparams)
      self.version = version

      self.network = self.IRMLayer(self.network)
      self.net_parameters, self.net_dummies = self.find_parameters(
          self.network)

      self.optimizer = torch.optim.Adam(
          self.net_parameters,
          lr=self.hparams["lr"],
          weight_decay=self.hparams["wd"])

  def find_parameters(self, network):
      """
      Alternative to network.parameters() to separate real parameters
      from dummmies.
      """
      parameters = []
      dummies = []

      for name, param in network.named_parameters():
          if "dummy" in name:
              dummies.append(param)
          else:
              parameters.append(param)
      return parameters, dummies

  class IRMLayer(torch.nn.Module):
      """
      Add a "multiply by one and sum zero" dummy operation to
      any layer. Then you can take gradients with respect these
      dummies. Often applied to Linear and Conv2d layers.
      """

      def __init__(self, layer):
          super().__init__()
          self.layer = layer
          self.dummy_mul = torch.nn.Parameter(torch.Tensor([1.0]))
          self.dummy_sum = torch.nn.Parameter(torch.Tensor([0.0]))

      def forward(self, x):
          return self.layer(x) * self.dummy_mul + self.dummy_sum

  def fit(self, envs, num_iterations, callback=False):
      for epoch in range(num_iterations):
          losses_env = []
          gradients_env = []
          for x, y in envs["train"]["envs"]:
              losses_env.append(self.loss(self.network(x), y))
              gradients_env.append(grad(
                  losses_env[-1], self.net_dummies, create_graph=True))

          # Average loss across envs
          losses_avg = sum(losses_env) / len(losses_env)
          gradients_avg = grad(
              losses_avg, self.net_dummies, create_graph=True)

          penalty = 0
          for gradients_this_env in gradients_env:
              for g_env, g_avg in zip(gradients_this_env, gradients_avg):
                  if self.version == 1:
                      penalty += g_env.pow(2).sum()
                  else:
                      raise NotImplementedError

          obj = (1 - self.hparams["irm_lambda"]) * losses_avg
          obj += self.hparams["irm_lambda"] * penalty

          self.optimizer.zero_grad()
          obj.backward()
          self.optimizer.step()

          if callback:
              # compute errors
              utils.compute_errors(self, envs)

  def predict(self, x):
      return self.network(x)


class IRMv1(IRM):
    """
    IRMv1 with penalty \sum_e \| \nabla_{w|w=1} \mR_e (\Phi \circ \vec{w}) \|_2^2
    From https://arxiv.org/abs/1907.02893v1 
    """

    def __init__(self, in_features, out_features, task, hparams):
        super().__init__(in_features, out_features, task, hparams, version=1)


class AndMask(Model):
  """
  AndMask: Masks the grqdients features for which
  the gradients signs across envs disagree more than 'tau'
  From https://arxiv.org/abs/2009.00329
  """

  HPARAMS = dict()
  HPARAMS["lr"] = (1e-3, lambda: 10**random.uniform(-4, 0))
  HPARAMS['wd'] = (0., lambda: 10**random.uniform(-5, 0))
  HPARAMS["tau"] = (0.9, lambda: random.uniform(0.8, 1))

  def __init__(self, in_features, out_features, task, hparams):
      super().__init__(in_features, out_features, task, hparams)

  def fit(self, envs, num_iterations, callback=False):
      for epoch in range(num_iterations):
          losses = [self.loss(self.network(x), y)
                    for x, y in envs["train"]["envs"]]
          self.mask_step(
              losses, list(self.parameters()),
              tau=self.hparams["tau"],
              wd=self.hparams["wd"],
              lr=self.hparams["lr"]
          )

          if callback:
              # compute errors
              utils.compute_errors(self, envs)

  def predict(self, x):
      return self.network(x)

  def mask_step(self, losses, parameters, tau=0.9, wd=0.1, lr=1e-3):
      with torch.no_grad():
          gradients = []
          for loss in losses:
              gradients.append(list(torch.autograd.grad(loss, parameters)))
              gradients[-1][0] = gradients[-1][0] / gradients[-1][0].norm()

          for ge_all, parameter in zip(zip(*gradients), parameters):
              # environment-wise gradients (num_environments x num_parameters)
              ge_cat = torch.cat(ge_all)

              # treat scalar parameters also as matrices
              if ge_cat.dim() == 1:
                  ge_cat = ge_cat.view(len(losses), -1)

              # creates a mask with zeros on weak features
              mask = (torch.abs(torch.sign(ge_cat).sum(0))
                      > len(losses) * tau).int()

              # mean gradient (1 x num_parameters)
              g_mean = ge_cat.mean(0, keepdim=True)

              # apply the mask
              g_masked = mask * g_mean

              # update
              parameter.data = parameter.data - lr * g_masked \
                  - lr * wd * parameter.data


class IGA(Model):
  """
  Inter-environmental Gradient Alignment
  From https://arxiv.org/abs/2008.01883v2
  """
  HPARAMS = dict()
  HPARAMS["lr"] = (1e-3, lambda: 10**random.uniform(-4, -2))
  HPARAMS['wd'] = (0., lambda: 10**random.uniform(-6, -2))
  HPARAMS['penalty'] = (1000, lambda: 10**random.uniform(1, 5))

  def __init__(self, in_features, out_features, task, hparams):
      super().__init__(in_features, out_features, task, hparams)

      self.optimizer = torch.optim.Adam(
          self.parameters(),
          lr=self.hparams["lr"],
          weight_decay=self.hparams["wd"])

  def fit(self, envs, num_iterations, callback=False):
      for epoch in range(num_iterations):
          losses = [self.loss(self.network(x), y)
                    for x, y in envs["train"]["envs"]]
          gradients = [
              grad(loss, self.parameters(), create_graph=True)
              for loss in losses
          ]
          # average loss and gradients
          avg_loss = sum(losses) / len(losses)
          avg_gradient = grad(avg_loss, self.parameters(), create_graph=True)

          # compute trace penalty
          penalty_value = 0
          for gradient in gradients:
              for gradient_i, avg_grad_i in zip(gradient, avg_gradient):
                  penalty_value += (gradient_i - avg_grad_i).pow(2).sum()

          self.optimizer.zero_grad()
          (avg_loss + self.hparams['penalty'] * penalty_value).backward()
          self.optimizer.step()

          if callback:
              # compute errors
              utils.compute_errors(self, envs)

  def predict(self, x):
      return self.network(x)


MODELS = {
    "ERM": ERM,
    "IRMv1": IRMv1,
    "ANDMask": AndMask,
    "IGA": IGA,
    "Oracle": ERM,
    "MetaDiag": MetaDiagModel,
    "Projection": ProjectionModel
}

if __name__ == '__main__':
  pass
