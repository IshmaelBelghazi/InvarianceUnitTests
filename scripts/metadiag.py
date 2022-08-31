#!/usr/bin/env python3
import json
import random
import pytest

import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer
from torch.nn.modules.lazy import LazyModuleMixin

from .basemodel import Model


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

class ERM(Model):
    def __init__(self, in_features, out_features, task, hparams="default"):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))

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

def quotient(X, Y, mode=''):
  if not mode in ['pinv', 'transpose']:
    err_msg = f'Unrecognised quotient mode {mode}'
    raise ValueError(err_msg)
  if mode == 'pinv':
    return 0.5 * (X @ pinv(Y) + Y @ pinv(X))
  elif mode == 'transpose':
    return 0.5 * (X @ Y.T + Y @ X.T)

def cross_cov(Y, X):
    N = X.shape[0]
    mu_Y = Y.mean(dim=0, keepdims=True)
    Y_demeaned = Y - mu_Y
    mu_X = X.mean(dim=0, keepdims=True)
    X_demeaned = X - mu_X
    return (Y_demeaned.T @ X_demeaned) / (N * (N - 1))

class MetaDiag(LazyModuleMixin):
  def __init__(self, tau, pi=0.1, use_double=False):
    super(MetaDiag, self).__init__()
    self.pi = pi
    self.use_double = use_double
    self.tau = tau

    self.register_buffer('V', UninitializedBuffer())
    self.register_buffer('Lambda', UninitializedBuffer())
    self.register_buffer('P', UninitializedBuffer())

  def forward(self, input_):

    if (isinstance(self.V, UninitializedBuffer)
        or isinstance(self.Lambda, UninitializedBuffer)):
      err_msg = f'V or Lambda not initialized'
      raise RuntimeError(err_msg)

    if isinstance(self.P, UninitializedBuffer):
      with torch.no_grad():
        threshold = torch.quantile(self.Lambda, self.tau, interpolation='nearest')
        P = self.V[:, torch.where(self.Lambda <= threshold)[0]]
        self.P.materialize(P.shape, device=P.device, dtype=P.dtype)

      return input_ @ self.P

  def fit(self, train_envs):
    with torch.no_grad():

        covs = []
        crosses = []
        for (X, Y) in train_envs['envs']:

          if use_double:
            X, Y = X.double(), Y.double()

          covs += [sqrtm(torch.cov(X.T))]
          crosses += [cross_cov(Y.view(-1, 1), X)]

        Ds = covs + crosses
        Cs = [D.T @ D for D in Ds]

        if pi > 0.0:
            Cs = [
                C + pi * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
                for C in Cs
            ]

        mats = Cs
        S = 0.
        for i in range(len(mats)):
            for j in range(i + 1, len(mats)):
                S += quotient(mats[i], mats[j])

        factor = 2 / (len(mats) * (len(mats) - 1))
        S *= factor

        #print(f"S condition number is: {torch.linalg.cond(S)}")
        Lambda, V = torch.linalg.eig(S)

        if V.is_complex():
            if V.imag.norm(p="fro") > 1e-5:
                err_msg = f"Complex part of generalized eigen-system too large"
                print(err_msg)
                # raise RuntimeError(err_msg)

        V = V.real.float()
        if isinstance(self.V, UninitializedBuffer):
          self.V.materialize(V.shape, device=V.device, dtype=V.dtype)
          self.V.copy_(V)

        Lambda = Lambda.real.float()
        if isinstance(self.Lambda, UninitializedBuffer):
          self.Lambda.materialize(Lambda.shape, device=Lambda.device, dtype=Lambda.dtype)
          self.Lambda.copy_(Lambda)

        return Lambda, V

class MetaDiagModel(Model):


      HPARAMS = {}
      HPARAMS["lr"] = (1e-3, lambda: 10**random.uniform(-4, -2))
      HPARAMS['wd'] = (0., lambda: 10**random.uniform(-6, -2))
      HPARAMS['pi'] = (0.01, lambda: 0.01)
      HPARAMS['use_double'] = (True, lambda: True)
      HPARAMS['tau'] = (0.5, lambda: 0.5)

  def __init__(self, in_features, out_features, task, hparams):
      super().__init__(in_features, out_features, task, hparams)

      self.optimizer = torch.optim.Adam(
          self.network.parameters(),
          lr=self.hparams["lr"],
          weight_decay=self.hparams["wd"])

      self.encoder = MetaDiag(pi=self.HPARAMS['pi'],
                              tau=self.HPARAMS['tau'],
                              use_double=self.HPARAMS['use_double'])

    def predict(self, x):
        return self.network(self.encoder(x))


    def fit(self, envs, num_iterations, callback=False):

      self.encoder.fit(envs['train'])

      x = torch.cat([self.encoder(xe) for xe, ye in envs["train"]["envs"]])
      y = torch.cat([ye for xe, ye in envs["train"]["envs"]])

      for epoch in range(num_iterations):
          self.optimizer.zero_grad()
          self.loss(self.network(self.encoder(x)), y).backward()
          self.optimizer.step()

          if callback:
              # compute errors
              utils.compute_errors(self, envs)






if __name__ == '__main__':
  pass
