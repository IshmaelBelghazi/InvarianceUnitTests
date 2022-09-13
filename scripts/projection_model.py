#!/usr/bin/env python3

#!/usr/bin/env python3
import json
import random
import pytest
import itertools
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.parameter import UninitializedBuffer
from torch.nn.modules.lazy import LazyModuleMixin

from basemodel import Model
import utils



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

def quotient(X, Y, mode):
  if mode not in ['product', 'quadratic']:
    err_msg = f'Unrecognized mode {mode}'
    raise ValueError(err_msg)

  if mode == 'product':
    return sqrtm(X) @ sqrtm(Y)

def cross_cov(Y, X):
    N = X.shape[0]
    mu_Y = Y.mean(dim=0, keepdims=True)
    Y_demeaned = Y - mu_Y
    mu_X = X.mean(dim=0, keepdims=True)
    X_demeaned = X - mu_X
    return (Y_demeaned.T @ X_demeaned) / (N * (N - 1))

class ProjectionDualDiag(nn.Module, LazyModuleMixin):
  def __init__(self, tau, quotient_mode, project, pi=0.1, use_double=False):
    super().__init__()
    self.pi = pi
    self.use_double = use_double
    self.tau = tau
    self.quotient_mode = quotient_mode
    self.project = project

    self.register_buffer('V', UninitializedBuffer())
    self.register_buffer('Lambda', UninitializedBuffer())
    self.register_buffer('P', UninitializedBuffer())

  def forward(self, input_):

    if (isinstance(self.V, UninitializedBuffer)
        or isinstance(self.Lambda, UninitializedBuffer)
        or isinstance(self.P, UninitializedBuffer)
        ):
      err_msg = f'V or Lambda or P not initialized'
      raise RuntimeError(err_msg)

    return input_ @ self.P

  def fit(self, train_envs):

    with torch.no_grad():

      # def normalize_spectrum(mat):
      #   Lambda, V = torch.linalg.eigh(mat)
      #   Vp = Lambda / sum(Lambda)
      #   return V @ torch.diag(Vp) @ V.T

      covs = []
      crosses = []
      for (X, Y) in train_envs['envs']:

        if self.use_double:
          X, Y = X.double(), Y.double()

        covs += [torch.cov(X.T)]
        assert Y.ndim == 2
        crosses += [cross_cov(Y, X)]

      Cs = covs + [D.T @ D for D in crosses]

      if self.pi > 0.0:
          Cs = [
              C + self.pi * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
              for C in Cs
          ]
      ncovs = len(covs)
      covs = Cs[:ncovs]
      crosses = Cs[ncovs:]

      CXX = sum(covs) / len(covs)
      CXYYX = sum(crosses) / len(crosses)
      S = quotient(CXX, CXYYX, mode=self.quotient_mode)

      if self.project:
        S = 0.5 * (S + S.T)


      #is_symmetric = torch.allclose(S, S.T)
      #eig_fun = torch.linalg.eigh if is_symmetric else torch.linalg.eigh
      eig_fun = torch.linalg.eig
      if self.project:
        eig_fun = torch.linalg.eigh

      print(f"S condition number is: {torch.linalg.cond(S)}")
      Lambda, V = eig_fun(S)
      if V.is_complex():
          if V.imag.norm(p="fro") > 1e-5:
              err_msg = f"Complex part of generalized eigen-system too large"
              print(err_msg)
              # raise RuntimeError(err_msg)
      # When not projecting on the space of symmetric matrices the solver
      # returns complex datatype
      if not self.project:
        V = V.real
        Lambda = Lambda.real

      V = V.float()
      if isinstance(self.V, UninitializedBuffer):
        self.V.materialize(V.shape, device=V.device, dtype=V.dtype)
        self.V.copy_(V)

      # if not self.symmetrize:
      Lambda = Lambda.float()
      if isinstance(self.Lambda, UninitializedBuffer):
        self.Lambda.materialize(Lambda.shape, device=Lambda.device, dtype=Lambda.dtype)
        self.Lambda.copy_(Lambda)

      if isinstance(self.P, UninitializedBuffer):
        threshold = torch.quantile(self.Lambda, self.tau)
        idx = torch.where(self.Lambda <= threshold)[0]
        Vmatp = V[:, idx]
        if self.project:
          P = Vmatp
        else:
          P = torch.linalg.pinv(Vmatp).T

        self.P.materialize(P.shape, device=P.device, dtype=P.dtype)
        self.P.copy_(P)
        print(Lambda)

      return Lambda, V

class ProjectionModel(Model):


  HPARAMS = {}
  HPARAMS["lr"] = (1e-3, lambda: 10**random.uniform(-4, -2))
  HPARAMS['wd'] = (0., lambda: 10**random.uniform(-6, -2))
  HPARAMS['pi'] = (0., lambda: 0.)
  HPARAMS['use_double'] = (True, lambda: True)
  HPARAMS['tau'] = (0.5, lambda: 0.5)
  HPARAMS['quotient_mode'] = ('quadratic', lambda: 'transpose')
  HPARAMS['project'] = (True, lambda: False)

  def __init__(self, in_features, out_features, task, hparams):
    super().__init__(in_features, out_features, task, hparams)

    self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=self.hparams["lr"],
        weight_decay=self.hparams["wd"])

    self.encoder = ProjectionDualDiag(pi=self.hparams['pi'],
                            tau=self.hparams['tau'],
                            use_double=self.hparams['use_double'],
                            quotient_mode=self.hparams['quotient_mode'],
                            project=self.hparams['project'],
                            )
  def predict(self, x):
      return self.network(self.encoder(x))


  def fit(self, envs, num_iterations, callback=False):

    self.encoder.fit(envs['train'])
    x = torch.cat([self.encoder(xe) for xe, _ in envs["train"]["envs"]])
    y = torch.cat([ye for _, ye in envs["train"]["envs"]])

    for epoch in range(num_iterations):
      self.optimizer.zero_grad()
      self.loss(self.network(x), y).backward()
      self.optimizer.step()

      if callback:
        # compute errors
        utils.compute_errors(self, envs)
    print(self.network.weight)

  def __str__(self):
    stem = self.__class__.__name__
    return stem





if __name__ == '__main__':
  pass
