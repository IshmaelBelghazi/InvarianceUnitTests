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
  if not mode in ['symmetrized_product', 'product']:
    err_msg = f'Unrecognised quotient mode {mode}'
    raise ValueError(err_msg)
  elif mode == 'product':
    return X @ Y
  elif mode == 'symmetrized_product':
    return 0.5 * (X @ Y + Y @ X)

def cross_cov(Y, X):
    N = X.shape[0]
    mu_Y = Y.mean(dim=0, keepdims=True)
    Y_demeaned = Y - mu_Y
    mu_X = X.mean(dim=0, keepdims=True)
    X_demeaned = X - mu_X
    return (Y_demeaned.T @ X_demeaned) / (N * (N - 1))

class MetaDiag(nn.Module, LazyModuleMixin):
  def __init__(self, tau, quotient_mode, pi=0.1, use_double=False, symmetrize=False, include_diag=False):
    super().__init__()
    self.pi = pi
    self.use_double = use_double
    self.tau = tau
    self.quotient_mode = quotient_mode
    self.symmetrize = symmetrize
    self.include_diag = include_diag

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
      def normalize_spectrum(mat):
        Lambda, V = torch.linalg.eigh(mat)
        Vp = Lambda / sum(Lambda)
        return V @ torch.diag(Vp) @ V.T

      covs = []
      crosses = []
      for (X, Y) in train_envs['envs']:

        if self.use_double:
          X, Y = X.double(), Y.double()

        # covs += [sqrtm(torch.cov(X.T))]
        covs += [torch.cov(X.T)]
        assert Y.ndim == 2
        crosses += [cross_cov(Y, X)]

      Cs = covs + [D.T @ D for D in crosses]
      Cs = [normalize_spectrum(C) for C in Cs]

      if self.pi > 0.0:
          Cs = [
              C + self.pi * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
              for C in Cs
          ]

      S = 0.
      for i in range(len(Cs)):
          offset = 0 if self.include_diag else 1
          for j in range(i + offset, len(Cs)):
              S += quotient(sqrtm(Cs[i]), sqrtm(Cs[j]), mode=self.quotient_mode)
      factor = 2 / (len(Cs) * (len(Cs) - 1 + 2 * offset))
      S *= factor

      print(f"S_asym norm: {(0.5 * (S - S.T)).norm(p='fro')}.")
      eig_fun = torch.linalg.eig
      if self.symmetrize:
        S = 0.5 * (S + S.T)
        eig_fun = torch.linalg.eigh

      print(f"S condition number is: {torch.linalg.cond(S)}")
      Lambda, V = eig_fun(S)
      if V.is_complex():
          if V.imag.norm(p="fro") > 1e-5:
              err_msg = f"Complex part of generalized eigen-system too large"
              print(err_msg)
              # raise RuntimeError(err_msg)
      #if not self.symmetrize:
      V = V.real
      V = V.float()
      if isinstance(self.V, UninitializedBuffer):
        self.V.materialize(V.shape, device=V.device, dtype=V.dtype)
        self.V.copy_(V)

      # if not self.symmetrize:
      Lambda = Lambda.real.float()
      Lambda = Lambda.float()
      if isinstance(self.Lambda, UninitializedBuffer):
        self.Lambda.materialize(Lambda.shape, device=Lambda.device, dtype=Lambda.dtype)
        self.Lambda.copy_(Lambda)

      if isinstance(self.P, UninitializedBuffer):
        threshold = torch.quantile(self.Lambda, self.tau)# {{{{{interpolation='nearest')
        # Vmat =self.V[:, torch.where(self.Lambda >= threshold)[0]]
        #Vmat = self.V.T[:, torch.where(self.Lambda <= threshold)[0]]
        # Vmat = self.V[:, torch.where(self.Lambda <= threshold)[0]]
        # Vmat = self.V[torch.where(self.Lambda <= threshold)[0], :].T
        # P = Vmat
        #P = torch.linalg.pinv(Vmat)
        # scaler = self.Lambda.clone()
        scaler = torch.ones_like(self.Lambda)
        scaler[self.Lambda <= threshold] = 0
        Vmatp = torch.linalg.pinv(self.V) @ torch.diag(scaler)
        P = Vmatp.T
        self.P.materialize(P.shape, device=P.device, dtype=P.dtype)
        self.P.copy_(P)
        print(Lambda)

      return Lambda, V

class MetaDiagModel(Model):


  HPARAMS = {}
  HPARAMS["lr"] = (1e-3, lambda: 10**random.uniform(-4, -2))
  HPARAMS['wd'] = (0., lambda: 10**random.uniform(-6, -2))
  HPARAMS['pi'] = (0.0, lambda: 0.)
  HPARAMS['use_double'] = (True, lambda: True)
  HPARAMS['tau'] = (0.5, lambda: 0.5)
  HPARAMS['quotient_mode'] = ('product', lambda: 'transpose')
  HPARAMS['symmetrize'] = (False, lambda: True)
  HPARAMS['include_diag'] = (True, lambda: True)

  def __init__(self, in_features, out_features, task, hparams):
    super().__init__(in_features, out_features, task, hparams)

    self.optimizer = torch.optim.Adam(
        self.network.parameters(),
        lr=self.hparams["lr"],
        weight_decay=self.hparams["wd"])

    self.encoder = MetaDiag(pi=self.hparams['pi'],
                            tau=self.hparams['tau'],
                            use_double=self.hparams['use_double'],
                            quotient_mode=self.hparams['quotient_mode'],
                            symmetrize=self.hparams['symmetrize'],
                            include_diag=self.hparams['include_diag']
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
