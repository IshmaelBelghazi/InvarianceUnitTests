#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch

class Model(torch.nn.Module):
  HPARAMS = dict()

  def get_hparams(cls, mode):
    if mode == "default":
      return {k: v[0] for k, v in cls.HPARAMS.items()}
    elif mode == "random":
      return = {k: v[1]() for k, v in self.HPARAMS.items()}
    return None

  def __init__(self, in_features, out_features, task, hparams):
      super().__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.task = task
      self.hparams = hparams

      # network architecture
      # self.network = torch.nn.Linear(in_features, out_features)
      self.network = torch.nn.LazyLinear(out_features)


      # loss
      if self.task == "regression":
          self.loss = torch.nn.MSELoss()
      else:
          self.loss = torch.nn.BCEWithLogitsLoss()
      # callbacks
      self.callbacks = {}
      for key in ["errors"]:
          self.callbacks[key] = {
              "train": [],
              "validation": [],
              "test": []
          }

if __name__ == '__main__':
  pass
