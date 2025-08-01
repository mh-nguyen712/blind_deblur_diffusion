# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Config file for training NCSNv2 on bedroom."""

from configs.default_lsun_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.batch_size = 128
    training.sde = 'vesde'
    training.continuouse = False
    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'none'
    sampling.corrector = 'ald'
    sampling.n_steps_each = 3
    sampling.snr = 0.095
    # data
    data = config.data
    data.category = 'bedroom'
    data.image_size = 128
    # model
    model = config.model
    model.name = 'ncsnv2_128'
    model.scale_by_sigma = True
    model.sigma_max = 190
    model.num_scales = 1086
    model.ema_rate = 0.9999
    model.sigma_min = 0.01
    model.normalization = 'InstanceNorm++'
    model.nonlinearity = 'elu'
    model.nf = 128
    model.interpolation = 'bilinear'
    # optim
    optim = config.optim
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.amsgrad = False
    optim.eps = 1e-8
    optim.warmup = 0
    optim.grad_clip = -1

    return config
