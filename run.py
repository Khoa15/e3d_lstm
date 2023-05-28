# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Main function to run the code."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from src.data_provider import datasets_factory
from src.models.model_factory import Model
import src.trainer as trainer
from src.utils import preprocess
import tensorflow as tf

import argparse

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train_data_paths', type=str, default='', help='train data paths.')
parser.add_argument('--valid_data_paths', type=str, default='', help='validation data paths.')
parser.add_argument('--save_dir', type=str, default='', help='dir to store trained net.')
parser.add_argument('--gen_frm_dir', type=str, default='', help='dir to store result.')
parser.add_argument('--is_training', type=bool, default=True, help='training or testing')
parser.add_argument('--dataset_name', type=str, default='mnist', help='The name of dataset.')
parser.add_argument('--input_length', type=int, default=10, help='input length.')
parser.add_argument('--total_length', type=int, default=20, help='total input and output length.')
parser.add_argument('--img_width', type=int, default=64, help='input image width.')
parser.add_argument('--img_channel', type=int, default=1, help='number of image channel.')
parser.add_argument('--patch_size', type=int, default=1, help='patch size on one dimension.')
parser.add_argument('--reverse_input', type=bool, default=False,
                    help='reverse the input/outputs during training.')
parser.add_argument('--model_name', type=str, default='e3d_lstm', help='The name of the architecture.')
parser.add_argument('--pretrained_model', type=str, default='', help='.ckpt file to initialize from.')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64',
                    help='COMMA separated number of units of e3d lstms.')
parser.add_argument('--filter_size', type=int, default=5, help='filter of a e3d lstm layer.')
parser.add_argument('--layer_norm', type=bool, default=True,
                    help='whether to apply tensor layer norm.')
parser.add_argument('--scheduled_sampling', type=bool, default=True,
                    help='for scheduled sampling')
parser.add_argument('--sampling_stop_iter', type=int,
                    default=50000,
                    help='for scheduled sampling.')
parser.add_argument('--sampling_start_value', type=float,
                    default=1.0,
                    help='for scheduled sampling.')
parser.add_argument('--sampling_changing_rate',
                    type=float,
                    default=0.00002,
                    help='for scheduled sampling.')
parser.add_argument('--lr', type=float,
                    default=0.001,
                    help='learning rate.')
parser.add_argument('--batch_size',
                    type=int,
                    default=8,
                    help='batch size for training.')
parser.add_argument('--max_iterations',
                    type=int,
                    default=80000,
                    help='max num of steps.')
parser.add_argument('--display_interval',
                    type=int,
                    default=1,
                    help='number of iters showing training loss.')
parser.add_argument('--test_interval',
                    type=int,
                    default=1000,
                    help='number of iters for test.')
parser.add_argument('--snapshot_interval',
                    type=int,
                    default=1000,
                    help='number of iters saving models.')
parser.add_argument('--num_save_samples',
                    type=int,
                    default=10,
                    help='number of sequences to be saved.')
parser.add_argument('--n_gpu',
                    type=int,
                    default=1,
                    help=('how many GPUs to distribute the training across.'))
parser.add_argument('--allow_gpu_growth',
                    type=bool,
                    default=True,
                    help=('allow gpu growth'))

args = parser.parse_args()
# train_data_paths = args.train_data_paths
# valid_data_paths = args.valid_data_paths
# save_dir = args.save_dir
# gen_frm_dir = args.gen_frm_dir
# is_training = args.is_training
# dataset_name = args.dataset_name
# input_length = args.input_length
# total_length = args.total_length
# img_width = args.img_width
# img_channel = args.img_channel
# patch_size = args.patch_size
# reverse_input = args.reverse_input
# model_name = args.model_name
# pretrained_model = args.pretrained_model
# num_hidden = [int(x) for x in args.num_hidden.split(',')]
# filter_size = args.filter_size
# layer_norm = args.layer_norm
# scheduled_sampling = args.scheduled_sampling
# sampling_stop_iter = args.sampling_stop_iter
# sampling_start_value = args.sampling_start_value
# sampling_changing_rate = args.sampling_changing_rate
# lr = args.lr
# batch_size = args.batch_size
# max_iterations = args.max_iterations
# display_interval = args.display_interval
# test_interval = args.test_interval
# snapshot_interval = args.snapshot_interval
# num_save_samples = args.num_save_samples
# n_gpu = args.n_gpu
def main():
  if tf.gfile.Exists(args.save_dir):
      tf.gfile.DeleteRecursively(args.save_dir)
      tf.gfile.MakeDirs(args.save_dir)
  if tf.gfile.Exists(args.gen_frm_dir):
      tf.gfile.DeleteRecursively(args.gen_frm_dir)
      tf.gfile.MakeDirs(args.gen_frm_dir)

  gpu_list = np.asarray(
      os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
  n_gpu = len(gpu_list)
  print('Initializing models')

  model = Model(args)

  if args.is_training:
      train_wrapper(model)
  else:
      test_wrapper(model)

def schedule_sampling(eta, itr):
    zeros = np.zeros(
        (args.batch_size, args.total_length - args.input_length - 1,
         args.img_width // args.patch_size, args.img_width // args.patch_size,
         args.patch_size**2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones(
        (args.img_width // args.patch_size, args.img_width // args.patch_size,
         args.patch_size**2 * args.img_channel))
    zeros = np.zeros(
        (args.img_width // args.patch_size, args.img_width // args.patch_size,
         args.patch_size**2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(
        real_input_flag,
        (args.batch_size, args.total_length - args.input_length - 1,
         args.img_width // args.patch_size, args.img_width // args.patch_size,
         args.patch_size**2 * args.img_channel))
    return eta, real_input_flag

def train_wrapper(model):
  if args.pretrained_model:
      model.load(args.pretrained_model)
  # load data
  train_input_handle, test_input_handle = datasets_factory.data_provider(
      args.dataset_name,
      args.train_data_paths,
      args.valid_data_paths,
      args.batch_size * args.n_gpu,
      args.img_width,
      seq_length=args.total_length,
      is_training=True)

  eta = args.sampling_start_value

  for itr in range(1, args.max_iterations + 1):
    if train_input_handle.no_batch_left():
        train_input_handle.begin(do_shuffle=True)
    ims = train_input_handle.get_batch()
    if args.dataset_name == 'penn':
        ims = ims['frame']
    ims = preprocess.reshape_patch(ims, args.patch_size)

    eta, real_input_flag = schedule_sampling(eta, itr)

    trainer.train(model, ims, real_input_flag, args, itr)

    if itr % args.snapshot_interval == 0:
        model.save(itr)

    if itr % args.test_interval == 0:
        trainer.test(model, test_input_handle, args, itr)

    train_input_handle.next()


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name,
        args.train_data_paths,
        args.valid_data_paths,
        args.batch_size * args.n_gpu,
        args.img_width,
        is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')


if __name__ == '__main__':
    main()

