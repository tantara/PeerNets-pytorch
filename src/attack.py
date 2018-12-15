import argparse
from collections import OrderedDict
import numpy as np
import os

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True

from utee import misc, selector

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='mnist', help='|'.join(selector.known_models))
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 64)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--model_root', default='/base/models', help='folder to save the model')
parser.add_argument('--data_root', default='/tmp/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('--clip_min', type=int, default=0, help='minimum clipping value')
parser.add_argument('--clip_max', type=int, default=1, help='maximum clipping value')
parser.add_argument('--n_classes', type=int, default=10, help='maximum clipping value')
parser.add_argument('--n_sample', type=int, default=100, help='number of samples to infer the scaling factor')
parser.add_argument('--rho', type=float, default=1.0, help='levels of perturbation')
args = parser.parse_args()

args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

assert torch.cuda.is_available(), 'no cuda'
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# load model and dataset fetcher
model_raw, ds_fetcher = selector.select(args.type, model_root=args.model_root)
print(model_raw)

# generate noise
train_ds, _ = ds_fetcher(args.n_sample, data_root=args.data_root, train=True, fix_shuffle=True, input_size=args.input_size)

noise_path = os.path.join(args.model_root, '{:s}-noise{:.1f}-{:d}.npy'.format(args.type, args.rho, args.n_sample))
fr_path = os.path.join(args.model_root, '{:s}-fr{:.1f}-{:d}.npy'.format(args.type, args.rho, args.n_sample))
eps_path = os.path.join(args.model_root, '{:s}-eps{:.1f}-{:d}.npy'.format(args.type, args.rho, args.n_sample))
if os.path.exists(noise_path):
    print("Load noise from "+noise_path)
    noise = np.load(noise_path)
    fooling_rate = np.load(fr_path)
    eps = np.load(eps_path)
else:
    noise, fooling_rate, eps = misc.generate_noise(model_raw, train_ds, rho=args.rho, input_size=args.input_size, ngpu=args.ngpu, once=True, clip_min=args.clip_min, clip_max=args.clip_max, n_classes=args.n_classes)
    print("Save noise , fooling_rate, eps to eg., "+noise_path)
    np.save(noise_path, noise)
    np.save(fr_path, fooling_rate)
    np.save(eps_path, eps)

res_str = "type={}, rho={:.1f}, eps={:.2f}, fooling_rate={:.4f}".format(args.type, args.rho, eps, fooling_rate)
print(res_str)

# attack model
val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
acc1, acc5 = misc.attack_model(model_raw, val_ds, noise, ngpu=args.ngpu)

# print sf
res_str = "type={}, acc1={:.4f}, acc5={:.4f}".format(args.type, acc1, acc5)
print(res_str)
with open('acc1_acc5.txt', 'a') as f:
    f.write(res_str + '\n')
