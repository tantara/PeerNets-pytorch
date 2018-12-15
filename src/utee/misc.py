import cv2
import os
import shutil
import pickle as pkl
import time
import numpy as np
import hashlib

class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)
logger = Logger()

print = logger.info
def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)

def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))

def str2img(str_b):
    return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
    return cv2.imencode('.jpg', img)[1].tostring()

def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()

def eval_model(model, ds, n_sample=None, ngpu=1):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    correct1, correct5 = 0, 0
    n_passed = 0
    model = model.eval()
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)
        data =  Variable(torch.FloatTensor(data)).cuda()
        indx_target = torch.LongTensor(target)
        output = model(data)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += float(idx_pred[:, :1].cpu().eq(idx_gt1).sum())
        correct5 += float(idx_pred[:, :5].cpu().eq(idx_gt5).sum())

        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def generate_noise(model, ds, rho, input_size, clip_min=0, clip_max=1, n_sample=None, ngpu=1, once=False):
    import tqdm
    import torch
    from utee.universal_perturbation import UniversalPerturbation
    from utee.pytorch_classifiers import PyTorchClassifier

    max_iter = 20 # default: 20
    norm = 2 # or 1: L1, 2: L2, and np.inf
    delta = 0.05 # default: 0.2

    n_sample = len(ds) if n_sample is None else n_sample
    n_total = 0
    eps_total = 0.0
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_total += len(data)

        data = data.reshape((data.shape[0], -1))
        data *= 255
        l2_norm = np.linalg.norm(data, axis=1)
        l2_sum = l2_norm.sum()
        eps_total += l2_sum

        if idx >= n_sample - 1:
            break

    eps = float(eps_total / n_total)
    print('eps : {:2f}'.format(eps))

    model = model.eval()
    ptc = PyTorchClassifier((clip_min, clip_max), model, None, None, (1, input_size, input_size), 10)
    up = UniversalPerturbation(ptc, attacker='deepfool', attacker_params={"max_iter": 100}, norm=norm, eps=rho*eps, max_iter=max_iter, delta=delta)

    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        data_adv = up.generate(data.numpy())
        noise = up.v
        fooling_rate = up.fooling_rate

        if once or idx >= n_sample - 1:
            break

    return noise, fooling_rate, eps

def attack_model(model, ds, noise, n_sample=None, ngpu=1):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    correct1, correct5 = 0, 0
    n_passed = 0
    model = model.eval()
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    noise = torch.from_numpy(noise)

    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)
        data += noise
        data =  Variable(torch.FloatTensor(data)).cuda()
        indx_target = torch.LongTensor(target)
        output = model(data)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += float(idx_pred[:, :1].cpu().eq(idx_gt1).sum())
        correct5 += float(idx_pred[:, :5].cpu().eq(idx_gt5).sum())

        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def load_state_dict(model, model_urls, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        raise KeyError('missing keys in state_dict: "{}"'.format(missing))
