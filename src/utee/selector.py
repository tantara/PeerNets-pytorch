from utee import misc
import os

print = misc.logger.info

known_models = [
    'mnist', # 28x28
    'cifar10', 'cifar100', # 32x32
]

def mnist(cuda=True, model_root=None):
    print("Building and initializing mnist parameters")
    from mnist import model, dataset
    m = model.mnist(pretrained=os.path.join(model_root, 'mnist.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get

def cifar10(cuda=True, model_root=None):
    print("Building and initializing cifar10 parameters")
    from cifar import model, dataset
    m = model.cifar10(128, pretrained=os.path.join(model_root, 'cifar10.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get10

def cifar100(cuda=True, model_root=None):
    print("Building and initializing cifar100 parameters")
    from cifar import model, dataset
    m = model.cifar100(128, pretrained=os.path.join(model_root, 'cifar100.pth'))
    if cuda:
        m = m.cuda()
    return m, dataset.get100

def select(model_name, **kwargs):
    assert model_name in known_models, model_name
    kwargs.setdefault('model_root', os.path.expanduser('/base/models'))
    return eval('{}'.format(model_name))(**kwargs)