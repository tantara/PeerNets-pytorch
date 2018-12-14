from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utee import misc
print = misc.logger.info


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 6, 5, 1, 2)),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(6, 16, 5, 1)),
            ('pool2', nn.MaxPool2d(2))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16*5*5, 120)),
            ('fc2', nn.Linear(120, 84)),
            ('fc3', nn.Linear(84, 10)),
        ]))
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features.forward(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mnist(pretrained=None):
    model = LeNet5()
    if pretrained is not None:
        state_dict = torch.load(pretrained)
        model.load_state_dict(state_dict)
    return model