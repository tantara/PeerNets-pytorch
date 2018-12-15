import torch.nn as nn
import torch.nn.functional as F

from utee import misc
print = misc.logger.info

class PeerRegularization(nn.Module):
    def __init__(self):
        super(PeerRegularization, self).__init__()
    
    def forward(self, x):
        return x