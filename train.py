import os

import torch.optim as optim
import torch

from dataset import *
from model import *

if os.path.exists("./model.pt"):
    model = torch.load("./model.pt")
    print("Loaded model")
else:
    model = ByteT2T()
    print("Initialized model")

model(torch.randint(low=0, high=255, size=(1, 32)), torch.randint(low=0, high=255, size=(1, 32)))
