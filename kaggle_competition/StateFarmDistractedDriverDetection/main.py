import pandas as pd
import numpy as np
import os
import configparser
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import gc
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2 as cv
import glob
import wandb
import random as random



config = configparser.ConfigParser()
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
config.read(os.path.join(ROOT.parent.parent.absolute(), 'config.ini'))
DATA_PATH = os.path.join(config['PATH']['DATA'], 'SFDDD')

data = pd.read_csv(os.path.join(DATA_PATH, 'driver_imgs_list.csv'))

print(data.columns)
value_count = data.drop(columns='img').groupby('subject')['classname'].value_counts().unstack()

# plt.figure(figsize=(20, 6))
# value_count.plot(kind='bar', ax=plt.gca())
# plt.xlabel('subject')
# plt.ylabel('classname')
# plt.xticks(rotation=0)
# plt.subplots_adjust(bottom=0.2)
# plt.show()

imgs = [cv.imread(file) for file in glob.glob(os.path.join(DATA_PATH, 'imgs/train/c0/*.jpg'))]
print(imgs[0].shape)
