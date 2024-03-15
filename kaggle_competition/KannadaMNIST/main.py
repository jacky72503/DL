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


class CharData(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.X = images
        self.y = labels

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = np.array(self.X.iloc[idx, :], dtype='uint8').reshape([28, 28, 1])
        if self.transform is not None:
            img = self.transform(img)

        if self.y is not None:
            y = np.zeros(10, dtype='float32')
            y[self.y.iloc[idx]] = 1
            return img, y
        else:
            return img


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def criterion(input, target, size_average=True):
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l


def train(epoch, history=None):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader),
                optimizer.state_dict()['param_groups'][0]['lr'],
                loss.data))
    exp_lr_scheduler.step()


def evaluate(epoch, history=None):
    model.eval()
    loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss += criterion(output, target, size_average=False).data

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).cpu().sum().numpy()

    loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    if history is not None:
        history.loc[epoch, 'val_loss'] = loss.cpu().numpy()
        history.loc[epoch, 'val_accuracy'] = accuracy

    print('Val loss: {:.4f}, Val accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(val_loader.dataset),
        100. * accuracy))


config = configparser.ConfigParser()
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
config.read(os.path.join(ROOT.parent.parent.absolute(), 'config.ini'))
DATA_PATH = os.path.join(config['PATH']['DATA'], 'KannadaMNIST')

df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
target = df_train['label']
df_train.drop('label', axis=1, inplace=True)

X_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
X_test.drop('id', axis=1, inplace=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 594
n_epochs = 100

X_train, X_val, y_train, y_val = train_test_split(df_train, target, stratify=target, random_state=42, test_size=0.01)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

train_dataset = CharData(X_train, y_train, train_transform)
val_dataset = CharData(X_val, y_val, test_transform)
test_dataset = CharData(X_test, transform=test_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = Net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=n_epochs // 4, gamma=0.1)

history = pd.DataFrame()

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train(epoch, history)
    evaluate(epoch, history)

torch.save(model.state_dict(), 'KannadaMNIST.pt')

history['train_loss'].plot();
plt.show()

history.dropna()['val_accuracy'].plot();
plt.show()

print('max', history.dropna()['val_accuracy'].max())
print('max in last 5', history.dropna()['val_accuracy'].iloc[-5:].max())
print('avg in last 5', history.dropna()['val_accuracy'].iloc[-5:].mean())

model.eval()
predictions = []

for idx, data in enumerate(tqdm(test_loader)):
    data = data.to(device)
    output = model(data).max(dim=1)[1]
    predictions += list(output.data.cpu().numpy())

submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
submission['label'] = predictions
submission.to_csv('submission.csv', index=False)
