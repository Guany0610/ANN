import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np


class DatasetCustom(Dataset):
    def __init__(self, data, transform=None):
        self.train_set = data[:, :len(data[0])-3]
        self.train_label = data[:, len(data[0])-3:len(data[0])]
        self.transform = transform

    def __getitem__(self, item):
        if self.transform is None:
            item_set = self.train_set[item]
            item_label = self.train_label[item]
        else:
            item_set = self.transform(self.train_set[item])
            item_label = self.transform(self.train_label[item])
        return item_set, item_label

    def __len__(self):
        return len(self.train_set)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(9, 3)

    def forward(self, x):
        return self.fc(x)
        # return nn.functional.softmax(self.fc(x), dim=0)


def Train(model, dataloader, criterion, optimizer, epochs):
    loss = 0
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            for i in range(len(inputs)):
                output = model(inputs[i].float())
                optimizer.zero_grad()
                loss = criterion(output, labels[i])
                loss.backward()
                optimizer.step()

                for weight in model.parameters():
                    weight.data.clamp_(0, 20)

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))


def Test(model, dataloader):
    total = 0
    correct = 0
    for inputs, labels in dataloader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs.float())
        _, predicts = torch.max(outputs, 1)
        _, results = torch.max(labels, 1)
        total += len(labels)
        correct += (predicts == results).sum()
    print('Accuracy:{}'.format(correct / total))


if __name__ == '__main__':

    sample = np.array([[0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0.],
                       [1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0.],
                       [1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1.]])
    sample = np.repeat(sample, 66, axis=0)
    noise = np.random.normal(0.0, 0.4, (len(sample), len(sample[0])-3))
    sample += np.column_stack((noise, np.zeros((len(noise), 3))))

    sample[:, :len(sample[0]) - 3][sample[:, :len(sample[0]) - 3] > 1.] = 1.
    sample[:, :len(sample[0]) - 3][sample[:, :len(sample[0]) - 3] < 0.] = 0.
    TotalDataset = DatasetCustom(sample, torch.DoubleTensor)

    TrainDatasetSize = int(0.7 * len(TotalDataset))
    TestDatasetSize = len(TotalDataset) - TrainDatasetSize
    TrainDataset, TestDataset = torch.utils.data.random_split(TotalDataset, [TrainDatasetSize, TestDatasetSize])

    TrainDataLoader = DataLoader(dataset=TrainDataset, batch_size=20, shuffle=True)
    TestDataLoader = DataLoader(dataset=TestDataset, batch_size=20, shuffle=True)

    ANN = NeuralNetwork().cuda()
    Criterion = nn.CrossEntropyLoss()
    Optimizer = optim.SGD(ANN.parameters(), lr=0.1, momentum=0.9)

    Train(ANN, TrainDataLoader, Criterion, Optimizer, 10)

    Test(ANN, TestDataLoader)

    # shit = ANN.parameters()
    for param in ANN.parameters():
        print(np.array(param.data.cpu()))

