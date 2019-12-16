import torch
import torch.nn as nn
import pandas as pd
from torch import optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from app.util import transform


class Generator(nn.Module):
    def __init__(self, in_nodes, hid_nodes, out_nodes, epoch):
        super(Generator, self).__init__()

        self.pipe = nn.Sequential(
            nn.Linear(in_nodes, hid_nodes[0]),
            nn.ReLU(),
            nn.Linear(hid_nodes[0], hid_nodes[1]),
            nn.ReLU(),
            nn.Linear(hid_nodes[1], out_nodes),
            nn.LogSoftmax(dim=1)
        )
        self.__epoch = epoch

    def forward(self, x):
        return self.pipe(x)


class NeuralNetwork(object):
    def __init__(self, in_nodes, hid_nodes, out_nodes, learn_rate, epoch):
        self.__model = Generator(in_nodes, hid_nodes, out_nodes, epoch)
        self.__gen_optimizer = optim.SGD(params=self.__model.parameters(), lr=learn_rate, momentum=0.9)

        self.__epoch = epoch
        self.__out_nodes = out_nodes


    def fit(self, train_data):
        objective = nn.NLLLoss()

        for epoch in range(self.__epoch):
            for idx, (x, y) in enumerate(train_data):
                x = x.view(x.shape[0], -1)
                output = self.__model(x)
                loss = objective(output, y)
                loss.backward()

                self.__gen_optimizer.step()
                self.__gen_optimizer.zero_grad()

                if (idx + 1) % 100 == 0:
                    print(f'epoch [{epoch + 1}/{self.__epoch}], step [{idx + 1}/{len(train_data)}], loss: {loss.item()}')


    def accuracy(self, test_data):
        self.__model.eval()

        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_data:
                x = x.view(x.shape[0], -1)
                output = self.__model(x)
                _, pred = torch.max(output.data, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        
        return correct / total

    def predict(self, test_data):
        actual = torch.LongTensor()
        predicted = torch.LongTensor()
        with torch.no_grad():
            for x, y in test_data:
                x = x.view(x.shape[0], -1)
                actual = torch.cat((actual, y))
                output = self.__model(x)
                _, pred = torch.max(output.data, 1)
                predicted = torch.cat((predicted, pred))
        return predicted, actual


def __run():
    from sklearn.metrics import confusion_matrix, accuracy_score
    import os.path

    model_file = "model/nn.pt"

    if os.path.isfile(model_file):
        with open(model_file, 'rb') as f:
            net = torch.load(f)
    else:
        in_nodes = 784 # number of features
        hid_nodes = [128, 64]
        out_nodes = 10 # number of digits (target classes)

        train_data = torchvision.datasets.MNIST('mnist_dataset', download=True, train=True, transform=transforms.ToTensor())
        train_data = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        net = NeuralNetwork(in_nodes=in_nodes, hid_nodes=hid_nodes, out_nodes=out_nodes, learn_rate=0.003, epoch=10)
        net.fit(train_data)

        with open(model_file, 'wb') as f:
            torch.save(net, f)

    test_data = torchvision.datasets.MNIST('mnist_dataset', download=True, train=False, transform=transforms.ToTensor())
    test_data = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    predicted, actual = net.predict(test_data)

    print("accuracy score:", accuracy_score(actual, predicted))
    print("confusion matrix:\n", confusion_matrix(actual, predicted))

    result = pd.DataFrame(columns=['Actual', 'Predicted'])
    result['Actual'] = actual
    result['Predicted'] = predicted
    print(f"actual vs. predicted:\n {result}")


if __name__ == "__main__":
    __run()