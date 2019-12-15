import torch
import torch.nn as nn
import pandas as pd
from torch import optim
import numpy as np

from app.util import transform


class Generator(nn.Module):
    def __init__(self, in_nodes, hid_nodes, out_nodes, epoch):
        super(Generator, self).__init__()

        self.pipe = nn.Sequential(
            nn.Linear(in_nodes, hid_nodes),
            nn.ReLU(),
            nn.Linear(hid_nodes, out_nodes),
            nn.Softmax(dim=0)
        )
        self.__epoch = epoch

    def forward(self, x):
        return self.pipe(x)


class NeuralNetwork(object):
    def __init__(self, in_nodes, hid_nodes, out_nodes, learn_rate, epoch):
        self.__nn = Generator(in_nodes, hid_nodes, out_nodes, epoch)
        self.__gen_optimizer = optim.Adam(params=self.__nn.parameters(), lr=learn_rate, betas=(0.5, 0.999))

        self.__epoch = epoch
        self.__out_nodes = out_nodes

    def fit(self, train_x, train_y):
        objective = loss = nn.MSELoss()

        for _ in range(self.__epoch):
            for idx, (_, record) in enumerate(train_x.iteritems()):
                scaled = transform(record)
                target = self.__target(train_y[idx])

                scaled = torch.tensor(scaled.values, dtype=torch.float)
                target = torch.tensor(target, dtype=torch.float)

                output = self.__nn(scaled)
                loss = objective(output, target)
                loss.backward()

                self.__gen_optimizer.step()
                self.__gen_optimizer.zero_grad()


    def __target(self, label):
        target = np.zeros(self.__out_nodes) + 0.01
        target[label] = 0.99
        return target


    def predict(self, test_x):
        predicted = pd.Series()
        for idx, record in test_x.iteritems():
            scaled = transform(record)
            scaled = torch.tensor(scaled.values, dtype=torch.float)
            output = self.__nn(scaled)
            predicted.at[idx] = np.argmax(output.detach().numpy())
        return predicted


def __run():
    from sklearn.metrics import confusion_matrix, accuracy_score
    import pickle
    import os.path

    model_file = "model/nn-2.pkl"

    if os.path.isfile(model_file):
        with open(model_file, 'rb') as f:
            neural_net = pickle.load(f)
    else:
        in_nodes = 784 # number of features
        hid_nodes = 100
        out_nodes = 10 # number of digits (target classes)

        train_data_file = "mnist_dataset/mnist_train.csv"
        train_df = pd.read_csv(train_data_file, header=None)
        train_x = train_df.iloc[:, 1:].T
        train_y = train_df.iloc[:, 0]

        neural_net = NeuralNetwork(in_nodes=in_nodes, hid_nodes=hid_nodes, out_nodes=out_nodes, learn_rate=0.3, epoch=2)
        neural_net.fit(train_x, train_y)

        with open(model_file, 'wb') as f:
            pickle.dump(neural_net, f)

    test_data_file = "mnist_dataset/mnist_test.csv"
    test_df = pd.read_csv(test_data_file, header=None)
    test_x = test_df.iloc[:, 1:].T
    actual = test_df.iloc[:, 0]

    predicted = neural_net.predict(test_x)

    print("accuracy score:", accuracy_score(actual, predicted))

    print("confusion matrix:\n", confusion_matrix(actual, predicted))

    result = pd.DataFrame(columns=['Actual', 'Predicted'])
    result['Actual'] = actual
    result['Predicted'] = predicted
    print(f"actual vs. predicted:\n {result}")


if __name__ == "__main__":
    __run()