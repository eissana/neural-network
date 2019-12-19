import numpy as np
import scipy.special
import pandas as pd

from app.util import transform, img_read


class NeuralNetwork(object):
    def __init__(self, in_nodes, hid_nodes, out_nodes, learn_rate, epoch):
        self.__in_nodes = in_nodes
        self.__hid_nodes = hid_nodes
        self.__out_nodes = out_nodes
        self.__epoch = epoch

        self.__learn_rate = learn_rate

        np.random.seed(1)
        self.__weights_in_hid = np.random.normal(0.0, 1. / np.sqrt(self.__hid_nodes), (self.__hid_nodes, self.__in_nodes))
        self.__weights_hid_out = np.random.normal(0.0, 1. / np.sqrt(self.__out_nodes), (self.__out_nodes, self.__hid_nodes))

        self.__activation = scipy.special.expit

    def fit(self, train_x, train_y):
        for e in range(self.__epoch):
            for idx, (_, record) in enumerate(train_x.T.iteritems()):
                scaled = transform(record)
                target = self.__target(train_y[idx])

                self.__train(scaled, target)

                if (idx + 1) % 1000 == 0:
                    print(f'epoch [{e + 1}/{self.__epoch}], step [{idx + 1}/{train_x.shape[0]}]')

    def __train(self, inputs_list, targets_list):
        inputs = self.__convert_data(inputs_list)
        targets = self.__convert_data(targets_list)

        hid_outputs, final_outputs = self.__predict(inputs)

        out_errors = targets - final_outputs
        hid_errors = np.dot(self.__weights_hid_out.T, out_errors)

        self.__weights_hid_out += self.__learn_rate * np.dot(out_errors * final_outputs * (1. - final_outputs), hid_outputs.T)
        self.__weights_in_hid += self.__learn_rate * np.dot(hid_errors * hid_outputs * (1. - hid_outputs), inputs.T)


    def predict(self, test_x):
        predicted = pd.Series()
        for idx, record in test_x.T.iteritems():
            inputs = self.__convert_data(record)
            scaled = transform(inputs)
            _, final_outputs = self.__predict(scaled)
            predicted.at[idx] = np.argmax(final_outputs)
        
        return predicted

    def __convert_data(self, data_list):
        return np.array(data_list, ndmin=2).T

    def __predict(self, inputs):
        hid_inputs = np.dot(self.__weights_in_hid, inputs)
        hid_outputs = self.__activation(hid_inputs)

        final_inputs = np.dot(self.__weights_hid_out, hid_outputs)
        final_outputs = self.__activation(final_inputs)
        
        return hid_outputs, final_outputs

    def __target(self, label):
        target = np.zeros(self.__out_nodes) + 0.01
        target[label] = 0.99
        return target


def __run():
    from sklearn.metrics import confusion_matrix, accuracy_score
    import pickle
    import os.path
    from app.__init__ import MODELS_DIR

    model_file = f"{MODELS_DIR}/nn-.pkl"

    if os.path.isfile(model_file):
        with open(model_file, 'rb') as f:
            net = pickle.load(f)
    else:
        in_nodes = 784 # number of features
        hid_nodes = 100
        out_nodes = 10 # number of digits (target classes)

        train_data_file = "mnist_dataset/mnist_train.csv"
        train_df = pd.read_csv(train_data_file, header=None)
        train_x = train_df.iloc[:, 1:]
        train_y = train_df.iloc[:, 0]

        net = NeuralNetwork(in_nodes=in_nodes, hid_nodes=hid_nodes, out_nodes=out_nodes, learn_rate=0.3, epoch=2)
        net.fit(train_x, train_y)

        with open(model_file, 'wb') as f:
            pickle.dump(net, f)

    # digit = 1
    # test_x = pd.DataFrame(img_read(f"images/{digit}.png").reshape((-1, 1)))
    # predicted = net.predict(test_x)
    # actual = pd.Series([digit])

    test_data_file = "mnist_dataset/mnist_test.csv"
    test_df = pd.read_csv(test_data_file, header=None)
    test_x = test_df.iloc[:, 1:]
    actual = test_df.iloc[:, 0]

    predicted = net.predict(test_x)

    print("accuracy score:", accuracy_score(actual, predicted))

    print("confusion matrix:\n", confusion_matrix(actual, predicted))

    result = pd.DataFrame(columns=['Actual', 'Predicted'])
    result['Actual'] = actual
    result['Predicted'] = predicted
    print(f"actual vs. predicted:\n {result}")


if __name__ == "__main__":
    __run()