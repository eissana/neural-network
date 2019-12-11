import numpy as np
import scipy.special


class NeuralNetwork(object):
    def __init__(self, in_nodes, hid_nodes, out_nodes, learn_rate, epoch):
        self.__in_nodes = in_nodes
        self.__hid_nodes = hid_nodes
        self.__out_nodes = out_nodes
        self.__epoch = epoch

        self.__learn_rate = learn_rate

        self.__weights_in_hid = np.random.normal(0.0, 1. / np.sqrt(self.__hid_nodes), (self.__hid_nodes, self.__in_nodes))
        self.__weights_hid_out = np.random.normal(0.0, 1. / np.sqrt(self.__out_nodes), (self.__out_nodes, self.__hid_nodes))

        self.__activation = scipy.special.expit

    def train(self, inputs_list, targets_list):
        inputs = self.__convert_data(inputs_list)
        targets = self.__convert_data(targets_list)

        hid_outputs, final_outputs = self.__query(inputs)

        out_errors = targets - final_outputs
        hid_errors = np.dot(self.__weights_hid_out.T, out_errors)

        self.__weights_hid_out += self.__learn_rate * np.dot(out_errors * final_outputs * (1. - final_outputs), hid_outputs.T)
        self.__weights_in_hid += self.__learn_rate * np.dot(hid_errors * hid_outputs * (1. - hid_outputs), inputs.T)


    def query(self, inputs_list):
        inputs = self.__convert_data(inputs_list)
        _, final_outputs = self.__query(inputs)
        
        return final_outputs

    def __convert_data(self, data_list):
        return np.array(data_list, ndmin=2).T

    def __query(self, inputs):
        hid_inputs = np.dot(self.__weights_in_hid, inputs)
        hid_outputs = self.__activation(hid_inputs)

        final_inputs = np.dot(self.__weights_hid_out, hid_outputs)
        final_outputs = self.__activation(final_inputs)
        
        return hid_outputs, final_outputs


def __img_show(file, row):
    with open(file, "r") as f:
        import matplotlib.pyplot as plt

        data_list = f.readlines()
        img = np.asfarray(data_list[row].split(',')[1:]).reshape((28, 28))
        plt.imshow(img, cmap='Greys', interpolation='None')
        plt.show()


def __transform(data):
    data_list = data.split(',')
    label = data_list[0]
    return np.asfarray(data_list[1:]) / 255.0 * 0.99 + 0.01, label


def __main():
    train_data_file = "mnist_dataset/mnist_train_100.csv"
    # __img_show(train_data_file, 3)

    test_data_file = "mnist_dataset/mnist_test_10.csv"
    # __img_show(test_data_file, 1)

    in_nodes = 784 # number of features
    hid_nodes = 100
    out_nodes = 10 # number of digits (target classes)
    nn = NeuralNetwork(in_nodes=in_nodes, hid_nodes=hid_nodes, out_nodes=out_nodes, learn_rate=0.3, epoch=2)

    with open(train_data_file, "r") as f:
        train_data = f.readlines()

    for row in train_data:
        inputs, label = __transform(row)

        target_list = np.zeros(out_nodes) + 0.01
        target_list[int(label)] = 0.99

        nn.train(inputs, target_list)

    with open(test_data_file, "r") as f:
        test_data = f.readlines()

    for row in test_data:
        test_input, label = __transform(row)
        print(f"actual: {label}, predicted: {np.argmax(nn.query(test_input))}")


if __name__ == "__main__":
    __main()