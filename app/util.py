import numpy as np
import matplotlib.pyplot as plt
import imageio


def transform(data):
    return data / 255.0 * 0.99 + 0.01


def img_show(file, row):
    with open(file, "r") as f:
        data_list = f.readlines()
        row_data = np.asfarray(data_list[row].split(',')[1:])
        dim = np.sqrt(row_data.size).astype(int)
        img = row_data.reshape((dim, dim))
        plt.imshow(img, cmap='Greys', interpolation='None')
        plt.show()

def img_read(image_file):
    image_arr = imageio.imread(image_file, as_gray=True)
    dim = np.sqrt(np.array(image_arr).size).astype(int)
    image_data = 255.0 - image_arr.reshape((dim, dim))

    return image_data

if __name__ == "__main__":
    train_data_file = "mnist_dataset/mnist_train.csv"
    img_show(train_data_file, 10)

    test_data_file = "mnist_dataset/mnist_test.csv"
    img_show(test_data_file, 2)

    image_file = "images/1.png"
    plt.imshow(img_read(image_file), cmap='Greys', interpolation='None')
    plt.show()
