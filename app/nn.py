import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.transforms as transforms


def __run():
    in_nodes = 784 # number of features
    hid_nodes = [128, 64]
    out_nodes = 10 # number of digits (target classes)
    epoch = 2
    learn_rate = 0.003


    train_data = torchvision.datasets.MNIST('mnist_dataset', download=True, train=True, transform=transforms.ToTensor())
    train_data = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(in_nodes, hid_nodes[0]),
        nn.ReLU(),
        nn.Linear(hid_nodes[0], hid_nodes[1]),
        nn.ReLU(),
        nn.Linear(hid_nodes[1], out_nodes),
        nn.LogSoftmax(dim=1)
    )

    objective = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)    

    for e in range(epoch):
        for i, (x, y) in enumerate(train_data):
            x = x.view(x.shape[0], -1)
            output = model(x)
            loss = objective(output, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 100 == 0:
                print(f'epoch [{e + 1}/{epoch}], step [{i + 1}/{len(train_data)}], loss: {loss.item()}')

    model.eval()

    test_data = torchvision.datasets.MNIST('mnist_dataset', download=True, train=False, transform=transforms.ToTensor())
    test_data = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_data:
            x = x.view(x.shape[0], -1)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        print(f'Test Accuracy of the model on the {len(test_data)} test images: {100 * correct / total}')

    torch.save(model.state_dict(), 'model/nn.pt')       


if __name__ == "__main__":
    __run()