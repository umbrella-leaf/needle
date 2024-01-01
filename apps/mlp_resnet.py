import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(dim, hidden_dim)
    norm1 = norm(hidden_dim)
    relu1 = nn.ReLU()
    dropout = nn.Dropout(p=drop_prob)
    linear2 = nn.Linear(hidden_dim, dim)
    norm2 = norm(dim)
    model = nn.Sequential(linear1, norm1, relu1, dropout, linear2, norm2)
    res = nn.Residual(model)
    relu2 = nn.ReLU()
    block = nn.Sequential(res, relu2)
    return block
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(dim, hidden_dim)
    relu = nn.ReLU()
    res_blocks = []
    for i in range(num_blocks):
      res_blocks.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    linear2 = nn.Linear(hidden_dim, num_classes)
    model = nn.Sequential(linear1, relu, *res_blocks, linear2)
    return model
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    total, hit = 0, 0
    total_loss = 0
    if opt:
      model.train()
      for idx, batch in enumerate(dataloader):
        images, labels = batch
        output = model(images)
        opt.reset_grad()
        loss = loss_func(output, labels)
        total_loss += loss.numpy()
        loss.backward()
        opt.step()
        hit += (output.numpy().argmax(1) == labels.numpy()).sum()
        total += labels.shape[0]
    else:
      model.eval()
      for idx, batch in enumerate(dataloader):
        images, labels = batch
        output = model(images)
        loss = loss_func(output, labels)
        total_loss += loss.numpy()
        hit += (output.numpy().argmax(1) == labels.numpy()).sum()
        total += labels.shape[0]
    average_loss = total_loss / (idx + 1)
    average_error = (total - hit) / total
    return average_error, average_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
      image_filename=data_dir + "/train-images-idx3-ubyte.gz",
      label_filename=data_dir + "/train-labels-idx1-ubyte.gz"
    )
    test_dataset = ndl.data.MNISTDataset(
      image_filename=data_dir + "/t10k-images-idx3-ubyte.gz",
      label_filename=data_dir + "/t10k-labels-idx1-ubyte.gz"
    )
    train_loader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_dataset, batch_size)
    model = MLPResNet(28*28, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(epochs):
      train_err, train_loss = epoch(train_loader, model, opt)
    test_err, test_loss = epoch(test_loader, model)
    return (train_err, train_loss, test_err, test_loss)
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
