"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time
device = ndl.cpu()

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # 图像获取
    with gzip.open(image_filename, mode="rb") as image_file:
        raw_data = image_file.read()
        num_total_bytes = len(raw_data)
        num_data_bytes = num_total_bytes - 16
        _, num_images, H, W, *data = struct.unpack(f">4i{num_data_bytes}B", raw_data)
        X = np.array(data, dtype=np.float32) / 255
        X = X.reshape(num_images, H, W, 1)
    # 标签获取
    with gzip.open(label_filename, mode="rb") as label_file:
        raw_data = label_file.read()
        num_total_bytes = len(raw_data)
        num_data_bytes = num_total_bytes - 8
        _, num_images, *data = struct.unpack(f">2i{num_data_bytes}B", raw_data)
        y = np.array(data, dtype=np.uint8)
    return X, y

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        is_train = False
        model.eval()
    else:
        is_train = True
        model.train()
    total_examples = 0
    avg_acc = 0
    avg_loss = 0
    for X, y in dataloader:
        num_examples = X.shape[0]
        logits = model(X)
        ls = loss_fn(logits, y)
        if is_train:
            opt.reset_grad()
            ls.backward()
            opt.step()
        loss = ls.numpy().item()
        acc = (np.sum(logits.numpy().argmax(axis=-1) == y.numpy()) / num_examples).item()
        avg_loss = (total_examples/(total_examples + num_examples)*avg_loss + 
                    num_examples/(total_examples + num_examples)*loss)
        avg_acc = (total_examples/(total_examples + num_examples)*avg_acc + 
                    num_examples/(total_examples + num_examples)*acc)
        
        total_examples += num_examples
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = loss_fn()

    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn, opt)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = loss_fn()
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        is_train = False
        model.eval()
    else:
        is_train = True
        model.train()
    total_examples = 0
    avg_acc = 0
    avg_loss = 0
    for i in range(0, len(data) - seq_len, seq_len):
        X, y = ndl.data.get_batch(data, i, seq_len, device, dtype)
        num_examples = X.shape[0] * X.shape[1]
        logits, _ = model(X)
        ls = loss_fn(logits, y)
        if is_train:
            opt.reset_grad()
            ls.backward()
            if clip is not None:
                opt.clip_grad_norm(clip)
            opt.step()
        loss = ls.numpy().item()
        acc = (np.sum(logits.numpy().argmax(axis=-1) == y.numpy()) / num_examples).item()
        avg_loss = (total_examples/(total_examples + num_examples)*avg_loss + 
                    num_examples/(total_examples + num_examples)*loss)
        avg_acc = (total_examples/(total_examples + num_examples)*avg_acc + 
                    num_examples/(total_examples + num_examples)*acc)
        
        total_examples += num_examples
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = loss_fn()

    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn, opt, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_fn = loss_fn()
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
