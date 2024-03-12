"""The module.
"""
from typing import Optional, Tuple
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
from .nn_basic import ReLU, Sigmoid, Tanh


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = 1 / hidden_size
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k**0.5, high=k**0.5, 
                                        device=device, dtype=dtype), device=device, dtype=dtype)
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k**0.5, high=k**0.5, 
                                        device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = bias
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-k**0.5, high=k**0.5, 
                                            device=device, dtype=dtype), device=device, dtype=dtype)
            self.bias_hh = Parameter(init.rand(hidden_size, low=-k**0.5, high=k**0.5, 
                                            device=device, dtype=dtype), device=device, dtype=dtype)
        self.nonlinear = Tanh() if nonlinearity == "tanh" else ReLU()
        self.input_size = input_size
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X: Tensor, h: Optional[Tensor] = None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        out = h @ self.W_hh + X @ self.W_ih
        if self.bias:
            out = out + (self.bias_ih + self.bias_hh).reshape((1, self.hidden_size)).broadcast_to(out.shape)
        return self.nonlinear(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        rnn_cells = [RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) 
                     for _ in range(num_layers - 1)]
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)] + rnn_cells
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X: Tensor, h0: Optional[Tensor] = None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
        H = ops.split(X, 0)
        h0 = ops.split(h0, 0)
        hn = []
        for l in range(self.num_layers):
            H_list = []
            h = h0[l]
            for t in range(seq_len):
                h = self.rnn_cells[l](H[t], h)
                H_list.append(h)
            H = H_list
            hn.append(h)
        H = ops.stack(H, axis=0)
        hn = ops.stack(hn, axis=0)
        return H, hn
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = 1 / hidden_size
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-k**0.5, high=k**0.5, 
                                        device=device, dtype=dtype), device=device, dtype=dtype)
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-k**0.5, high=k**0.5, 
                                        device=device, dtype=dtype), device=device, dtype=dtype)
        self.bias = bias
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-k**0.5, high=k**0.5, 
                                            device=device, dtype=dtype), device=device, dtype=dtype)
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-k**0.5, high=k**0.5, 
                                            device=device, dtype=dtype), device=device, dtype=dtype)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.input_size = input_size
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION


    def forward(self, X: Tensor, h: Optional[Tuple[Tensor, Tensor]] = None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h, c = h
        
        out = h @ self.W_hh + X @ self.W_ih
        if self.bias:
            out = out + (self.bias_ih + self.bias_hh).reshape((1, 4*self.hidden_size)).broadcast_to(out.shape)
        i, f, g, o = ops.split(out.reshape((bs, 4, self.hidden_size)), 1)
        i, f, g, o = self.sigmoid(i), self.sigmoid(f), self.tanh(g), self.sigmoid(o)
        c_ = f*c + i*g
        h_ = o * self.tanh(c_)
        return h_, c_
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        lstm_cells = [LSTMCell(hidden_size, hidden_size, bias, device, dtype) 
                     for _ in range(num_layers - 1)]
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)] + lstm_cells
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, X: Tensor, h: Optional[Tuple[Tensor, Tensor]] = None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        H = ops.split(X, 0)
        h0 = ops.split(h0, 0)
        c0 = ops.split(c0, 0)
        hn = []
        cn = []
        for l in range(self.num_layers):
            H_list = []
            h = h0[l]
            c = c0[l]
            for t in range(seq_len):
                h, c = self.lstm_cells[l](H[t], (h, c))
                H_list.append(h)
            H = H_list
            hn.append(h)
            cn.append(c)
        H = ops.stack(H, axis=0)
        hn = ops.stack(hn, axis=0)
        cn = ops.stack(cn, axis=0)
        return H, (hn, cn)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, 
                                           mean=0, std=1, device=device, dtype=dtype),
                                dtype=dtype, device=device)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_onehot = init.one_hot(self.num_embeddings, x, device=x.device)
        out = x_onehot.reshape((seq_len*bs, self.num_embeddings)) @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION