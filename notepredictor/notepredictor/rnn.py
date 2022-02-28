import math

import torch
from torch import nn
import torch.nn.functional as F

import geotorch
from geotorch.so import torus_init_

# ExpRNN adapted from the MIT licensed geotorch package
# https://github.com/Lezcano/geotorch/blob/master/examples/sequential_mnist.py

class modrelu(nn.Module):
    def __init__(self, features):
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class ExpRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, 
            constraint='almostorthogonal', **constraint_params):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent_kernel = nn.Linear(
            hidden_size, hidden_size, bias=False)
        self.input_kernel = nn.Linear(input_size, hidden_size)
        self.nonlinearity = modrelu(hidden_size)

        if constraint == "orthogonal":
            geotorch.orthogonal(self.recurrent_kernel, "weight")
        elif constraint == "lowrank":
            geotorch.low_rank(self.recurrent_kernel, "weight", hidden_size)
        elif constraint == "almostorthogonal":
            geotorch.almost_orthogonal(
                self.recurrent_kernel, "weight", **constraint_params)
        else:
            raise ValueError("Unexpected constraints. Got {}".format(constraint))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")
        # Initialize the recurrent kernel à la Cayley, as having a block-diagonal matrix
        # seems to help in classification problems
        def init_(x):
            x.uniform_(0.0, math.pi / 2.0)
            c = torch.cos(x.data)
            x.data = -torch.sqrt((1.0 - c) / (1.0 + c))

        K = self.recurrent_kernel
        # We initialize it by assigning directly to it from a sampler
        K.weight = torus_init_(K.weight, init_=init_)

    def default_hidden(self, input_):
        return input_.new_zeros(input_.size(0), self.hidden_size, requires_grad=False)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        return self.nonlinearity(out)


class normalize(nn.Module):
    def __init__(self, size=None):
        super().__init__()

    def forward(self, inputs):
        return inputs/inputs.norm(2,-1,keepdim=True)

class ExpRNN(nn.Module):
    def __init__(self, input_size, hidden_size, **kw):
        super().__init__()
        self.num_layers = kw.pop('num_layers', 1)
        self.batch_first = kw.pop('batch_first', True)
        self.batch_dim = 0 if self.batch_first else 1
        self.time_dim = 1 - self.batch_dim
        assert not kw.pop('bidirectional', None)

        feedforward = []
        if kw.pop('normalize', False): feedforward.append(normalize())
        dropout = kw.pop('dropout', 0)
        if dropout: feedforward.append(nn.Dropout(dropout))
        self.feedforward = nn.Sequential(*feedforward)
        
        self.cells = nn.ModuleList([ExpRNNCell(input_size, hidden_size, **kw)])
        for _ in range(self.num_layers-1):
            self.cells.append(ExpRNNCell(hidden_size, hidden_size, **kw))

    def forward(self, input, states=None):
        if states is None: 
            states = (input.new_zeros(
                self.num_layers, input.shape[self.batch_dim], self.cells[0].hidden_size),)
        assert len(states)==1
        state = list(states[0].unbind(0))
        result = []
        with geotorch.parametrize.cached():
            for x in torch.unbind(input, dim=self.time_dim):
                h = x
                for i,cell in enumerate(self.cells):
                    h = state[i] = cell(self.feedforward(h), state[i])
                result.append(h) 
        # stack top hidden states along time dim
        # and final states along layer dim (0)
        return torch.stack(result, self.time_dim), (torch.stack(state, 0),)


def rnn_shim(cls):
    """LSTM API for GRU and RNN.
    
    hidden state is first element of state tuple"""
    class shim(cls):
        def forward(self, input, states=(None,)):
            assert len(states)==1
            out, h = super().forward(input, *states)
            return out, (h,)
    return shim

GRU = rnn_shim(nn.GRU)
RNN = rnn_shim(nn.RNN)
LSTM = nn.LSTM


class GenericRNN(nn.Module):
    kind_cls = {
        'gru':GRU,
        'lstm':LSTM,
        'elman':RNN,
        'exprnn':ExpRNN
        }
    def __init__(self, kind, *a, **kw):
        super().__init__()
        if kw.get('bidirectional'): raise ValueError("""
            bidirectional GenericRNN not supported.
            """)
        cls = GenericRNN.kind_cls[kind]
        self.kind = kind
        self.rnn = cls(*a, **kw)

    def __getattr__(self, a):
        try:
            return  super().__getattr__(a)
        except AttributeError:
            return getattr(self.rnn, a)

    def forward(self, x, initial_state):
        """
        Args:
            x: Tensor[batch x time x channel] if batch_first else [time x batch x channel]
            initial_state: List[Tensor[layers x batch x hidden]]], list of components 
            with 0 being hidden state (e.g. 1 is cell state for LSTM). 
        Returns:
            hidden: hidden states of top layers Tensor[batch x time x hidden]
                or [time x batch x hidden]
            new_states: List[Tensor[layers x batch x hidden]]
        """
        hidden, final_state = self.rnn.forward(x, initial_state)  #forward or __call__?
        return hidden, final_state

    ## NOTE: individual time-step API might be useful, not actually needed yet though
    # def step(self, x, state):
    #     """
    #     Args:
    #         x: Tensor[batch x channel]
    #         state: List[Tensor[layers x batch x hidden]]], list of components 
    #         with 0 being hidden state (e.g. 1 is cell state for LSTM). 
    #     Returns:
    #         hidden: hidden state of top layer [batch x hidden]
    #         new_states: List[Tensor[layers x batch x hidden]]
    #     """
    #     time_idx = 1 if self.rnn.batch_first else 0
    #     x = x.unsqueeze(time_idx)
    #     hidden, state = self.forward(x, state)
    #     return hidden.squeeze(time_idx), state