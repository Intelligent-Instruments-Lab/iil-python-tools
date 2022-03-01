import math

import torch
from torch import nn
import torch.nn.functional as F

class ExpRNN(nn.Module):
    def __init__(self, input_size, hidden_size, **kw):
	    raise NotImplementedError("see `exprnn` branch")

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