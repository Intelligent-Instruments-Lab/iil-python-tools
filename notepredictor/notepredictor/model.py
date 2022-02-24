import torch
from torch import nn
import torch.nn.functional as F

import geotorch

# shim torch RNN,GRU classes to have same API as LSTM
def rnn_shim(cls):
    """LSTM API for GRU and RNN.
    
    hidden state is first element of state tuple"""
    class shim(cls):
        def forward(self, input, states):
            assert len(states)==1
            out, h = super().forward(input, *states)
            return out, (h,)
    return shim

def cell_shim(cls):
    """LSTMCell API for GRUCell and RNNCell.
    
    hidden state is first element of state tuple"""
    class shim(cls):
        def forward(self, input, states):
            assert len(states)==1
            return (super().forward(input, *states),)
    return shim

GRU = rnn_shim(nn.GRU)
GRUCell = cell_shim(nn.GRUCell)
RNN = rnn_shim(nn.RNN)
RNNCell = cell_shim(nn.RNNCell)
LSTM = nn.LSTM
LSTMCell = nn.LSTMCell

class ExpRNN(nn.Module):
    pass

class ExpRNNCell(nn.Module):
    pass

class GenericRNN(nn.Module):
    kind_cls = {
        'gru':(GRU, GRUCell),
        'lstm':(LSTM, LSTMCell),
        'elman':(RNN, RNNCell),
        'exprnn':(ExpRNN, ExpRNNCell)
        }
    # desiderata:
    # fused forward, cell for inference
    # support geotorch constraints
    # clean API for multiple layers, multiple cell states (e.g. LSTM)
    def __init__(self, kind, *a, **kw):
        super().__init__()
        if kw.get('bidirectional'): raise ValueError("""
            bidirectional GenericRNN not supported.
            """)
        cls, cell_cls = GenericRNN.kind_cls[kind]
        self.rnn = cls(*a, **kw)

        num_layers = kw.pop('num_layers', 1)
        kw.pop('batch_first', None)
        self.cells = [cell_cls(*a, **kw) for _ in range(num_layers)]

        # share parameters
        for n, cell in enumerate(self.cells):
            cell.weight_ih = getattr(self.rnn, f'weight_ih_l{n}')
            cell.weight_hh = getattr(self.rnn, f'weight_hh_l{n}')
            cell.bias_ih = getattr(self.rnn, f'bias_ih_l{n}')
            cell.bias_hh = getattr(self.rnn, f'bias_hh_l{n}')

    def forward(self, x, initial_state):
        hidden, final_state = self.rnn.forward(x, initial_state)  #forward or __call__?
        return hidden, final_state

    def step(self, x, states):
        """
        Args:
            x: input
            states: List[List[Tensor]], outer list is layers, inner list is 
                components with 0 being hidden (e.g. 1 is cell state for LSTM)
        Returns:
            hidden: hidden state of top layers
            new_states: List[List[Tensor]]
                (stack along outer list for compatibility with `forward`)
        """
        hidden = x
        new_states = []
        for state, layer in zip(states, self.cells):
            state = layer(hidden, state)
            new_states.append(state)
            hidden = state[0]
        return hidden, new_states


class PitchPredictor(nn.Module):
    defaults = dict(
        emb_size=128, hidden_size=512, domain_size=128
    )
    def __init__(self, **kw):
        super().__init__()
        for k,v in PitchPredictor.defaults.items():
            if k in kw:
                setattr(self, k, kw[k])
            else:
                setattr(self, k, v)

        self.emb = nn.Embedding(self.domain_size, self.emb_size)
        self.proj = nn.Linear(self.hidden_size, self.domain_size)
        
        self.rnn = GenericRNN('gru', self.emb_size, self.hidden_size, batch_first=True)
        
        self.h0 = torch.nn.Parameter(
            torch.randn(1,self.hidden_size)*self.hidden_size**-0.5)
        
        self.cell_state = None

        
    def forward(self, notes):
        """
        Args:
            notes: LongTensor[batch, time]
        """
        x = self.emb(notes) # batch, time, emb_size
        h0 = self.h0[None].expand(1, x.shape[0], -1).contiguous(), # 1 x batch x hidden_size
        h, _ = self.rnn(x, h0) #batch, time, hidden_size
        logits = self.proj(h[:,:-1]) # batch, time-1, 128
        logits = F.log_softmax(logits, -1) # logits = logits - logits.logsumexp(-1, keepdim=True)
        targets = notes[:,1:,None] #batch, time-1, 1
        return {
            'log_probs': logits.gather(-1, targets)[...,0],
            'logits': logits
        }
    
    def predict(self, note, sample=True):
        """
        Args:
            note: int
            sample: bool
        Returns:
            int if `sample` else Tensor[domain_size]
        """
        note = torch.LongTensor([note]) # 1
        x = self.emb(note) # 1, emb_size
        
        if self.cell_state is None:
            self.cell_state = [[self.h0.detach().clone()]]
        
        self.cell_state = self.rnn.step(x, self.cell_state)
        h = self.cell_state[0]
        
        logits = self.proj(h) # 1, 128
        
        ret = logits[0].softmax(0)
        if sample:
            ret = ret.multinomial(1).item()
        return ret
    
    def reset(self, start=True):
        """
        resets internal model state.
        """
        self.cell_state = None
        