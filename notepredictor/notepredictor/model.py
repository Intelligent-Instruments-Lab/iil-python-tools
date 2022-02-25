import torch
from torch import nn
import torch.nn.functional as F

# import geotorch

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


GRU = rnn_shim(nn.GRU)
RNN = rnn_shim(nn.RNN)
LSTM = nn.LSTM

class ExpRNN(nn.Module):
    pass

# class ExpRNNCell(nn.Module):
#     pass

class GenericRNN(nn.Module):
    kind_cls = {
        'gru':GRU,
        'lstm':LSTM,
        'elman':RNN,
        'exprnn':ExpRNN
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
        cls = GenericRNN.kind_cls[kind]
        self.rnn = cls(*a, **kw)

    def forward(self, x, initial_state):
        """
        Args:
            x: Tensor[batch x time x channel] if batch_first else [time x batch x channel]
            initial_states: List[Tensor[layers x batch x hidden]]], list of components 
            with 0 being hidden state (e.g. 1 is cell state for LSTM). 
        Returns:
            hidden: hidden states of top layers Tensor[batch x time x hidden]
                or [time x batch x hidden]
            new_states: List[Tensor[layers x batch x hidden]]
        """
        hidden, final_state = self.rnn.forward(x, initial_state)  #forward or __call__?
        return hidden, final_state

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



class PitchPredictor(nn.Module):
    defaults = dict(
        emb_size=128, hidden_size=512, domain_size=128, num_layers=1
    )
    def __init__(self, **kw):
        super().__init__()
        for k,v in PitchPredictor.defaults.items():
            if k in kw:
                setattr(self, k, kw[k])
            else:
                setattr(self, k, v)
        #TODO: fail on unconsumed kw
        extra_keys = kw.keys() - PitchPredictor.defaults.keys()
        if len(extra_keys):
            raise ValueError(f'unknown arguments: {extra_keys}')

        self.emb = nn.Embedding(self.domain_size, self.emb_size)
        self.proj = nn.Linear(self.hidden_size, self.domain_size)
        
        self.rnn = GenericRNN('gru', self.emb_size, self.hidden_size, 
            num_layers=self.num_layers, batch_first=True)
        
        # learnable initial state
        self.initial_state = nn.ParameterList([
             # layer x batch x hidden
            nn.Parameter(torch.randn(self.num_layers,1,self.hidden_size)*self.hidden_size**-0.5),
        ])

        # persistent state for inference
        for n,t in zip(self.cell_state_names(), self.initial_state):
            self.register_buffer(n, t.clone())

    def cell_state_names(self):
        return tuple(f'cell_state_{i}' for i in range(len(self.initial_state)))

    @property
    def cell_state(self):
        return tuple(getattr(self, n) for n in self.cell_state_names())
        
    def forward(self, notes):
        """
        Args:
            notes: LongTensor[batch, time]
        """
        x = self.emb(notes) # batch, time, emb_size
        ## broadcast intial state to batch size
        initial_state = tuple(
            t.expand(self.num_layers, x.shape[0], -1).contiguous() # 1 x batch x hidden
            for t in self.initial_state)
        h, _ = self.rnn(x, initial_state) #batch, time, hidden_size

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
        note = torch.LongTensor([[note]]) # 1x1 (batch, time)
        x = self.emb(note) # 1, 1, emb_size
        
        h, new_state = self.rnn(x, self.cell_state)
        for t,new_t in zip(self.cell_state, new_state):
            t[:] = new_t
        
        logits = self.proj(h) # 1, 1, hidden_size
        ret = logits.squeeze().softmax(0)

        if sample:
            ret = ret.multinomial(1).item()
        return ret
    
    def reset(self, start=True):
        """
        resets internal model state.
        """
        for n,t in zip(self.cell_state_names(), self.initial_state):
            getattr(self, n)[:] = t.detach()
        