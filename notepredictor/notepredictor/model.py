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
    # support geotorch constraints
    # clean API for multiple layers, multiple cell states (e.g. LSTM)
    def __init__(self, kind, *a, **kw):
        super().__init__()
        if kw.get('bidirectional'): raise ValueError("""
            bidirectional GenericRNN not supported.
            """)
        cls = GenericRNN.kind_cls[kind]
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


class PitchPredictor(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, emb_size=128, hidden_size=512, domain_size=128, num_layers=1):
        """
        """
        super().__init__()

        self.start_token = domain_size-2
        self.end_token = domain_size-1

        self.emb = nn.Embedding(domain_size, emb_size)
        self.proj = nn.Linear(hidden_size, domain_size)
        
        self.rnn = GenericRNN('gru', emb_size, hidden_size, 
            num_layers=num_layers, batch_first=True)
        
        # learnable initial state
        self.initial_state = nn.ParameterList([
             # layer x batch x hidden
            nn.Parameter(torch.randn(num_layers,1,hidden_size)*hidden_size**-0.5),
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
            t.expand(self.rnn.num_layers, x.shape[0], -1).contiguous() # 1 x batch x hidden
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
        if start:
            self.predict(self.start_token)

    @classmethod
    def from_checkpoint(cls, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = cls(**checkpoint['kw']['model'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model
        