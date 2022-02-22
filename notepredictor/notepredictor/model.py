import torch
from torch import nn
import torch.nn.functional as F

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
        
        self.rnn = nn.GRU(self.emb_size, self.hidden_size, batch_first=True)
        self.rnn_cell = nn.GRUCell(self.emb_size, self.hidden_size)
        self.rnn_cell.weight_ih = self.rnn.weight_ih_l0
        self.rnn_cell.weight_hh = self.rnn.weight_hh_l0
        self.rnn_cell.bias_ih = self.rnn.bias_ih_l0
        self.rnn_cell.bias_hh = self.rnn.bias_hh_l0
        
        self.h0 = torch.nn.Parameter(
            torch.randn(1,self.hidden_size)*self.hidden_size**-0.5)
        
        self.h = None

        
    def forward(self, notes):
        """
        Args:
            notes: LongTensor[batch, time]
        """
        x = self.emb(notes) # batch, time, emb_size
        h, _ = self.rnn(x, self.h0[None].expand(1, x.shape[0], -1)) #batch, time, hidden_size
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
        
        if self.h is None:
            self.h = self.h0.detach().clone()
        
        self.h = self.rnn_cell(x, self.h)
        
        logits = self.proj(self.h) # 1, 128
        
        ret = logits[0].softmax(0)
        if sample:
            ret = ret.multinomial(1).item()
        return ret
    
    def reset(self):
        """
        resets internal model state.
        """
        self.h = None
        