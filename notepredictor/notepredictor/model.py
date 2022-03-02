import math

import torch
from torch import nn
import torch.nn.functional as F

from .rnn import GenericRNN
from .distributions import CensoredMixturePointyBoi

class SineEmbedding(nn.Module):
    def __init__(self, n, f0=1e-3, interval=2):
        super().__init__()
        self.n = n
        self.register_buffer('fs', f0 * interval**torch.arange(n) * 2 * math.pi)

    def forward(self, x):
        x = x[...,None] * self.fs
        return x.sin()

class NotePredictor(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, 
            pitch_emb_size=128, time_emb_size=16, hidden_size=512,
            num_layers=1, kind='gru', dropout=0, 
            num_pitches=128, 
            time_components=5, time_res=1e-2,
            ):
        """
        """
        super().__init__()

        self.start_token = num_pitches
        self.end_token = num_pitches+1

        self.pitch_domain = num_pitches+2

        # TODO: upper truncation?
        self.time_dist = CensoredMixturePointyBoi(time_components, time_res, 0, 10)
        
        # embeddings for inputs
        self.pitch_emb = nn.Embedding(self.pitch_domain, pitch_emb_size)
        self.time_emb = SineEmbedding(time_emb_size)

        # RNN backbone
        self.rnn = GenericRNN(kind, pitch_emb_size+time_emb_size, hidden_size, 
            num_layers=num_layers, batch_first=True, dropout=dropout)

        # learnable initial RNN state
        self.initial_state = nn.ParameterList([
             # layer x batch x hidden
            nn.Parameter(torch.randn(num_layers,1,hidden_size)*hidden_size**-0.5)
            for _ in range(2 if kind=='lstm' else 1)
        ])

        # projection from RNN state to distribution parameters
        self.time_proj = nn.Linear(hidden_size, self.time_dist.n_params, bias=False)
        self.pitch_proj = nn.Linear(hidden_size + time_emb_size, self.pitch_domain)
        with torch.no_grad():
            self.time_proj.weight.mul_(1e-2)
            self.pitch_proj.weight.mul_(1e-2)

        # persistent RNN state for inference
        for n,t in zip(self.cell_state_names(), self.initial_state):
            self.register_buffer(n, t.clone())

    def cell_state_names(self):
        return tuple(f'cell_state_{i}' for i in range(len(self.initial_state)))

    @property
    def cell_state(self):
        return tuple(getattr(self, n) for n in self.cell_state_names())
        
    def forward(self, pitches, times):
        """
        Args:
            pitches: LongTensor[batch, time]
            times: FloatTensor[batch, time]
        """

        time_emb = self.time_emb(times) # batch, time, time_emb_size
        pitch_emb = self.pitch_emb(pitches) # batch, time, note_emb_size

        x = torch.cat((pitch_emb, time_emb), -1)
        ## broadcast intial state to batch size
        initial_state = tuple(
            t.expand(self.rnn.num_layers, x.shape[0], -1).contiguous() # 1 x batch x hidden
            for t in self.initial_state)
        h, _ = self.rnn(x, initial_state) #batch, time, hidden_size

        # RNN hidden state -> time prediction
        time_params = self.time_proj(h[:,:-1]) # batch, time-1, time_params
        time_targets = times[:,1:] # batch, time-1
        time_result = self.time_dist(time_params, time_targets)
        time_log_probs = time_result.pop('log_prob')

        # RNN hidden state, time -> pitch prediction
        # pitch_params = h[...,:self.pitch_domain] + self.pitch_bias # CI
        pitch_params = self.pitch_proj(torch.cat((h[:,:-1], time_emb[:,1:]), -1))
        pitch_logits = F.log_softmax(pitch_params, -1)
        pitch_targets = pitches[:,1:,None] #batch, time-1, 1
        pitch_log_probs = pitch_logits.gather(-1, pitch_targets)[...,0]

        r = {
            'pitch_log_probs': pitch_log_probs,
            'time_log_probs': time_log_probs,
            **time_result
        }
        with torch.no_grad():
            r['time_acc_30ms'] = (
                self.time_dist.cdf(time_params, time_targets + 0.03)
                - torch.where(time_targets - 0.03 >= 0,
                    self.time_dist.cdf(time_params, time_targets - 0.03),
                    time_targets.new_zeros([]))
            )
        return r
    
    # TODO: time
    def predict(self, note, time, sample=True):
        """
        Args:
            note: int
            sample: bool
        Returns:
            int if `sample` else Tensor[num_notes+2]
        """
        note = torch.LongTensor([[note]]) # 1x1 (batch, time)
        x = self.note_emb(note) # 1, 1, emb_size
        
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
        