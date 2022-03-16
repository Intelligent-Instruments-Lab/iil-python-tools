import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

from .rnn import GenericRNN
from .distributions import CensoredMixtureLogistic


# TODO: parameterized this wrong, meant to have increasing wavelength not frequency
class SineEmbedding(nn.Module):
    def __init__(self, n, w0=1e-3, interval=1.08):
        """
        Args:
            n (int): number of channels
            w0 (float): minimum wavelength in seconds
            interval (float): increase in frequency / decrease in wavelength per channel
        """
        super().__init__()
        self.n = n
        self.register_buffer('fs', interval**(-torch.arange(n)) / w0 * 2 * math.pi)

    def forward(self, x):
        x = x[...,None] * self.fs
        return x.sin()

class NotePredictor(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, 
            pitch_emb_size=128, time_emb_size=128, hidden_size=512,
            num_layers=1, kind='gru', dropout=0, 
            num_pitches=128, 
            time_bounds=(0,10), time_components=16, time_res=1e-2,
            ):
        """
        """
        super().__init__()

        self.start_token = num_pitches
        self.end_token = num_pitches+1

        self.pitch_domain = num_pitches+2

        # TODO: upper truncation?
        self.time_dist = CensoredMixtureLogistic(
            time_components, time_res, lo=time_bounds[0], hi=time_bounds[1])
        
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
        self.pitch_proj = nn.Linear(hidden_size, self.pitch_domain)
        self.cond_proj = nn.Linear(pitch_emb_size, hidden_size)
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
        teacher-forced probabilistic loss and diagnostics for training

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

        # IDEA: fit all factorizations at once.
        # add a 'missing' value for time / pitch / velocity as mode parameters
        # expand the batch to 6x wide with have ~/~/~, T/~/~, ~/P/~, T/P/~, ~/P/V, T/~/V inputs
        # the factorizations are:
        # _~~ -> T_~ -> TP_
        # _~~ -> T~_ -> T_V
        # ~_~ -> _P~ -> TP_
        # ~_~ -> ~P_ -> _PV
        # ~~_ -> _~V -> T_V
        # ~~_ -> ~_V -> _PV
        # i.e. ~~~ is counted 2x, and each other masked position is counted once
        

        # RNN hidden state, time -> pitch prediction
        pitch_params = self.pitch_proj(h[:,:-1])
        pitch_logits = F.log_softmax(pitch_params, -1)
        pitch_targets = pitches[:,1:,None] #batch, time-1, 1
        pitch_log_probs = pitch_logits.gather(-1, pitch_targets)[...,0]

        # RNN hidden state -> time prediction
        time_params = self.time_proj(
            self.cond_proj(pitch_emb[:,1:]).sigmoid() * h[:,:-1])
        # time_params = self.time_proj(
        #     torch.cat((h[:,:-1], pitch_emb[:,1:]), -1)
        #     ) # batch, time-1, time_params
        time_targets = times[:,1:]# batch, time-1
        time_result = self.time_dist(time_params, time_targets)
        time_log_probs = time_result.pop('log_prob')

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
    def predict(self, pitch, time):
        """
        supply the most recent note and return a prediction for the next note.

        Args:
            pitch: int. MIDI number of current note.
            time: float. elapsed time since previous note.
        Returns: dict of
            'pitch': int. predicted MIDI number of next note.
            'time': float. predicted time to next note.
        """
        with torch.no_grad():
            pitch = torch.LongTensor([[pitch]]) # 1x1 (batch, time)
            time = torch.FloatTensor([[time]]) # 1x1 (batch, time)
            x = torch.cat((
                self.pitch_emb(pitch), # 1, 1, pitch_emb_size
                self.time_emb(time)# 1, 1, time_emb_size
            ), -1)
            
            h, new_state = self.rnn(x, self.cell_state)
            for t,new_t in zip(self.cell_state, new_state):
                t[:] = new_t

            pitch_params = self.pitch_proj(h)
            pred_pitch = D.Categorical(logits=pitch_params).sample()
            
            time_params = self.time_proj(h*self.cond_proj(self.pitch_emb(pred_pitch)).sigmoid())
            # time_params = self.time_proj(torch.cat((
            #     h, self.pitch_emb(pred_pitch)
            # ), -1)) # 1, 1, time_params
            # TODO: importance sampling?
            pred_time = self.time_dist.sample(time_params).squeeze(0)

            ### TODO: generalize, move into sample
            ### DEBUG
            # pi only, fewer zeros:
            log_pi, loc, s = (
                t.squeeze() for t in self.time_dist.get_params(time_params))
            bias = 2#float('inf')
            log_pi = torch.where(loc <= self.time_dist.res, log_pi-bias, log_pi)
            idx = D.Categorical(logits=log_pi).sample()
            pred_time = loc[idx].clamp(0,10)

            return {
                'pitch': pred_pitch.item(), 
                'time': pred_time.item(),
                'pitch_params': pitch_params,
                'time_params': time_params
            }
    
    def reset(self, start=True):
        """
        resets internal model state.
        Args:
            start: if True, send a start token through the model with dt=0
                   but discard the prediction
        """
        for n,t in zip(self.cell_state_names(), self.initial_state):
            getattr(self, n)[:] = t.detach()
        if start:
            self.predict(self.start_token, 0.)

    @classmethod
    def from_checkpoint(cls, path):
        """
        create a Predictor from a checkpoint file containing hyperparameters and 
        model weights.
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = cls(**checkpoint['kw']['model'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model
        