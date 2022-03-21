import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

from .rnn import GenericRNN
from .distributions import CensoredMixtureLogistic

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
        self.proj = nn.Linear(n,n)

    def forward(self, x):
        x = x[...,None] * self.fs
        return self.proj(x.sin())

class MixEmbedding(nn.Module):
    def __init__(self, n, domain=(0,1)):
        """
        Args:
            n (int): number of channels
            domain (Tuple[float])
        """
        super().__init__()
        self.domain = domain
        self.lo = nn.Parameter(torch.randn(n))
        self.hi = nn.Parameter(torch.randn(n))
    def forward(self, x):
        """
        Args:
            x: Tensor[...]
        Returns:
            Tensor[...,n]
        """
        x = (x - self.domain[0])/(self.domain[1] - self.domain[0])
        x = x[...,None]
        return self.hi * x + self.lo * (1-x)

# class SelfGated(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         a, b = x.chunk(2, -1)
#         return a * b.sigmoid()

# class SelfGatedMLP(nn.Module):
#     def __init__(self, input, hidden, output, layers, dropout=0):
#         super().__init__()
#         h = input
#         def get_dropout():
#             if dropout > 0:
#                 return (nn.Dropout(dropout),)
#             else:
#                 return tuple()
#         self.net = []
#         for _ in range(layers):
#             self.net.append(nn.Sequential(
#                 *get_dropout(), nn.Linear(h, hidden*2), SelfGated()))
#             h = hidden
#         self.net.append(nn.Linear(hidden, output))
#         self.net = nn.Sequential(*self.net)

#         with torch.no_grad():
#             self.net[-1].weight.mul_(1e-2)

#     def forward(self, x):
#         return self.net(x)

class ModalityTransformer(nn.Module):
    """Model joint distribution of modalities autoregressively with random permutations"""
    def __init__(self, input_size, hidden_size, heads=4, layers=1):
        super().__init__()
        self.net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_size, heads, hidden_size, norm_first=False
                ), layers)

    def forward(self, h, modes):
        """
        Args:
            modes: each a Tensor[batch x time x input_size]
        """
        x = [h]+modes
        batch_size = h.shape[0]*h.shape[1]
        # fold time into batch, stack modes
        x = torch.stack([
            item.reshape(batch_size,-1)
            for item in x
        ],0)
        # now "time"(mode) x "batch"(+time) x channel

        # generate a mask
        # upper triangular (i.e. diagonal and above is True, meaning masked)
        # except h position should attend to self
        n = len(modes)+1
        mask = x.new_ones((n,n), dtype=bool).triu()
        mask[0,0] = False 

        x = self.net(x, mask)
        return list(x.reshape(n, *h.shape).unbind(0))[1:]


class NotePredictor(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, 
            emb_size=256, 
            rnn_hidden=2048, rnn_layers=1, kind='gru', 
            ar_hidden=2048, ar_layers=1, ar_heads=4,
            dropout=0.1, 
            num_pitches=128, 
            time_bounds=(0,10), time_components=32, time_res=1e-2,
            vel_components=16
            ):
        """
        """
        super().__init__()

        self.start_token = num_pitches
        self.end_token = num_pitches+1

        self.pitch_domain = num_pitches+2

        self.time_dist = CensoredMixtureLogistic(
            time_components, time_res, 
            lo=time_bounds[0], hi=time_bounds[1], init='time')
        self.vel_dist = CensoredMixtureLogistic(
            vel_components, 1.0, lo=0, hi=127, init='velocity')
        
        # embeddings for inputs
        self.pitch_emb = nn.Embedding(self.pitch_domain, emb_size)
        self.time_emb = SineEmbedding(emb_size)
        self.vel_emb = MixEmbedding(emb_size, (0, 127))

        # RNN backbone
        self.rnn = GenericRNN(kind, 
            3*emb_size, rnn_hidden, 
            num_layers=rnn_layers, batch_first=True, dropout=dropout)

        # learnable initial RNN state
        self.initial_state = nn.ParameterList([
             # layer x batch x hidden
            nn.Parameter(torch.randn(rnn_layers,1,rnn_hidden)*rnn_hidden**-0.5)
            for _ in range(2 if kind=='lstm' else 1)
        ])

        # projection from RNN state to distribution parameters
        self.h_proj = nn.Linear(rnn_hidden, emb_size)
        self.projections = nn.ModuleList([
            nn.Linear(emb_size, self.pitch_domain),
            nn.Linear(emb_size, self.time_dist.n_params, bias=False),
            nn.Linear(emb_size, self.vel_dist.n_params, bias=False)
        ])
        for p in self.projections:
            with torch.no_grad():
                p.weight.mul_(1e-2)

        self.xformer = ModalityTransformer(emb_size, ar_hidden, ar_heads, ar_layers)

        # persistent RNN state for inference
        for n,t in zip(self.cell_state_names(), self.initial_state):
            self.register_buffer(n, t.clone())

    def cell_state_names(self):
        return tuple(f'cell_state_{i}' for i in range(len(self.initial_state)))

    @property
    def cell_state(self):
        return tuple(getattr(self, n) for n in self.cell_state_names())
        
    def forward(self, pitches, times, velocities, validation=False):
        """
        teacher-forced probabilistic loss and diagnostics for training

        Args:
            pitches: LongTensor[batch, time]
            times: FloatTensor[batch, time]
            velocities: FloatTensor[batch, time]
        """
        batch_size, batch_len = pitches.shape

        pitch_emb = self.pitch_emb(pitches) # batch, time, emb_size
        time_emb = self.time_emb(times) # batch, time, emb_size
        vel_emb = self.vel_emb(velocities) # batch, time, emb_size

        embs = (pitch_emb, time_emb, vel_emb)

        x = torch.cat(embs, -1)[:,:-1] # skip last time position
        ## broadcast intial state to batch size
        initial_state = tuple(
            t.expand(self.rnn.num_layers, x.shape[0], -1).contiguous() # 1 x batch x hidden
            for t in self.initial_state)
        h, _ = self.rnn(x, initial_state) #batch, time, hidden_size

        # include initial hidden state for predicting first note
        h = torch.cat((
            self.initial_state[0][-1][None].expand(batch_size, 1, -1),
            h), -2)

        # fit all note factorizations at once.
        perm = torch.randperm(3)
        embs = [embs[i] for i in perm]
        mode_hs = self.xformer(self.h_proj(h), embs)
        mode_hs = [mode_hs[perm[i]] for i in perm]

        pitch_params, time_params, vel_params = [
            proj(h) for proj,h in zip(self.projections, mode_hs)]

        # get likelihoods
        pitch_logits = F.log_softmax(pitch_params, -1)
        pitch_targets = pitches[:,1:,None] #batch, time, 1
        pitch_log_probs = pitch_logits.gather(-1, pitch_targets)[...,0]

        time_targets = times# batch, time
        time_result = self.time_dist(time_params, time_targets)
        time_log_probs = time_result.pop('log_prob')

        vel_targets = velocities # batch, time
        vel_result = self.vel_dist(vel_params, vel_targets)
        vel_log_probs = vel_result.pop('log_prob')

        # should reduce over chunk dim with logsumexp?
        # i.e. average likelihood over factorizations, not LL?

        r = {
            'pitch_log_probs': pitch_log_probs,
            'time_log_probs': time_log_probs,
            'velocity_log_probs': vel_log_probs,
            **{'time_'+k:v for k,v in time_result.items()},
            **{'velocity_'+k:v for k,v in vel_result.items()}
        }
        if validation:
            with torch.no_grad():
                r['time_acc_30ms'] = (
                    self.time_dist.cdf(time_params, time_targets + 0.03)
                    - torch.where(time_targets - 0.03 >= 0,
                        self.time_dist.cdf(time_params, time_targets - 0.03),
                        time_targets.new_zeros([]))
                )
        return r
    
    # TODO: vel
    def predict(self, pitch, time, vel, force=(None, None, None)):
        """
        supply the most recent note and return a prediction for the next note.

        Args:
            pitch: int. MIDI number of current note.
            time: float. elapsed time since previous note.
            vel: float. (possibly dequantized) MIDI velocity from 0-127 inclusive.
            force: Tuple[Optional[Number]].

        Returns: dict of
            'pitch': int. predicted MIDI number of next note.
            'time': float. predicted time to next note.
            'velocity': float. unquantized predicted velocity of next note.
        """
        with torch.no_grad():
            pitch = torch.LongTensor([[pitch]]) # 1x1 (batch, time)
            time = torch.FloatTensor([[time]]) # 1x1 (batch, time)
            vel = torch.FloatTensor([[vel]]) # 1x1 (batch, time)

            embs = [
                self.pitch_emb(pitch), # 1, 1, emb_size
                self.time_emb(time),# 1, 1, emb_size
                self.vel_emb(vel)# 1, 1, emb_size
            ]
            x = torch.cat(embs, -1)
            
            h, new_state = self.rnn(x, self.cell_state)
            for t,new_t in zip(self.cell_state, new_state):
                t[:] = new_t

            h = self.h_proj(h)

            # TODO: permutations
            # TODO: optimize by removing unused positions
            # TODO: refactor with common distribution API
            pitch_h, = self.xformer(h, embs[:1])

            pitch_params = self.projections[0](pitch_h)
            pred_pitch = D.Categorical(logits=pitch_params).sample()

            embs[0] = self.pitch_emb(pred_pitch)
            _, time_h = self.xformer(h, embs[:2])

            time_params = self.projections[1](time_h)
            pred_time = self.time_dist.sample(time_params)

            embs[1] = self.time_emb(pred_time)
            _, _, vel_h = self.xformer(h, embs)

            vel_params = self.projections[2](vel_h)
            pred_vel = self.vel_dist.sample(vel_params)

            # ### TODO: generalize, move into sample
            # ### DEBUG
            # # pi only, fewer zeros:
            # log_pi, loc, s = (
            #     t.squeeze() for t in self.time_dist.get_params(time_params))
            # bias = 2#float('inf')
            # log_pi = torch.where(loc <= self.time_dist.res, log_pi-bias, log_pi)
            # idx = D.Categorical(logits=log_pi).sample()
            # pred_time = loc[idx].clamp(0,10)

            return {
                'pitch': pred_pitch.item(), 
                'time': pred_time.item(),
                'velocity': pred_vel.item(),
                'pitch_params': pitch_params,
                'time_params': time_params,
                'vel_params': vel_params
            }
    
    # TODO: start velocity
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
        