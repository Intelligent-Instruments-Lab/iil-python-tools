import math

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

    def forward(self, x):
        x = x[...,None] * self.fs
        return x.sin()

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
        x = (x - self.domain[0])*(self.domain[1] - self.domain[0])
        x = x[...,None]
        return self.hi * x + self.lo * (1-x)

class SelfGated(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a, b = x.chunk(2, -1)
        return a * b.sigmoid()

class SelfGatedMLP(nn.Module):
    def __init__(self, input, hidden, output, layers, dropout=0):
        super().__init__()
        h = input
        def get_dropout():
            if dropout > 0:
                return (nn.Dropout(dropout),)
            else:
                return tuple()
        self.net = []
        for _ in range(layers):
            self.net.append(nn.Sequential(
                *get_dropout(), nn.Linear(h, hidden*2), SelfGated()))
            h = hidden
        self.net.append(nn.Linear(hidden, output))
        self.net = nn.Sequential(*self.net)

        with torch.no_grad():
            self.net[-1].weight.mul_(1e-2)

    def forward(self, x):
        return self.net(x)

class NotePredictor(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, 
            pitch_emb_size=128, time_emb_size=128, vel_emb_size=128, 
            hidden_size=512, num_layers=1, kind='gru', dropout=0, 
            num_pitches=128, 
            time_bounds=(0,10), time_components=16, time_res=1e-2,
            vel_components=8
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
        self.pitch_emb = nn.Embedding(self.pitch_domain, pitch_emb_size)
        self.time_emb = SineEmbedding(time_emb_size)
        self.vel_emb = MixEmbedding(vel_emb_size, (0, 127))

        self.pitch_missing = nn.Parameter(torch.randn(pitch_emb_size))
        self.time_missing = nn.Parameter(torch.randn(time_emb_size))
        self.vel_missing = nn.Parameter(torch.randn(vel_emb_size))

        # RNN backbone
        self.rnn = GenericRNN(kind, 
            pitch_emb_size+time_emb_size+vel_emb_size, hidden_size, 
            num_layers=num_layers, batch_first=True, dropout=dropout)

        # learnable initial RNN state
        self.initial_state = nn.ParameterList([
             # layer x batch x hidden
            nn.Parameter(torch.randn(num_layers,1,hidden_size)*hidden_size**-0.5)
            for _ in range(2 if kind=='lstm' else 1)
        ])

        # projection from RNN state to distribution parameters
        self.param_proj = SelfGatedMLP(
            pitch_emb_size+time_emb_size+vel_emb_size+hidden_size, 
            hidden_size//2,
            self.pitch_domain+self.time_dist.n_params+self.vel_dist.n_params,
            layers=2, dropout=dropout)

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

        pitch_emb = self.pitch_emb(pitches) # batch, time, pitch_emb_size
        time_emb = self.time_emb(times) # batch, time, time_emb_size
        vel_emb = self.vel_emb(velocities) # batch, time, vel_emb_size

        x = torch.cat((pitch_emb, time_emb, vel_emb), -1)
        ## broadcast intial state to batch size
        initial_state = tuple(
            t.expand(self.rnn.num_layers, x.shape[0], -1).contiguous() # 1 x batch x hidden
            for t in self.initial_state)
        h, _ = self.rnn(x, initial_state) #batch, time, hidden_size

        # IDEA: fit all factorizations at once.
        # add 'missing' value for time / pitch / velocity as model parameters
        # expand the batch to 6x wide with ~/~/~, T/~/~, ~/P/~, ~/~/V, T/P/~, ~/P/V, T/~/V inputs
        # the factorizations are:
        # _~~ -> T_~ -> TP_
        # _~~ -> T~_ -> T_V
        # ~_~ -> _P~ -> TP_
        # ~_~ -> ~P_ -> _PV
        # ~~_ -> _~V -> T_V
        # ~~_ -> ~_V -> _PV
        # i.e. the fully masked positions are counted 2x, 
        # the single-masked positions are counted 2x,
        # and each double-masked position is counted once

        masks = [
            [2,  0, 0, 2,  0, 1, 1], #pitch
            [2,  2, 0, 0,  1, 0, 1], #time
            [2,  0, 2, 0,  1, 1, 0]  #velocity
        ]

        def mask_cat(missing, present, mask):
            missing = missing[None,None].expand(batch_size, batch_len-1, -1)
            return torch.cat([
                present if m==0 else missing for m in mask
            ], 0)
        
        pitch_features = mask_cat(self.pitch_missing, pitch_emb[:,1:], masks[0])
        time_features = mask_cat(self.time_missing, time_emb[:,1:], masks[1])
        vel_features = mask_cat(self.vel_missing, vel_emb[:,1:], masks[2])
        
        features = torch.cat((
            pitch_features, time_features, vel_features, h[:,:-1].repeat(7,1,1)
            ), -1) # cat along feature dim

        dist_params = self.param_proj(features) # combine features with h 

        # split again into time/pitch/vel params
        dist_params = dist_params.split([
            self.pitch_domain, self.time_dist.n_params, self.vel_dist.n_params
            ], -1)

        # chunk into 7 and discard unmasked positions;
        # stack the masked positions along new first dim
        pitch_params, time_params, vel_params = (
            torch.cat([
                ch[None].expand(m, -1, -1, -1) 
                for m,ch in zip(mask, dp.chunk(7, 0)) if m>0
                ], 0)
            for mask,dp in zip(masks, dist_params)
        )

        # get likelihoods
        pitch_logits = F.log_softmax(pitch_params, -1)
        pitch_targets = pitches[None,:,1:,None] #1, batch, time-1, 1
        pitch_log_probs = pitch_logits.gather(-1, pitch_targets)[...,0]

        time_targets = times[:,1:]# batch, time-1
        time_result = self.time_dist(time_params, time_targets)
        time_log_probs = time_result.pop('log_prob')

        vel_targets = velocities[:,1:]# batch, time-1
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
                'velocity': None,
                'pitch_params': pitch_params,
                'time_params': time_params
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
        