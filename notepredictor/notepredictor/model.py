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
        self.net = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                input_size, heads, hidden_size, norm_first=False
                ), layers)

    def forward(self, ctx, h_ctx, h_tgt):
        """
        Args:
            ctx: list of Tensor[batch x time x input_size], length note_dim-1
                these are the embedded ground truth values
            h_ctx: Tensor[batch x time x input_size]
                (need something to attend to when ctx is empty)
            h_tgt: list of Tensor[batch x time x input_size], length note_dim
                these are projections of the RNN state
        """
        h_tgt = list(h_tgt)
        ctx = list(ctx)
        # h_tgt is 'target' w.r.t TransformerDecoder
        # h_ctx and context are 'memory'
        batch_size = h_ctx.shape[0]*h_ctx.shape[1]
        # fold time into batch, stack modes
        tgt = torch.stack([
            item.reshape(batch_size,-1)
            for item in h_tgt
        ],0)
        mem = torch.stack([
            item.reshape(batch_size,-1)
            for item in [h_ctx, *ctx]
        ],0)
        # now "time"(mode) x "batch"(+time) x channel

        # generate a mask
        # this is both the target and memory mask
        n = len(h_tgt)
        mask = ~tgt.new_ones((n,n), dtype=bool).tril()

        x = self.net(tgt, mem, mask, mask)
        return list(x.reshape(n, *h_ctx.shape).unbind(0))


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

        self.note_dim = 3 # pitch, time, velocity

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
            self.note_dim*emb_size, rnn_hidden, 
            num_layers=rnn_layers, batch_first=True, dropout=dropout)

        # learnable initial RNN state
        self.initial_state = nn.ParameterList([
             # layer x batch x hidden
            nn.Parameter(torch.randn(rnn_layers,1,rnn_hidden)*rnn_hidden**-0.5)
            for _ in range(2 if kind=='lstm' else 1)
        ])

        # projection from RNN state to distribution parameters
        self.h_proj = nn.Linear(rnn_hidden, emb_size*(1+self.note_dim))
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
        
    def get_samplers(self, index_pitch=None, allow_start=False, allow_end=False):
        def sample_pitch(x):
            if not allow_start:
                x[...,self.start_token] = -np.inf
            if not allow_end:
                x[...,self.end_token] = -np.inf
            if index_pitch is not None:
                return x.argsort(-1, True)[...,index_pitch]
            else:
                return D.Categorical(logits=x).sample()

        return (
            sample_pitch, 
            lambda x: self.time_dist.sample(x),
            lambda x: self.vel_dist.sample(x),
        )

    @property
    def embeddings(self):
        return (
            self.pitch_emb,
            self.time_emb,
            self.vel_emb
        )
        
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
        ## broadcast initial state to batch size
        initial_state = tuple(
            t.expand(self.rnn.num_layers, x.shape[0], -1).contiguous() # 1 x batch x hidden
            for t in self.initial_state)
        h, _ = self.rnn(x, initial_state) #batch, time, hidden_size

        # include initial hidden state for predicting first note
        # h = torch.cat((
        #     self.initial_state[0][-1][None].expand(batch_size, 1, -1),
        #     h), -2)

        # fit all note factorizations at once.
        # TODO: perm each batch item independently?
        perm = torch.randperm(self.note_dim)
        hs = list(self.h_proj(h).chunk(self.note_dim+1, -1))
        h_ctx = hs[0]
        h_tgt = [hs[i+1] for i in perm]
        embs = [embs[i][:,1:] for i in perm[:-1]]
        mode_hs = self.xformer(embs, h_ctx, h_tgt)
        mode_hs = [mode_hs[i] for i in perm.argsort()]

        pitch_params, time_params, vel_params = [
            proj(h) for proj,h in zip(self.projections, mode_hs)]

        # get likelihoods
        pitch_logits = F.log_softmax(pitch_params, -1)
        pitch_targets = pitches[:,1:,None] #batch, time, 1
        pitch_log_probs = pitch_logits.gather(-1, pitch_targets)[...,0]

        time_targets = times[:,1:] # batch, time
        time_result = self.time_dist(time_params, time_targets)
        time_log_probs = time_result.pop('log_prob')

        vel_targets = velocities[:,1:] # batch, time
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
    
    # TODO: force
    def predict(self, 
            pitch, time, vel, 
            fix_pitch=None, fix_time=None, fix_vel=None, 
            index_pitch=None, allow_start=False, allow_end=False):
        """
        supply the most recent note and return a prediction for the next note.

        various constraints can be enforced on the next note.

        Args:
            pitch: int. MIDI number of current note.
            time: float. elapsed time since previous note.
            vel: float. (possibly dequantized) MIDI velocity from 0-127 inclusive.
            fix_*: same as above, but to fix a value for the predicted note
            index_pitch: Optional[int]. if not None, determinisitically take the nth
                most likely pitch instead of sampling.
            allow_start: if False, zero probability for sampling the start token
            allow_end: if False, zero probaility for sampling the end token

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

            h = self.h_proj(h).chunk(self.note_dim+1, -1)
            h_ctx = h[0]
            h_tgt = h[1:]

            modalities = list(zip(
                self.projections,
                self.get_samplers(index_pitch, allow_start, allow_end),
                self.embeddings,
                ))

            context = []
            predicted = []
            params = []

            fix = [
                None if item is None else torch.tensor([[item]], dtype=dtype)
                for item, dtype in zip(
                    [fix_pitch, fix_time, fix_vel],
                    [torch.long, torch.float, torch.float])]

            # permute h_tgt, embs, modalities
            # if any modalities are determined, embed them;
            det_idx, undet_idx = [], []
            for i,(item, embed) in enumerate(zip(fix, self.embeddings)):
                if item is None:
                    undet_idx.append(i)
                else:
                    det_idx.append(i)
                    context.append(embed(item))
                    predicted.append(item.item())
                    params.append(None)
            perm = det_idx + undet_idx
            iperm = np.argsort(perm)

            perm_h_tgt = [h_tgt[i] for i in perm]

            # for each undetermined modality, 
            # sample a new value conditioned on alteady determined ones

            # TODO: allow constraints; 
            # attempt to sort the strongest constraints first
            # constraints can be:
            # discrete set, in which case evaluate probs and then sample categorical
            # range, in which case truncate

            while len(undet_idx):
                i = undet_idx.pop(0) # index of modality to determine
                j = len(det_idx) # number already determined
                project, sample, embed = modalities[i]
                # determine value for the next modality
                hidden = self.xformer(context, h_ctx, perm_h_tgt[:j+1])[j]
                params.append(project(hidden))
                pred = sample(params[-1])
                predicted.append(pred.item())
                # prepare for next iteration
                if len(undet_idx):
                    context.append(embed(pred))
                det_idx.append(i)

            return {
                'pitch': predicted[iperm[0]], 
                'time': predicted[iperm[1]],
                'velocity': predicted[iperm[2]],
                'pitch_params': params[iperm[0]],
                'time_params': params[iperm[1]],
                'vel_params': params[iperm[2]]
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
            self.predict(self.start_token, 0., 0.)

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
        