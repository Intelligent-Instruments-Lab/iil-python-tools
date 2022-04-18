import math

import numpy as np

import torch
from torch import logit, nn
import torch.nn.functional as F
import torch.distributions as D

from .rnn import GenericRNN
from .distributions import CensoredMixtureLogistic, reweight_top_p

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

# class ModalityTransformer(nn.Module):
#     """
#     Model joint distribution of note modalities (e.g. pitch, time, velocity).

#     This is an autoregressive Transformer model for the *internal* structure of notes.
#     It is *not* autoregressive in time, but in modality.
#     At training time, it executes in parallel over all timesteps and modalities, with
#     time dependencies provided via the RNN backbone.

#     At sampling time it is called serially, one modality at a time, 
#     repeatedly at each time step.

#     Inspired by XLNet: http://arxiv.org/abs/1906.08237
#     """
#     def __init__(self, input_size, hidden_size, heads=4, layers=1):
#         super().__init__()
#         self.net = nn.TransformerDecoder(
#             nn.TransformerDecoderLayer(
#                 input_size, heads, hidden_size, norm_first=False
#                 ), layers)

#     def forward(self, ctx, h_ctx, h_tgt):
#         """
#         Args:
#             ctx: list of Tensor[batch x time x input_size], length note_dim-1
#                 these are the embedded ground truth values
#             h_ctx: Tensor[batch x time x input_size]
#                 projection of RNN state (need something to attend to when ctx is empty)
#             h_tgt: list of Tensor[batch x time x input_size], length note_dim
#                 these are projections of the RNN state for each target,
#                 which the Transformer will map to distribution parameters.
#         """
#         # explicitly broadcast
#         h_ctx, *ctx = torch.broadcast_tensors(h_ctx, *ctx)
#         h_ctx, *h_tgt = torch.broadcast_tensors(h_ctx, *h_tgt)

#         # h_tgt is 'target' w.r.t TransformerDecoder
#         # h_ctx and context are 'memory'
#         batch_size = h_ctx.shape[0]*h_ctx.shape[1]
#         # fold time into batch, stack modes
#         tgt = torch.stack([
#             item.reshape(batch_size,-1)
#             for item in h_tgt
#         ],0)
#         mem = torch.stack([
#             item.reshape(batch_size,-1)
#             for item in [h_ctx, *ctx]
#         ],0)
#         # now "time"(mode) x "batch"(+time) x channel

#         # generate a mask
#         # this is both the target and memory mask
#         # masking is such that each target can only depend on "previous" context
#         n = len(h_tgt)
#         mask = ~tgt.new_ones((n,n), dtype=bool).tril()

#         x = self.net(tgt, mem, mask, mask)
#         return list(x.reshape(n, *h_ctx.shape).unbind(0))


class NotePredictor(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, 
            emb_size=256, 
            rnn_hidden=2048, rnn_layers=1, kind='gru', 
            mlp_layers=0,
            dropout=0.1, 
            num_pitches=128, 
            num_instruments=272,
            time_bounds=(0,10), time_components=32, time_res=1e-2,
            vel_components=16
            ):
        """
        """
        super().__init__()

        self.note_dim = 4 # instrument, pitch, time, velocity

        self.instrument_start_token = 0
        self.instrument_domain = num_instruments+1

        self.pitch_start_token = num_pitches
        self.pitch_domain = num_pitches+1

        self.time_dist = CensoredMixtureLogistic(
            time_components, time_res, 
            lo=time_bounds[0], hi=time_bounds[1], init='time')
        self.vel_dist = CensoredMixtureLogistic(
            vel_components, 1.0, lo=0, hi=127, init='velocity')
        
        # embeddings for inputs
        self.instrument_emb = nn.Embedding(self.instrument_domain, emb_size)
        self.pitch_emb = nn.Embedding(self.pitch_domain, emb_size)
        self.time_emb = SineEmbedding(emb_size)
        self.vel_emb = MixEmbedding(emb_size, (0, 127))

        # RNN backbone
        self.rnn = GenericRNN(kind, 
            emb_size, rnn_hidden, 
            num_layers=rnn_layers, batch_first=True, dropout=dropout)

        # learnable initial RNN state
        self.initial_state = nn.ParameterList([
             # layer x batch x hidden
            nn.Parameter(torch.randn(rnn_layers,1,rnn_hidden)*rnn_hidden**-0.5)
            for _ in range(2 if kind=='lstm' else 1)
        ])

        # projection from RNN state to distribution parameters
        self.h_proj = nn.Linear(rnn_hidden, emb_size)
        # self.projections = nn.ModuleList([
        #     nn.Linear(emb_size, self.instrument_domain),
        #     nn.Linear(emb_size, self.pitch_domain),
        #     nn.Linear(emb_size, self.time_dist.n_params, bias=False),
        #     nn.Linear(emb_size, self.vel_dist.n_params, bias=False)
        # ])
        self.projections = nn.ModuleList([
            SelfGatedMLP(
                emb_size, emb_size, self.instrument_domain, mlp_layers, dropout),
            SelfGatedMLP(
                emb_size, emb_size, self.pitch_domain, mlp_layers, dropout),
            SelfGatedMLP(
                emb_size, emb_size, self.time_dist.n_params, mlp_layers, dropout),
            SelfGatedMLP(
                emb_size, emb_size, self.vel_dist.n_params, mlp_layers, dropout),
        ])

        self.end_proj = nn.Linear(rnn_hidden, 2)

        with torch.no_grad():
            for p in self.projections:
                p.net[-1].weight.mul_(1e-2)
            self.end_proj.weight.mul(1e-2)

        # persistent RNN state for inference
        for n,t in zip(self.cell_state_names(), self.initial_state):
            self.register_buffer(n, t.clone())
        self.step = 0

    def cell_state_names(self):
        return tuple(f'cell_state_{i}' for i in range(len(self.initial_state)))

    @property
    def cell_state(self):
        return tuple(getattr(self, n) for n in self.cell_state_names())

    @property
    def embeddings(self):
        return (
            self.instrument_emb,
            self.pitch_emb,
            self.time_emb,
            self.vel_emb
        )
        
    def forward(self, instruments, pitches, times, velocities, ends, validation=False):
        """
        teacher-forced probabilistic loss and diagnostics for training.

        Args:
            instruments: LongTensor[batch, time]
            pitches: LongTensor[batch, time]
            times: FloatTensor[batch, time]
            velocities: FloatTensor[batch, time]
            ends: LongTensor[batch, time]
        """
        batch_size, batch_len = pitches.shape

        # embed data to input vectors
        inst_emb = self.instrument_emb(instruments) # batch, time, emb_size
        pitch_emb = self.pitch_emb(pitches) # batch, time, emb_size
        time_emb = self.time_emb(times) # batch, time, emb_size
        vel_emb = self.vel_emb(velocities) # batch, time, emb_size

        embs = (inst_emb, pitch_emb, time_emb, vel_emb)

        # feed to RNN backbone
        x = sum(embs)
        ## broadcast initial state to batch size
        initial_state = tuple(
            t.expand(self.rnn.num_layers, x.shape[0], -1).contiguous() # 1 x batch x hidden
            for t in self.initial_state)
        h, _ = self.rnn(x, initial_state) #batch, time, hidden_size

        # fit all note factorizations 
        # e.g. inst->pitch->time->vel vs vel->time->inst->pitch
        trim_h = h[:,:-1]
        # always include hidden state, never include same modality,
        # other dependencies are random per time and position
        n = self.note_dim
        tr = torch.randint(2, (*trim_h.shape[:2],n,n), dtype=torch.bool, device=h.device)
        tr &= ~torch.eye(n,n, dtype=torch.bool, device=h.device)
        ar_mask = torch.cat((tr.new_ones(*trim_h.shape[:2],1,n), tr), -2).float()

        to_mask = torch.stack((
            self.h_proj(trim_h),
            *(emb[:,1:] for emb in embs)
        ), -1)
        mode_hs = (to_mask @ ar_mask).tanh().unbind(-1)
        
        # final projections to raw distribution parameters
        inst_params, pitch_params, time_params, vel_params = [
            proj(h) for proj,h in zip(self.projections, mode_hs)]

        # get likelihood of data for each modality
        inst_logits = F.log_softmax(inst_params, -1)
        inst_targets = instruments[:,1:,None] #batch, time, 1
        inst_log_probs = inst_logits.gather(-1, inst_targets)[...,0]

        pitch_logits = F.log_softmax(pitch_params, -1)
        pitch_targets = pitches[:,1:,None] #batch, time, 1
        pitch_log_probs = pitch_logits.gather(-1, pitch_targets)[...,0]

        time_targets = times[:,1:] # batch, time
        time_result = self.time_dist(time_params, time_targets)
        time_log_probs = time_result.pop('log_prob')

        vel_targets = velocities[:,1:] # batch, time
        vel_result = self.vel_dist(vel_params, vel_targets)
        vel_log_probs = vel_result.pop('log_prob')

        # end prediction
        # skip the first position for convenience 
        # (so masking is the same for end as for note parts)
        end_params = self.end_proj(h[:,1:])
        end_logits = F.log_softmax(end_params, -1)
        end_log_probs = end_logits.gather(-1, ends[:,1:,None])[...,0]

        r = {
            'end_log_probs': end_log_probs,
            'instrument_log_probs': inst_log_probs,
            'pitch_log_probs': pitch_log_probs,
            'time_log_probs': time_log_probs,
            'velocity_log_probs': vel_log_probs,
            **{'time_'+k:v for k,v in time_result.items()},
            **{'velocity_'+k:v for k,v in vel_result.items()}
        }
        # this just computes some extra diagnostics which are inconvenient to do in the
        # training script. should be turned off during training for performance.
        if validation:
            with torch.no_grad():
                r['time_acc_30ms'] = (
                    self.time_dist.cdf(time_params, time_targets + 0.03)
                    - torch.where(time_targets - 0.03 >= 0,
                        self.time_dist.cdf(time_params, time_targets - 0.03),
                        time_targets.new_zeros([]))
                )
        return r

    # TODO: remove allow_end here
    # allow_start should just be False
    def get_samplers(self, 
            instrument_top_p=None, exclude_instrument=None,
            pitch_topk=None, index_pitch=None, allow_start=False, allow_end=False, 
            pitch_top_p=None,
            sweep_time=False, min_time=None, max_time=None, bias_time=None, time_weight_top_p=None, time_component_temp=None,
            min_vel=None, max_vel=None):
        """
        this method converts the many arguments to `predict` into functions for
        sampling each note modality (e.g. pitch, time, velocity)
        """

        def sample_instrument(x):
            if not allow_start:
                x[...,self.instrument_start_token] = -np.inf
            if exclude_instrument is not None:
                x[...,exclude_instrument] = -np.inf
            probs = x.softmax(-1)
            if instrument_top_p is not None:
                probs = reweight_top_p(probs, instrument_top_p)
            return D.Categorical(probs).sample()

        def sample_pitch(x):
            if not allow_start:
                x[...,self.pitch_start_token] = -np.inf
            if index_pitch is not None:
                return x.argsort(-1, True)[...,index_pitch]
            elif pitch_topk is not None:
                return x.argsort(-1, True)[...,:pitch_topk].transpose(0,-1)
            else:
                probs = x.softmax(-1)
                if pitch_top_p is not None:
                    probs = reweight_top_p(probs, pitch_top_p)
                return D.Categorical(probs).sample()

        def sample_time(x):
            # TODO: respect trunc_time when sweep_time is True
            if sweep_time:
                if min_time is not None or max_time is not None:
                    raise NotImplementedError("""
                    trunc_time with sweep_time needs implementation
                    """)
                assert x.shape[0]==1, "batch size should be 1 here"
                log_pi, loc, s = self.time_dist.get_params(x)
                idx = log_pi.squeeze().argsort()[:9]
                loc = loc.squeeze()[idx].sort().values[...,None] # multiple times in batch dim
                # print(loc.shape)
                return loc
            else:
                trunc = (
                    -np.inf if min_time is None else min_time,
                    np.inf if max_time is None else max_time)
                return self.time_dist.sample(x, 
                    truncate=trunc, bias=bias_time,
                    component_temp=time_component_temp, weight_top_p=time_weight_top_p)

        def sample_velocity(x):
            trunc = (
                -np.inf if min_vel is None else min_vel,
                np.inf if max_vel is None else max_vel)
            return self.vel_dist.sample(x, truncate=trunc)

        return (
            sample_instrument,
            sample_pitch, 
            sample_time,
            sample_velocity,
        )
    
    # TODO: remove pitch_topk and sweep_time?
    def predict(self, 
            inst, pitch, time, vel, 
            fix_instrument=None, fix_pitch=None, fix_time=None, fix_vel=None, 
            pitch_topk=None, index_pitch=None, allow_start=False, allow_end=False,
            sweep_time=False, min_time=None, max_time=None, bias_time=None, 
            exclude_instrument=None,
            instrument_temp=None, pitch_temp=None, rhythm_temp=None, timing_temp=None,
            min_vel=None, max_vel=None):
        """
        consume the most recent note and return a prediction for the next note.

        various constraints can be enforced on the next note.

        Args:
            pitch: int. MIDI number of current note.
            time: float. elapsed time in seconds since previous note.
            vel: float. (possibly dequantized) MIDI velocity from 0-127 inclusive.
            fix_*: same as above, but to fix a value for the predicted note.
                sampled values will always condition on fixed values, so passing
                `fix_time=0`, for example, will make a probabilistically-sound
                prediction of a chord tone: "what is the next note given that it 
                happens immediately after the last one?"
            pitch_topk: Optional[int]. if not None, instead of sampling pitch, stack
                the top k most likely pitches along the batch dimension
            index_pitch: Optional[int]. if not None, deterministically take the nth
                most likely pitch instead of sampling.
            allow_start: if False, zero probability for sampling the start token
            allow_end: if False, zero probaility for sampling the end token
            sweep_time: if True, instead of sampling time, choose a diverse set of
                times and stack along the batch dimension
            min_time, max_time: if not None, truncate the time distribution
            bias_time: add this delay to the time 
                (after applying min/max but before clamping to 0).
                may be useful for latency correction.
            exclude_instrument: instrument id to exclude from sampling.
            instrument_temp: if not None, apply top_p sampling to instrument. 0 is
                deterministic, 1 is 'natural' according to the model
            pitch_temp: if not None, apply top_p sampling to pitch. 0 is
                deterministic, 1 is 'natural' according to the model
            rhythm_temp: if not None, apply top_p sampling to the weighting
                of mixture components. this affects coarse rhythmic patterns; 0 is
                deterministic, 1 is 'natural' according to the model
            timing_temp: if not None, apply temperature sampling to the time
                component. this affects fine timing; 0 is deterministic and precise,
                1 is 'natural' according to the model.
            min_vel, max_vel: if not None, truncate the velocity distribution

        Returns: dict of
            'pitch': int. predicted MIDI number of next note.
            'time': float. predicted time to next note.
            'velocity': float. unquantized predicted velocity of next note.
            '*_params': tensor. distrubution parameters for visualization purposes.
        """
        if (index_pitch is not None) and (pitch_temp is not None):
            print("warning: `index pitch` overrides `pitch_temp`")

        with torch.inference_mode():
            inst = torch.LongTensor([[inst]]) # 1x1 (batch, time)
            pitch = torch.LongTensor([[pitch]]) # 1x1 (batch, time)
            time = torch.FloatTensor([[time]]) # 1x1 (batch, time)
            vel = torch.FloatTensor([[vel]]) # 1x1 (batch, time)

            embs = [
                self.instrument_emb(inst),
                self.pitch_emb(pitch), # 1, 1, emb_size
                self.time_emb(time),# 1, 1, emb_size
                self.vel_emb(vel)# 1, 1, emb_size
            ]
            x = sum(embs)
            
            h, new_state = self.rnn(x, self.cell_state)
            for t,new_t in zip(self.cell_state, new_state):
                t[:] = new_t

            # h_parts = self.h_proj(h).chunk(self.note_dim+1, -1)
            # h_ctx = h_parts[0]
            # h_tgt = h_parts[1:]

            modalities = list(zip(
                self.projections,
                self.get_samplers(
                    instrument_temp, exclude_instrument,
                    pitch_topk, index_pitch, allow_start, allow_end, 
                    pitch_temp,
                    sweep_time, min_time, max_time, bias_time, 
                    rhythm_temp, timing_temp,
                    min_vel, max_vel),
                self.embeddings,
                ))

            context = [self.h_proj(h)] # embedded outputs for autoregressive prediction
            predicted = [] # raw outputs
            params = [] # distribution parameters for visualization

            fix = [
                None if item is None else torch.tensor([[item]], dtype=dtype)
                for item, dtype in zip(
                    [fix_instrument, fix_pitch, fix_time, fix_vel],
                    [torch.long, torch.long, torch.float, torch.float])]

            # if any modalities are determined, embed them
            # sort constrained modalities before unconstrained
            # TODO: option to skip modalities
            det_idx, cons_idx, uncons_idx = [], [], []
            for i,(item, embed) in enumerate(zip(fix, self.embeddings)):
                if item is None:
                    if (
                        i==0 and any(p is not None for p in (
                            instrument_temp, exclude_instrument)) or
                        i==1 and (pitch_topk or pitch_temp is not None) or
                        i==2 and any(p is not None for p in (
                            min_time, max_time, rhythm_temp, timing_temp)) or
                        i==3 and any(p is not None for p in (
                            min_vel, max_vel))
                        ):
                        cons_idx.append(i)
                    else:
                        uncons_idx.append(i)
                else:
                    det_idx.append(i)
                    context.append(embed(item))
                    predicted.append(item)
                    params.append(None)
            undet_idx = cons_idx + uncons_idx
            perm = det_idx + undet_idx # permutation from the canonical order
            iperm = np.argsort(perm) # inverse permutation back to canonical order

            md = ['instrument', 'pitch', 'time', 'vel']
            print('sampling order:', [md[i] for i in perm])

            # for each undetermined modality, 
            # sample a new value conditioned on alteady determined ones

            # TODO: allow constraints; 
            # attempt to sort the strongest constraints first
            # constraints can be:
            # discrete set, in which case evaluate probs and then sample categorical;
            # range, in which case truncate;
            # temperature?
            
            running_ctx = sum(context)
            # print(running_ctx)
            # perm_h_tgt = [h_tgt[i] for i in perm]
            while len(undet_idx):
                i = undet_idx.pop(0) # index of modality to determine
                # j = len(det_idx) # number already determined
                project, sample, embed = modalities[i]
                # determine value for the next modality
                hidden = running_ctx.tanh() #self.xformer(context, h_ctx, perm_h_tgt[:j+1])[j]
                params.append(project(hidden))
                pred = sample(params[-1])
                predicted.append(pred)
                # prepare for next iteration
                if len(undet_idx):
                    # context.append(embed(pred))
                    running_ctx += embed(pred)
                det_idx.append(i)

            pred_inst = predicted[iperm[0]]
            pred_pitch = predicted[iperm[1]]
            pred_time = predicted[iperm[2]]
            pred_vel = predicted[iperm[3]]

            if allow_end:
                end_params = self.end_proj(h)
                # print(end_params)
                end = D.Categorical(logits=end_params).sample()
            else:
                end = torch.zeros(h.shape[:-1])

            if sweep_time or pitch_topk:
                # return lists of predictions
                pred_inst = [x.item() for x in pred_inst]
                pred_pitch = [x.item() for x in pred_pitch]
                pred_time = [x.item() for x in pred_time]
                pred_vel = [x.item() for x in pred_vel]
                end = [x.item() for x in end]
                # print(pred_time, pred_pitch, pred_vel)
            else:
                # return single predictions
                pred_inst = pred_inst.item()
                pred_pitch = pred_pitch.item()
                pred_time = pred_time.item()
                pred_vel = pred_vel.item()
                end = end.item()

            self.step += 1
            return {
                'end': end,
                'step': self.step,
                'instrument': pred_inst,
                'pitch': pred_pitch, 
                'time': pred_time,
                'velocity': pred_vel,
                'inst_params': params[iperm[0]],
                'pitch_params': params[iperm[1]],
                'time_params': params[iperm[2]],
                'vel_params': params[iperm[3]]
            }
    
    def reset(self, start=True):
        """
        resets internal model state.
        Args:
            start: if True, send a start token through the model with dt=0
                   but discard the prediction
        """
        self.step = 0
        for n,t in zip(self.cell_state_names(), self.initial_state):
            getattr(self, n)[:] = t.detach()
        if start:
            self.predict(self.instrument_start_token, self.pitch_start_token, 0., 0.)

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
        