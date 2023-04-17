import math
from typing import List, Tuple, Dict, Union, Any
from numbers import Number
from collections import namedtuple

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

from .rnn import GenericRNN
from .distributions import CensoredMixtureLogistic, categorical_sample

from .util import arg_to_set

class Query:
    def __init__(self, modality, then=None, **kw):
        self.modality = modality
        self.then = then
        self.kw = kw
    def __repr__(self):
        return f"{self.modality} \n{self.then}"# {self.kw}"

Range = namedtuple('Range', ('lo', 'hi', 'weight'), defaults=(-np.inf, np.inf, 1))
Subset = namedtuple('Subset', ('values', 'weight'), defaults=(None, 1))

class SineEmbedding(nn.Module):
    def __init__(self, n, hidden, w0=1e-3, w1=10, scale='log'):
        """
        Args:
            n (int): number of sinusoids
            hidden (int): embedding size
            w0 (float): minimum wavelength
            w1 (float): maximum wavelength
            scale (str): if 'log', more wavelengths close to w0
        """
        super().__init__()
        if scale=='log':
            w0 = np.log(w0)
            w1 = np.log(w1)
        ws = torch.linspace(w0, w1, n)
        if scale=='log':
            ws = ws.exp()
        self.register_buffer('fs', 2 * math.pi / ws)
        self.proj = nn.Linear(n,hidden)

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

class GLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a, b = x.chunk(2, -1)
        return a * b.sigmoid()

class GLUMLP(nn.Module):
    def __init__(self, input, hidden, output, layers, dropout=0, norm=None):
        super().__init__()
        h = input
        def get_dropout():
            if dropout > 0:
                return (nn.Dropout(dropout),)
            return tuple()
        def get_norm():
            if norm=='layer':
                return (nn.LayerNorm(hidden),)
            return tuple()
        self.net = []
        for _ in range(layers):
            self.net.append(nn.Sequential(
                *get_dropout(), nn.Linear(h, hidden*2), GLU(), *get_norm()))
            h = hidden
        self.net.append(nn.Linear(hidden, output))
        self.net = nn.Sequential(*self.net)

        with torch.no_grad():
            self.net[-1].weight.mul_(1e-2)

    def forward(self, x):
        return self.net(x)


class Notochord(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, 
            emb_size=256, 
            rnn_hidden=2048, rnn_layers=1, kind='gru', 
            mlp_layers=0,
            dropout=0.1, norm=None,
            num_pitches=128, 
            num_instruments=320,
            time_sines=128, vel_sines=128,
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
            sharp_bounds=(1e-4,2e3),
            lo=time_bounds[0], hi=time_bounds[1], init='time')
        self.vel_dist = CensoredMixtureLogistic(
            vel_components, 1.0,
            sharp_bounds=(1e-3,128),
            lo=0, hi=127, init='velocity')
        
        # embeddings for inputs
        self.instrument_emb = nn.Embedding(self.instrument_domain, emb_size)
        self.pitch_emb = nn.Embedding(self.pitch_domain, emb_size)
        self.time_emb = (#torch.jit.script(
            SineEmbedding(
            time_sines, emb_size, 1e-3, 30, scale='log'))
        # self.vel_emb = MixEmbedding(emb_size, (0, 127))
        self.vel_emb = (#torch.jit.script(
            SineEmbedding(
            vel_sines, emb_size, 2, 512, scale='lin'))

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

        mlp_cls = GLUMLP#lambda *a: torch.jit.script(GLUMLP(*a))
        # projection from RNN state to distribution parameters
        self.h_proj = mlp_cls(
                rnn_hidden, emb_size, emb_size, 
                mlp_layers, dropout, norm)
        self.projections = nn.ModuleList([
            mlp_cls(
                emb_size, emb_size, self.instrument_domain, 
                mlp_layers, dropout, norm),
            mlp_cls(
                emb_size, emb_size, self.pitch_domain, 
                mlp_layers, dropout, norm),
            mlp_cls(
                emb_size, emb_size, self.time_dist.n_params,
                mlp_layers, dropout, norm),
            mlp_cls(
                emb_size, emb_size, self.vel_dist.n_params, 
                mlp_layers, dropout, norm),
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

        # volatile hidden states for caching purposes
        self.h = None
        self.h_query = None      

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
        
    def forward(self, instruments, pitches, times, velocities, ends,
            validation=False, ar_mask=None):
        """
        teacher-forced probabilistic loss and diagnostics for training.

        Args:
            instruments: LongTensor[batch, time]
            pitches: LongTensor[batch, time]
            times: FloatTensor[batch, time]
            velocities: FloatTensor[batch, time]
            ends: LongTensor[batch, time]
            validation: bool (computes some extra diagnostics)
            ar_mask: Optional[Tensor[note_dim x note_dim]] if None, generate random
                masks for training
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
        if ar_mask is None:
            # random binary mask
            ar_mask = torch.randint(2, (*trim_h.shape[:2],n,n), dtype=torch.bool, device=h.device)
            # zero diagonal
            ar_mask &= ~torch.eye(n,n, dtype=torch.bool, device=h.device)
        # include hidden state
        ar_mask = torch.cat((ar_mask.new_ones(*ar_mask.shape[:-2],1,n), ar_mask), -2).float()

        to_mask = torch.stack((
            self.h_proj(trim_h),
            *(emb[:,1:] for emb in embs)
        ), -1)
        # TODO: try without this tanh?
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


    def is_drum(self, inst):
        # TODO: add a constructor argument to specify which are drums
        # hardcoded for now
        return inst > 128 and inst < 257 or inst > 288

    
    def feed(self, inst, pitch, time, vel, **kw):
        """consume an event and advance hidden state
        
        Args:
            inst: int. instrument of current note.
                0 is start token
                1-128 are General MIDI instruments
                129-256 are drumkits (MIDI 1-128 on channel 13)
                257-288 are 'anonymous' melodic instruments
                289-320 are 'anonymous' drumkits
            pitch: int. MIDI pitch of current note.
                0-127 are MIDI pitches / drums
                128 is start token
            time: float. elapsed time in seconds since previous event.
            vel: float. (possibly dequantized) MIDI velocity from 0-127 inclusive.
                0 indicates a note-off event
            **kw: ignored (allows doing e.g. noto.feed(**noto.query(...)))
        """
        with torch.inference_mode():
            inst = torch.LongTensor([[inst]]) # 1x1 (batch, time)
            pitch = torch.LongTensor([[pitch]]) # 1x1 (batch, time)
            time = torch.FloatTensor([[time]]) # 1x1 (batch, time)
            vel = torch.FloatTensor([[vel]]) # 1x1 (batch, time)

            embs = [
                self.instrument_emb(inst), # 1, 1, emb_size
                self.pitch_emb(pitch), # 1, 1, emb_size
                self.time_emb(time),# 1, 1, emb_size
                self.vel_emb(vel)# 1, 1, emb_size
            ]
            x = sum(embs)

            self.h, new_state = self.rnn(x, self.cell_state)
            for t,new_t in zip(self.cell_state, new_state):
                t[:] = new_t

            self.h_query = None

    # TODO: add end prediction to deep_query
    def deep_query(self, query, predict_end=True):
        """flexible querying with nested Query objects.
        see query_vtip for an example.

        Args:
            query: Query object
        """
        if self.h_query is None:
            with torch.inference_mode():
                self.h_query = self.h_proj(self.h)
        event = self._deep_query(
            query, hidden=self.h_query[:,0], event={})
        
        if predict_end:
            # print('END')
            # print(f'{self.h}')
            with torch.inference_mode():
                end_params = self.end_proj(self.h)
                event['end'] = end_params.softmax(-1)[...,1].item()
                # event['end'] = D.Categorical(logits=end_params).sample().item()
        else:
            event['end'] = 0#torch.zeros(self.h.shape[:-1])

        return event
    
    def _deep_query(self, query, hidden, event):
        m = query.modality
        sq = query.then
        try:
            idx = ('inst','pitch','time','vel').index(m)
        except ValueError:
            raise ValueError(f'unknown modality "{m}"')
        
        project = self.projections[idx]
        embed = self.embeddings[idx]

        if m=='time':
            dist = self.time_dist
            sample = dist.sample
        elif m=='vel':
            dist = self.vel_dist
            sample = dist.sample
        else:
            sample = categorical_sample

        # Query.then can be:

        # None (no further sub-queries)
        # or str (no further sub-queries, set the 'path' property of the event)

        # Query describing what to do next

        # pair of Number, Query
        # setting a deterministic value for the current query,
        # and the query to execute next

        # list of Range, Query pairs
        # or list of Subset, Query pairs
        #    describing what to do next given where the sampled value falls
        #    Range and Subset can also carry weights


        with torch.inference_mode():
            #
            if sq is None or isinstance(sq, str) or isinstance(sq, Query):
                # this is the final query (sq is None or a path tag)
                # or there are no cases, just proceed to next subquery
                action = sq
                params = project(hidden.tanh())
                result = sample(params, **query.kw)

            elif len(sq)==0:
                raise ValueError(f"subquery has no cases: {sq}")
            
            # deterministic single value cases
            # elif len(sq)==1 and len(sq[0])==3 and isinstance(sq[0][0], Number):
            #     # singleton list case
            #     key, action = sq[0]
            #     result = key
            elif len(sq)==2 and isinstance(sq[0], Number):
                # bare pair case
                key, action = sq
                result = key

            elif m in ('inst', 'pitch'):
                # weighted subsets case
                try:
                    assert all(isinstance(s, Subset) for s,_ in sq)
                except Exception:
                    raise ValueError(f"""
                    subqueries should all be Subset, Query tuples here: {sq}
                    """)

                params = project(hidden.tanh())

                 # sample subset
                if len(sq) > 1:
                    all_probs = params.softmax(-1)
                    probs = [
                        all_probs[...,s.values].sum(-1) * s.weight
                        for s,_ in sq]
                    # print(f'deep_query {m} {sq=} {probs=}')
                    idx = categorical_sample(torch.tensor(probs).log())
                else:
                    idx = 0
                s,action = sq[idx]
                # sample from range
                result = sample(params, whitelist=s.values, **query.kw)

            elif m in ('time', 'vel'):
                # weighted ranges case
                assert all(isinstance(r, Range) for r,_ in sq)

                params = project(hidden.tanh())
                # sample range
                if len(sq) > 1:
                    probs = [
                        (dist.cdf(params, r.hi) - dist.cdf(params, r.lo)
                        ) * r.weight
                        for r,_ in sq]
                    # print(f'deep_query {m} {probs=}')
                    idx = categorical_sample(torch.tensor(probs).log())
                else:
                    idx = 0
                r,action = sq[idx]
                # sample from range
                result = sample(params, truncate=(r.lo, r.hi), **query.kw)

            try:
                event[m] = result.item()
            except Exception:
                event[m] = result

            # embed, add to hidden, recurse into subquery
            if isinstance(action, Query):
                emb = embed(result)
                hidden = hidden + emb
                return self._deep_query(action, hidden, event)
            else:
                event['path'] = action
                return event
            
    def query_tipv_onsets(self,
        min_time=None, max_time=None, 
        include_inst=None,
        include_pitch=None,
        truncate_quantile_time=None,
        truncate_quantile_pitch=None,
        rhythm_temp=None, timing_temp=None,
        min_vel=None, max_vel=None
        ):
        """
        for onset-only_models
        """
        q = Query(
            'time',
            truncate=(min_time or -np.inf, max_time or np.inf), 
            truncate_quantile=truncate_quantile_time,
            weight_top_p=rhythm_temp, component_temp=timing_temp,
            then=Query(
                'inst',
                whitelist=include_inst,
                then=Query(
                    'pitch',
                    whitelist=include_pitch,
                    truncate_quantile=truncate_quantile_pitch,
                    then=Query(
                        'vel',
                        truncate=(min_vel or 0.5, max_vel or np.inf),
                    )
                )
            )
        )
        return self.deep_query(q)
    
    def query_itpv_onsets(self,
        min_time=None, max_time=None, 
        include_inst=None,
        include_pitch=None,
        truncate_quantile_time=None,
        truncate_quantile_pitch=None,
        rhythm_temp=None, timing_temp=None,
        min_vel=None, max_vel=None
        ):
        """
        for onset-only_models
        """
        q = Query(
            'inst',
            whitelist=include_inst,
            then=Query(
                'time',
                truncate=(min_time or -np.inf, max_time or np.inf), 
                truncate_quantile=truncate_quantile_time,
                weight_top_p=rhythm_temp, component_temp=timing_temp,
                then=Query(
                    'pitch',
                    whitelist=include_pitch,
                    truncate_quantile=truncate_quantile_pitch,
                    then=Query(
                        'vel',
                        truncate=(min_vel or 0.5, max_vel or np.inf),
                    )
                )
            )
        )
        return self.deep_query(q)


    def query_vtip(self,
            note_on_map:Dict[int,List[int]], 
            note_off_map:Dict[int,List[int]], 
            min_time=None, max_time=None, 
            truncate_quantile_time=None,
            truncate_quantile_pitch=None,
            steer_density=None, # truncate_quantile_type ? 
            ):
        """ Query in velocity-time-instrument-pitch order,
            efficiently truncating the joint distribution to just allowable
            (velocity>0)/instrument/pitch triples.

            Args:
                note_on_map: possible note-ons as {instrument: [pitch]} 
                note_off_map: possible note-offs as {instrument: [pitch]} 
        """
 
        no_on = all(len(ps)==0 for ps in note_on_map.values())
        no_off = all(len(ps)==0 for ps in note_off_map.values())
        if no_on and no_off:
            raise ValueError(f"""
                no possible notes {note_on_map=} {note_off_map=}""")
        
        def get_subquery(ipm, path):
            return Query(
                'time',
                Query(
                    'inst', [(
                        Subset([i]), 
                        Query('pitch', path, 
                            whitelist=list(ps), 
                            truncate_quantile=truncate_quantile_pitch)
                    ) for i,ps in ipm.items() if len(ps)]
                ),
                truncate=(min_time or -np.inf, max_time or np.inf), truncate_quantile=truncate_quantile_time
            )
        w = 1 if steer_density is None else 2**(steer_density*2-1)
        
        w_on = 0 if no_on else w
        w_off = 0 if no_off else 1/w
        
        return self.deep_query(Query('vel', [
            (Range(-np.inf,0.5,w_off), get_subquery(note_off_map, 'note off')),
            (Range(0.5,np.inf,w_on), get_subquery(note_on_map, 'note on'))
        ]))
    
    def query_vipt(self,
        note_on_map, note_off_map,
        min_time=None, max_time=None, 
        truncate_quantile_time=None,
        truncate_quantile_pitch=None,
        steer_density=None,
        ):
        """
        Query in velocity-instrument-pitch-time order,
            efficiently truncating the joint distribution to just allowable
            (velocity>0)/instrument/pitch triples.

            Args:
                note_on_map: possible note-ons as {instrument: [pitch]} 
                note_off_map: possible note-offs as {instrument: [pitch]} 
        """

        no_on = all(len(ps)==0 for ps in note_on_map.values())
        no_off = all(len(ps)==0 for ps in note_off_map.values())
        if no_on and no_off:
            raise ValueError(f"""
                no possible notes {note_on_map=} {note_off_map=}""")
        
        def get_subquery(note_map, path):
            return Query(
                'inst', 
                then=[(Subset([i]), Query(
                    'pitch', 
                    whitelist=list(ps), 
                    truncate_quantile=truncate_quantile_pitch,
                    then=Query(
                        'time', path,         
                        truncate=(min_time or -np.inf, max_time or np.inf), truncate_quantile=truncate_quantile_time
                    )
                )) for i,ps in note_map.items() if len(ps)]
            )
        
        w = 1 if steer_density is None else 2**(steer_density*2-1)
        
        w_on = 0 if no_on else w
        w_off = 0 if no_off else 1/w
        
        return self.deep_query(Query('vel', [
            (Range(-np.inf,0.5,w_off), get_subquery(note_off_map, 'note off')),
            (Range(0.5,np.inf,w_on), get_subquery(note_on_map, 'note on'))
        ]))
    
    # def query_ipvt(self,
    #     note_map, 
    #     min_time=-np.inf, max_time=np.inf, 
    #     min_vel=-np.inf, max_vel=np.inf,
    #     truncate_quantile_time=None,
    #     truncate_quantile_pitch=None,
    #     ):
    #     """
    #     """
      
    #     return self.deep_query(Query(
    #         'inst', then=[(
    #             Subset([i]), Query(
    #                 'pitch', 
    #                 whitelist=list(ps), 
    #                 truncate_quantile=truncate_quantile_pitch,
    #                 then=Query(
    #                     'vel',
    #                     truncate=(min_vel or -np.inf, max_vel or np.inf),
    #                     then=Query(
    #                         'time',         
    #                         truncate=(min_time or -np.inf, max_time or np.inf), truncate_quantile=truncate_quantile_time
    #                     )
    #                 )
    #             )
    #         ) for i,ps in note_map.items() if len(ps)]
    #     ))

    # TODO: remove pitch_topk and sweep_time?
    # TODO: rewrite this to build queries and dispatch to deep_query
    def query(self,
            next_inst=None, next_pitch=None, next_time=None, next_vel=None,
            pitch_topk=None, index_pitch=None,
            allow_end=False,
            sweep_time=False, 
            min_time=None, max_time=None,
            truncate_quantile_time=None,
            include_inst=None, exclude_inst=None,
            include_pitch=None, exclude_pitch=None,
            truncate_quantile_pitch=None,
            allow_anon=True, include_drum=None,
            instrument_temp=None, pitch_temp=None, velocity_temp=None,
            rhythm_temp=None, timing_temp=None,
            min_vel=None, max_vel=None,
            handle=None, return_params=False):
        """
        return a prediction for the next note.

        various constraints on the the next note can be requested.

        Args:
            # hard constraints
            next_*: same as the arguments to feed, but to fix a value for the predicted note.
                sampled values will always condition on fixed values, so passing
                `next_inst=1`, for example, will make the event appropriate
                for the piano (instrument 1) to play.
                
            # partial constraints
            allow_end: if False, zero probability of sampling the end marker
            min_time, max_time: if not None, truncate the time distribution
            include_inst: instrument id(s) to include in sampling.
                (if not None, all others will be excluded)
            exclude_inst: instrument id(s) to exclude from sampling.
            include_pitch: pitch(es) to include in sampling.
                (if not None, all others will be excluded)
            exclude_pitch: pitch(es) to exclude from sampling.
            allow_anon: bool. if False, zero probability of anon instruments
            include_drum: like `include_pitch`, but only in effect when 
                instrument is a drumkit
            min_vel, max_vel: if not None, truncate the velocity distribution

            # sampling strategies
            truncate_quantile_pitch: applied after include_pitch, exclude_pitch
            truncate_quantile_time: applied after min_time, max_time
            instrument_temp: if not None, apply top_p sampling to instrument. 0 is
                deterministic, 1 is 'natural' according to the model
            pitch_temp: if not None, apply top_p sampling to pitch. 0 is
                deterministic, 1 is 'natural' according to the model
            velocity_temp: if not None, apply temperature sampling to the velocity
                component.
            rhythm_temp: if not None, apply top_p sampling to the weighting
                of mixture components. this affects coarse rhythmic patterns;
                0 is deterministic, 1 is 'natural' according to the model
            timing_temp: if not None, apply temperature sampling to the time
                component. this affects fine timing; 0 is deterministic and 
                precise, 1 is 'natural' according to the model.
            index_pitch: Optional[int]. if not None, deterministically take the
                nth most likely pitch instead of sampling.

            # multiple predictions
            pitch_topk: Optional[int]. if not None, instead of sampling pitch, 
                stack the top k most likely pitches along the batch dimension
            sweep_time: if True, instead of sampling time, choose a diverse set
                of times and stack along the batch dimension

            # other
            handle: metadata to be included in the returned dict, if not None
            return_params: if True, return tensors of distribution parameters
                under the keys `inst_params`, `pitch_params`, `time_params`,
                and `vel_params`.

        Returns: dict of
            'end': int. value of 1 indicates the *current* event (the one 
                passed as arguments to `predict`) was the last event, and the
                predicted event should *not* be played. if `allow end` is false, 
                this will always be 0.
            'step': int. number of steps since calling `reset`.
            'inst': int. id of predicted instrument.
                1-128 are General MIDI standard melodic instruments
                129-256 are drumkits for MIDI programs 1-128
                257-288 are 'anonymous' melodic instruments
                289-320 are 'anonymous' drumkits
            'pitch': int. predicted MIDI number of next note, 0-128.
            'time': float. predicted time to next note in seconds.
            'vel': float. unquantized predicted velocity of next note.
                0-127; hard 0 indicates a note-off event.
            '*_params': tensor. distribution parameters for visualization
                purposes.

            note: `instrument`, `pitch`, `time`, `velocity` may return lists,
                when using `sweep_time` or `pitch_topk`. that part of the API 
                is very experimental and likely to break.
        """
         # validate options:
        if (index_pitch is not None) and (pitch_temp is not None):
            print("warning: `index pitch` overrides `pitch_temp`")

        inst_intervention = any(p is not None for p in (
            instrument_temp, include_inst, exclude_inst))

        pitch_intervention = (pitch_topk or any(p is not None for p in (
            pitch_temp, include_pitch, exclude_pitch, include_drum)))

        time_intervention = any(p is not None for p in (
            min_time, max_time, rhythm_temp, timing_temp))

        vel_intervention = any(p is not None for p in (
            min_vel, max_vel, velocity_temp))

        exclude_inst = arg_to_set(exclude_inst)
        if not allow_anon:
            exclude_inst |= set(range(257, 321))
        constrain_inst = list((
            set(range(self.instrument_domain)) - {self.instrument_start_token}
            if include_inst is None 
            else arg_to_set(include_inst)
        ) - exclude_inst)
        if len(constrain_inst)==0:
            raise ValueError("""
            every instrument has been excluded. check values of 
            `include_inst` and `exclude_inst`
            """)
        # elif len(constrain_inst)==1:
        #     print("""
        #     warning: recommended to use `next_inst`, not 
        #     `include_inst` to allow only one specific instrument
        #     """)
        
        constrain_pitch = list((
            set(range(self.pitch_domain)) - {self.pitch_start_token}
            if include_pitch is None 
            else arg_to_set(include_pitch)
        ) - arg_to_set(exclude_pitch))
        if len(constrain_pitch)==0:
            raise ValueError("""
            every pitch has been excluded. check values of 
            `include_pitch` and `exclude_pitch`
            """)
        elif len(constrain_pitch)==1:
            print("""
            warning: recommended to use `next_pitch`, not 
            `include_pitch` to allow only one specific pitch
            """)

        # TODO: this got really complicated to support include_drum...
        # really want to edit the whole joint distribution of pitch,inst in 
        # cases where certain pitches or drums need to be excluded...
        # would that be practical? if there are ~40000 inst x pitch combos?
        # would need to run the instrument head for a whole batch of all
        # allowable pitches or vice-versa...
        def sample_instrument(x):
            # if include_drum is supplied, make sure to exclude drum instruments
            # when no pitch is in the allowed drums
            if include_drum is not None:
                pit = predicted_by_name('pitch')
                pits = [pit] if pit is not None else constrain_pitch
                if pits is not None and all(pit not in include_drum for pit in pits):
                    nonlocal constrain_inst
                    if constrain_inst is None:
                        constrain_inst = range(1,self.instrument_domain)
                    constrain_inst = [
                        i for i in constrain_inst if not self.is_drum(i)]

            # if constrain_inst is not None:
            #     preserve_x = x[...,constrain_inst]
            #     x = torch.full_like(x, -np.inf)
            #     x[...,constrain_inst] = preserve_x
            # probs = x.softmax(-1)
            # if instrument_temp is not None:
            #     probs = reweight_top_p(probs, instrument_temp)
            # return D.Categorical(probs).sample()

            return categorical_sample(x, 
                whitelist=constrain_inst,
                top_p=instrument_temp)

        def sample_pitch(x):
            # conditional constraint
            if include_drum is not None:
                # if this event is / must be a drum,
                # use include_drum instead of constrain_inst
                inst = predicted_by_name('instrument')
                insts = [inst] if inst is not None else constrain_inst
                if insts is not None and all(self.is_drum(i) for i in insts):
                    nonlocal constrain_pitch
                    constrain_pitch = include_drum

            if pitch_topk is not None:
                raise NotImplementedError

            return categorical_sample(x,
                whitelist=constrain_pitch, 
                index=index_pitch,
                top_p=pitch_temp,
                truncate_quantile=truncate_quantile_pitch
                )
            # if constrain_pitch is not None:
            #     preserve_x = x[...,constrain_pitch]
            #     x = torch.full_like(x, -np.inf)
            #     x[...,constrain_pitch] = preserve_x
            # # x is modified logits

            # if index_pitch is not None:
            #     return x.argsort(-1, True)[...,index_pitch]
            # elif pitch_topk is not None:
            #     return x.argsort(-1, True)[...,:pitch_topk].transpose(0,-1)
            
            # probs = x.softmax(-1)
            # if pitch_temp is not None:
            #     probs = reweight_top_p(probs, pitch_temp)

            # if steer_pitch is not None:
            #     return steer_categorical(probs, steer_pitch)
            # else:
            #     return D.Categorical(probs).sample()

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
                loc = loc.squeeze()[idx].sort().values[...,None] 
                # multiple times in batch dim
                # print(loc.shape)
                return loc
            
            trunc = (
                -np.inf if min_time is None else min_time,
                np.inf if max_time is None else max_time)

            return self.time_dist.sample(x, 
                truncate=trunc,
                component_temp=timing_temp, 
                weight_top_p=rhythm_temp,
                truncate_quantile=truncate_quantile_time
                )

        def sample_velocity(x):
            trunc = (
                -np.inf if min_vel is None else min_vel,
                np.inf if max_vel is None else max_vel)
            return self.vel_dist.sample(
                x, component_temp=velocity_temp, truncate=trunc,
                # truncate_quantile=truncate_quantile_vel
                )

        with torch.inference_mode():
            if self.h_query is None:
                self.h_query = self.h_proj(self.h)

            modalities = list(zip(
                self.projections,
                (sample_instrument, sample_pitch, sample_time, sample_velocity),
                self.embeddings,
                ))

            context = [self.h_query] # embedded outputs for autoregressive prediction
            predicted = [] # raw outputs
            params = [] # distribution parameters for visualization

            fix = [
                None if item is None else torch.tensor([[item]], dtype=dtype)
                for item, dtype in zip(
                    [next_inst, next_pitch, next_time, next_vel],
                    [torch.long, torch.long, torch.float, torch.float])]

            # if any modalities are determined, embed them
            # sort constrained modalities before unconstrained
            # TODO: option to skip modalities
            det_idx, cons_idx, uncons_idx = [], [], []
            for i,(item, embed) in enumerate(zip(fix, self.embeddings)):
                if item is None:
                    if (
                        i==0 and inst_intervention or
                        i==1 and pitch_intervention or
                        i==2 and time_intervention or
                        i==3 and vel_intervention):
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

            mode_names = ['instrument', 'pitch', 'time', 'velocity']
            name_to_idx = {k:v for k,v in zip(mode_names, iperm)}
            def predicted_by_name(name):
                idx = name_to_idx[name]
                if len(predicted) > idx:
                    return predicted[idx]
                return None
            # print('sampling order:', [mode_names[i] for i in perm])

            # for each undetermined modality, 
            # sample a new value conditioned on already determined ones
            
            running_ctx = sum(context)
            # print(running_ctx)
            # perm_h_tgt = [h_tgt[i] for i in perm]
            while len(undet_idx):
                # print(running_ctx.norm())
                i = undet_idx.pop(0) # index of modality to determine
                # j = len(det_idx) # number already determined
                project, sample, embed = modalities[i]
                # determine value for the next modality
                hidden = running_ctx.tanh()
                params.append(project(hidden))
                pred = sample(params[-1])
                predicted.append(pred)
                # prepare for next iteration
                if len(undet_idx):
                    # context.append(embed(pred))
                    running_ctx += embed(pred)
                det_idx.append(i)

            pred_inst = predicted_by_name('instrument')
            pred_pitch = predicted_by_name('pitch')
            pred_time = predicted_by_name('time')
            pred_vel = predicted_by_name('velocity')

            if allow_end:
                end_params = self.end_proj(self.h)
                # print(end_params)
                end = D.Categorical(logits=end_params).sample()
            else:
                end = torch.zeros(self.h.shape[:-1])

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
            r = {
                'end': end,
                'step': self.step,
                'inst': pred_inst,
                'pitch': pred_pitch, 
                'time': pred_time,
                'vel': pred_vel,
            }

            if handle is not None:
                r['handle'] = handle

            if return_params:
                r |= {
                    'inst_params': params[iperm[0]],
                    'pitch_params': params[iperm[1]],
                    'time_params': params[iperm[2]],
                    'vel_params': params[iperm[3]]
                }

            return r


    def predict(self, inst, pitch, time, vel, **kw):
        """
        DEPRECATED: alias for feed_query
        """
        self.feed(inst, pitch, time, vel)
        return self.query(**kw)

    def feed_query(self, inst, pitch, time, vel, **kw):
        """
        call self.feed with *args, then self.query with **kwargs.
        """
        self.feed(inst, pitch, time, vel)
        return self.query(**kw)

    def query_feed(self, *a, **kw):
        """
        call self.query with *args **kwargs, then self.feed with result,
            and return result
        """
        r = self.query(*a, **kw)
        self.feed(r['inst'], r['pitch'], r['time'], r['vel'])
        return r

    def feed_query_feed(self, inst, pitch, time, vel, **kw):
        """
        call self.feed with *args, then self.query with **kwargs.
        """
        self.feed(inst, pitch, time, vel)
        return self.query_feed(**kw)
    
    def reset(self, start=True):
        """
        resets internal model state.
        Args:
            start: if True, send start tokens through the model
        """
        self.step = 0
        for n,t in zip(self.cell_state_names(), self.initial_state):
            getattr(self, n)[:] = t.detach()
        if start:
            self.feed(
                self.instrument_start_token, self.pitch_start_token, 0., 0.)

    @classmethod
    def from_checkpoint(cls, path):
        """
        create a Notochord from a checkpoint file containing 
        hyperparameters and model weights.
        """
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = cls(**checkpoint['kw']['model'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model
        