import math
import numpy as np

import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

def reweight_top_p(probs, top_p):
    """given tensor of probabilities, apply top p / "nucleus" filtering"""
    # NOTE: this is fudged slightly, it doesn't 'interpolate' the cutoff bin
    desc_probs, idx = probs.sort(-1, descending=True)
    iidx = idx.argsort(-1)
    cumprob = desc_probs.cumsum(-1)
    # first index where cumprob >= top_p is the last index we don't zero
    to_zero = (cumprob >= top_p).roll(1, -1)
    to_zero[...,0] = False
    # unsort
    to_zero = to_zero.gather(-1, iidx)
    weighted_probs = torch.zeros_like(probs).where(to_zero, probs)
    return weighted_probs / weighted_probs.sum(-1, keepdim=True)
    

class CensoredMixtureLogistic(nn.Module):
    def __init__(self, n, res=1e-2, lo='-inf', hi='inf', 
            sharp_bounds=(1e-4,2e3), init=None):
        super().__init__()
        self.n = n
        self.res = res
        self.sharp_bounds = sharp_bounds
        self.register_buffer('lo', torch.tensor(float(lo)))
        self.register_buffer('hi', torch.tensor(float(hi)))
        # TODO: init is not general-purpose
        #TODO
        if init=='time':
            self.bias = nn.Parameter(torch.cat((
                torch.zeros(n), torch.logspace(-3,1,n), torch.zeros(n)
                )))
        elif init=='velocity':
            self.bias = nn.Parameter(torch.cat((
                torch.zeros(n), torch.linspace(0,127,n), torch.zeros(n)
                )))
        else:
            self.bias = nn.Parameter(torch.cat((
                torch.zeros(n), torch.randn(n), torch.zeros(n)
                )))

    @property
    def n_params(self):
        return self.n*3

    def get_params(self, h):
        assert h.shape[-1] == self.n_params
        h = h+self.bias
        # get parameters from unconstrained hidden state:
        logit_pi, loc, log_s = torch.chunk(h, 3, -1)
        # mixture coefficients
        log_pi = logit_pi - logit_pi.logsumexp(-1,keepdim=True)
        # location
        loc = loc.clamp(self.lo-10*self.res, self.hi+10*self.res)
        # sharpness
        s = F.softplus(log_s).clamp(*self.sharp_bounds)
        return log_pi, loc, s


    def forward(self, h, x):
        """log prob of x under distribution parameterized by h
        Args:
            h: Tensor[...,n_params]
            x: Tensor[...]
            "..." dims must broadcast
        """
        log_pi, loc, s = self.get_params(h)    

        d = self.res/2
        x = x.clamp(self.lo, self.hi)[...,None]
        x_ = (x - loc) * s
        sd = s*d

        # # censoring
        lo_cens = x <= self.lo+d
        hi_cens = x >= self.hi-d
        ones = torch.ones_like(s)
        zeros = torch.zeros_like(s)

        diff_term = torch.where(lo_cens | hi_cens, 
            ones, sd.exp() - (-sd).exp()
            ).log()
        minus_sp_term = torch.where(hi_cens, -sd, F.softplus(-sd-x_))
        plus_sp_term = torch.where(lo_cens, zeros, x_ + F.softplus(sd-x_))

        log_delta_cdf = diff_term - minus_sp_term - plus_sp_term
    
        # log prob
        r = {
            'log_prob': (log_pi + log_delta_cdf).logsumexp(-1)
        }
        # diagnostics
        with torch.no_grad():
            ent = D.Categorical(logits=log_pi).entropy()
            r |= {
                # 'min_sharpness': s.min(),
                'max_sharpness': s.max(),
                'mean_sharpness': (s*log_pi.exp()).sum(-1).mean(),
                # 'min_entropy': ent.min(),
                # 'max_entropy': ent.max(),
                'mean_cmp_entropy': ent.mean(),
                'marginal_cmp_entropy': D.Categorical(
                    log_pi.exp().mean(list(range(log_pi.ndim-1)))).entropy(),
                # 'min_loc': loc.min(),
                # 'max_loc': loc.max()
            }
        return r

    def cdf(self, h, x):
        """
        Args:
            h: Tensor[...,n_params]
            x: Tensor[...]
                `h` should broadcast with `x[...,None]`
        Returns:
            cdf: Tensor[...] (shape of `x` broadcasted with `h[...,0]`)
        """
        log_pi, loc, s = self.get_params(h)  
        cdfs = self.cdf_components(loc, s, x)
        cdf = (cdfs * log_pi.softmax(-1)).sum(-1)
        return cdf

    def cdf_components(self, loc, s, x):
        x_ = (x[...,None] - loc) * s
        return x_.sigmoid()        

    # TODO: 'discrete_sample' method which would re-quantize and then allow
    # e.g. nucleus sampling on the categorical distribution?
    def sample(self, h, truncate=None, shape=None, 
        weight_top_p=None, component_temp=None, bias=None):
        """
        Args:
            h: Tensor[...,n_params]
            truncate: Optional[Tuple[2]]. lower and upper bound for truncation.
            shape: Optional[int]. additional sample shape to be prepended to dims.
            weight_top_p: top_p ("nucleus") filtering for mixture weights.
                default is 1 (no change to distribution). 0 would sample top
                component (after truncation) only.
            component_temp: Optional[float]. sampling temperature of each mixture 
                component. default is 1. 0 would sample component location only,
                ignoring sharpness.
            bias: applied outside of truncation but inside of clamping,
                useful e.g. for latency correction when sampling delta-time
        Returns:
            Tensor[*shape,...] (h without last dimension, prepended with `shape`)
        """
        if shape is None:
            unwrap = True
            shape = 1
        else:
            unwrap = False

        if truncate is None:
            truncate = (-np.inf, np.inf)
        truncate = torch.tensor(truncate).to(self.bias.device)

        if component_temp is None:
            component_temp = 1

        if bias is None:
            bias = 0

        log_pi, loc, s = self.get_params(h)
        s = s/component_temp
        scale = 1/s

        # cdfs: [...,bound,component]
        cdfs = self.cdf_components(loc[...,None,:], s[...,None,:], truncate) 
        # prob. mass of each component witin bounds
        trunc_probs = cdfs[...,1,:] - cdfs[...,0,:] # [...,component]
        probs = log_pi.exp() * trunc_probs # reweighted mixture component probs
        if weight_top_p is not None:
            # reweight with top_p
            probs = reweight_top_p(probs, weight_top_p)

        ## DEBUG
        # print(loc)
        # print(s)
        # print(trunc_probs)
        # print(probs)
        #, log_pi.exp(), trunc_probs)

        c = D.Categorical(probs).sample((shape,))
        # move sample dimension first
        loc = loc.movedim(-1, 0).gather(0, c)
        scale = scale.movedim(-1, 0).gather(0, c)
        upper = cdfs[...,1,:].movedim(-1, 0).gather(0, c)
        lower = cdfs[...,0,:].movedim(-1, 0).gather(0, c)

        u = torch.rand(shape, *h.shape[:-1], device=self.bias.device)
        # truncate
        u = u * (upper-lower) + lower

        # x = loc + scale * (u.log() - (1 - u).log())
        x = loc + bias - scale * (1/u - 1).log()
        x = x.clamp(self.lo, self.hi)
        return x[0] if unwrap else x


class CensoredMixturePointyBoi(nn.Module):
    def __init__(self, n, res=1e-2, lo='-inf', hi='inf', sharp_bounds=(1e-5,2e3)):
        super().__init__()
        self.n = n
        self.res = res
        self.sharp_bounds = sharp_bounds
        # self.register_buffer('max_sharp', torch.tensor(float(max_sharp)))
        self.register_buffer('lo', torch.tensor(float(lo)))
        self.register_buffer('hi', torch.tensor(float(hi)))
        # TODO: init is not general-purpose
        self.bias = nn.Parameter(torch.cat((
            torch.zeros(n), torch.logspace(-3,1,n), torch.zeros(n)
            )))

    @property
    def n_params(self):
        return self.n*3

    def get_params(self, h):
        assert h.shape[-1] == self.n_params
        h = h+self.bias
        # get parameters fron unconstrained hidden state:
        logit_pi, loc, log_s = torch.chunk(h, 3, -1)
        # mixture coefficients
        log_pi = logit_pi - logit_pi.logsumexp(-1,keepdim=True)
        # location
        loc = loc.clamp(self.lo-10*self.res, self.hi+10*self.res)
        # sharpness
        # s = log_s.exp()
        # s = torch.min(F.softplus(log_s), self.max_sharp)
        s = F.softplus(log_s).clamp(*self.sharp_bounds)
        # s = log_s.exp().clamp(*self.sharp_bounds)
        return log_pi, loc, s

    def forward(self, h, x):
        """log prob of x under distribution parameterized by h"""
        log_pi, loc, s = self.get_params(h)    

        x = x.clamp(self.lo, self.hi)[...,None]
        xp, xm = x+self.res/2, x-self.res/2

        # numerical crimes follow

        # censoring
        lo_cens = x <= self.lo
        xm_ = torch.where(lo_cens, -h.new_ones([]), (xm-loc)*s)
        axm_ = torch.where(lo_cens, h.new_zeros([]), xm_.abs())
        hi_cens = x >= self.hi
        xp_ = torch.where(hi_cens, h.new_ones([]), (xp-loc)*s)
        axp_ = torch.where(hi_cens, h.new_zeros([]), xp_.abs())

        log_delta_cdf = (
            (xp_ - xm_ + xp_*axm_ - axp_*xm_).log() 
            - (axp_ + axm_ + axp_*axm_).log1p() 
            - math.log(2))
        
        # log prob
        r = {
            'log_prob': (log_pi + log_delta_cdf).logsumexp(-1)
        }
        # diagnostics
        with torch.no_grad():
            ent = D.Categorical(logits=log_pi).entropy()
            r |= {
                'min_sharpness': s.min(),
                'max_sharpness': s.max(),
                'min_entropy': ent.min(),
                'max_entropy': ent.max(),
                'marginal_entropy': D.Categorical(
                    log_pi.exp().mean(list(range(log_pi.ndim-1)))).entropy(),
                'min_loc': loc.min(),
                'max_loc': loc.max()
            }
        return r

    def cdf(self, h, x):
        log_pi, loc, s = self.get_params(h)  
        x_ = (x[...,None] - loc) * s 
        cdfs = x_ / (1+x_.abs()) * 0.5 + 0.5
        cdf = (cdfs * log_pi.softmax(-1)).sum(-1)
        return cdf


    def sample(self, h, shape=1):
        """
        Args:
            shape: additional sample shape to be prepended to dims
        """
        # if shape is None: shape = []

        log_pi, loc, s = self.get_params(h)
        c = D.Categorical(logits=log_pi).sample((shape,))
        # move sample dimension first
        loc = loc.movedim(-1, 0).gather(0, c)
        s = s.movedim(-1, 0).gather(0, c)

        u = torch.rand(shape, *h.shape[:-1])*2-1
        x_ = u / (1 - u.abs())
        x = x_ / s + loc

        return x.clamp(self.lo, self.hi)
