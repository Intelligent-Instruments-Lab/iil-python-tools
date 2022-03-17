import math

import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

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
                torch.zeros(n), torch.linspace(1,126,n), torch.zeros(n)
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
        """log prob of x under distribution parameterized by h"""
        log_pi, loc, s = self.get_params(h)    

        d = self.res/2
        x = x.clamp(self.lo, self.hi)[...,None]
        x_ = (x - loc) * s
        sd = s*d

        # # censoring
        lo_cens = x <= self.lo+d
        hi_cens = x >= self.hi-d
        ones = torch.ones_like(x_)
        zeros = torch.zeros_like(x_)

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
        log_pi, loc, s = self.get_params(h)  
        x_ = (x[...,None] - loc) * s
        cdfs = x_.sigmoid()
        cdf = (cdfs * log_pi.softmax(-1)).sum(-1)
        return cdf

    def sample(self, h, shape=1):
        """
        Args:
            shape: additional sample shape to be prepended to dims
        """
        log_pi, loc, s = self.get_params(h)
        scale = 1/s

        c = D.Categorical(logits=log_pi).sample((shape,))
        # move sample dimension first
        loc = loc.movedim(-1, 0).gather(0, c)
        scale = scale.movedim(-1, 0).gather(0, c)

        u = torch.rand(shape, *h.shape[:-1])

        x = loc + scale * (u.log() - (1 - u).log())
        return x.clamp(self.lo, self.hi)


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
