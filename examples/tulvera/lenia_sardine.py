lenia_pause(0)
lenia_reset(0)

@swim
def pat(d=0.5, i=0):
    lenia.set_brush(P("draw, erase", i), 0.1, P('0.1,0.5,0.9',i), 0.5)
    a(pat, d=0.5, i=i+1)

hush(pat)



##### LL

from sardine import *


llosc = OSC(ip="127.0.0.1", port=6789,
    name="LivingLooper",
    ahead_amount=0.25)

def ll(p, v, i=0):
    O(llosc, 'll/test/'+p, value=v).out(i)

O(llosc, 'll/test/'+p, value=v).out(i)

@swim
def ll1(d=1, i=0, j=0):
    ll('mix', 0, i)
    ll('gain', 0, i)
    ll('loop', 0, i)
    ll('pitch/ratio', 1, i)
    ll('pitch/disperse', 0.001, i)
    ll('pitch/time', 0.001, i)
    a(ll1, i=i+1, j=j*i)

hush(ll1)

ll('mix', 0)
ll('loop',0)

@swim
def sd1(d=1, i=0):
    S('bd').out(i)
    a(sd1, d=1/16, i=i+1)

hush()

@swim
def sd1(d=0.06, i=0):
    S('bd', speed='[1:20, 1]', legato=1, gain=1).out(i)
    a(sd1, d=d, i=i+1)

c.link()
c._superdirt_nudge = 1
c.bpm = 120
@swim 
def normal(d=0.5, i=0):
    S('bd', trig=euclid(4,8)).out(i, div=1)
    S('cp', trig=euclid(2,8)).out(i, div=32)
    S('sn', trig=euclid(8,8)).out(i, div=32)
    a(normal, d=1/32, i=i+1)

hush()

c.bpm = 120
@swim 
def normal(d=0.5, i=0):
    S('bd', trig=euclid(4,8)).out(i, div=32)
    S('cp', trig=euclid(2,8)).out(i, div=32)
    S('sn', trig=euclid(8,8)).out(i, div=32)
    print(P('$.p', i))
    a(normal, d=1/32, i=i+1)

##############

c.bpm = 120

@swim 
def jarm_debug(d=0.5, i=0):
    S('bd, .').out(i)
    # S('<hh, drum:r*8>').out(i)
    # S('amencutup:r*8, ..').out(i)
    # S('cp!8',speed='[0.1:1,0.1]*2', amp='[0.1:0.2,0.05]').out(i)
    a(jarm_debug, d=1/12, i=i+1)

hush(jarm_debug)

c.bpm = 600
@swim
def fast_func(d=0.5,i=0):
    S('amencutup:7!80, drum:r*8!80').out(i)
    a(fast_func, d=1/8, i=i+1)

@swim
def fast_func2(d=0.5,i=0):
    S('amencutup:6!80, trump:3!80', speed='[1:12],[1:20],[1:40]').out(i)
    a(fast_func2, d=1/8, i=i+1)

@swim
def fast_func(d=0.5,i=0):
    S('bd').out(i)
    a(fast_func, d=1/16, i=i+1)

hush()
