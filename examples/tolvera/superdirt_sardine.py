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
