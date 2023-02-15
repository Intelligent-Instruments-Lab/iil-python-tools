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
