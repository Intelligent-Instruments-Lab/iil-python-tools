lenia_pause(0)
lenia_reset(0)

@swim
def pat(d=0.5, i=0):
    lenia.set_brush(P("draw, erase", i), 0.1, P('0.1,0.5,0.9',i), 0.5)
    a(pat, d=0.5, i=i+1)

hush(pat)
