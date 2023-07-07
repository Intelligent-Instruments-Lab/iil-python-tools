from iipyper import repeat

def headless(r, rate=1/60):
    '''
    Repeat the render function 
    '''
    @repeat(rate)
    def _():
        r()
