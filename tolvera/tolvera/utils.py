from iipyper import _lock

def headless(r):
    '''
    Run a render function in a loop, with a lock, forever.
    '''
    while True:
        with _lock:
            r()

