class Updater:
    '''
    Decouples OSC handling from updating
    Updating is rate-limited by a counter

    e.g.
    ```
    obs_updater = tol.Updater()
    @osc.args("/tolvera/obstacles/set")
    def _(address, *args):
        obs_updater.set(args)
    # ...
    obs_updater(obs.set)
    ```
    '''
    def __init__(self, f, state=None, count=5, update=False):
        self.f = f
        self.count = count
        self.counter = 0
        self.update = update
        self.state = state
    def set(self, state):
        '''
        Set the Updater's state
        '''
        self.state = state
        self.update = True
    def __call__(self):
        '''
        Update the target function with internal state
        '''
        self.counter += 1
        if not (self.update and 
                self.counter>self.count and 
                self.state is not None):
            return
        self.f(self.state)
        self.counter = 0
        self.update = False

class UpdaterOSC(Updater):
    '''
    e.g.:
    def update_pos(s):
        p.field[s[0]].particle.pos = [s[1], s[2]]
    updaters = [
        UpdaterOSC(osc, "/tolvera/obstacles/pos", update_pos),
        # ...
    ]
    # ...
    [u() for u in updaters]
    '''
    def __init__(self, osc, path: str, f, state=None, count=10, update=False):
        super().__init__(f,state,count,update)
        self.osc = osc
        self.path = path
        osc.add_handler(self.path, self.handler)
    def handler(self, address, *args):
        # FIXME: why is args[0] address?
        self.set(args[1:])
