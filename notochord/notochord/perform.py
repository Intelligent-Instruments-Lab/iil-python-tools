class MIDIConfig(dict):
    @property
    def channels(self):
        return set(self)
    @property
    def insts(self):
        return set(self.values())
    def inv(self, inst):
        for chan,inst_ in self.items():
            if inst_==inst:
                return chan
        raise KeyError