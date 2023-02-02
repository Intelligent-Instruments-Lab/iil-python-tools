"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from notochord import Notochord
from iipyper import MIDI, run, Timer, repeat
# import numpy as np
import time
# from time import time_ns

def main(
        player_channel=0, 
        noto_channel=1,
        player_inst=257,
        noto_inst=258,
        checkpoint="artifacts/notochord-latest.ckpt"):
    midi = MIDI()

    if checkpoint is not None:
        noto = Notochord.from_checkpoint(checkpoint)
        noto.eval()
    else:
        noto = None

    timer = Timer()

    class Counter():
        def __init__(self):
            self.count = 0
        def plus(self, n=1):
            self.count += n
        def __call__(self):
            return self.count

    class Prediction:
        def __init__(self, event):
            self.event = event
    pending = Prediction(None)

    noto_events = Counter()
    player_events = Counter()

    def query():
        # force prediction of noto_inst
        # the first time
        insts = [noto_inst]
        if noto_events() > 2*player_events():
            insts.append(player_inst)

        pending.event = noto.query(
            min_time=timer.read(),
            include_inst=insts)
        print('prediction:', pending.event)

    @midi.handle(type='program_change')
    def _(msg):
        """
        Program change events set instruments
        """
        nonlocal player_inst, noto_inst
        if msg.channel == player_channel:
            player_inst = msg.program
        if msg.channel == noto_channel:
            noto_inst = msg.program

    @midi.handle(type='control_change', control=0, channel=player_channel)
    def _(msg):
        """
        any CC0 message on player channel resets Notochord
        """
        noto.reset()

    @midi.handle(type=('note_on', 'note_off'), channel=player_channel)
    def _(msg):
        """
        MIDI NoteOn events from the player
        """
        if noto is None:
            print('Notochord model not loaded')
            return

        print('PLAYER:', msg)

        pitch = msg.note
        vel = msg.velocity

        # feed event to Notochord
        # query for new prediction
        noto.feed(player_inst, pitch, timer.punch(), vel)
        query()
        player_events.plus()
        # print(timer.read())

    @repeat(1e-3, lock=False)
    def _():
        """Loop, checking if predicted next event happens"""
        # check if current prediction has passed
        # if so, feed and send MIDI
        # otherwise, query for a new prediction
        # if pending.event is not None:
            # print(time.perf_counter_ns(), timer.t, timer.read(), pending.event['time'])
        if pending.event is not None and timer.read() > pending.event['time']:
        # if so, check if it is a notochord-controlled instrument
            if pending.event['inst'] == noto_inst:
                # prediction happens -- send and feed
                midi.send('note_on', 
                    note=pending.event['pitch'], 
                    velocity=int(pending.event['vel']), 
                    channel=noto_channel)
                noto.feed(
                    pending.event['inst'], 
                    pending.event['pitch'], 
                    timer.punch(),
                    pending.event['vel'])
                noto_events.plus()
                print('NOTOCHORD:', pending.event)

            # query for new prediction
            query()

if __name__=='__main__':
    run(main)
