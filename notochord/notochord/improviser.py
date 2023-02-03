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
        player_inst=18,
        noto_inst=20,
        max_note_len=5,
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

    # map (inst, pitch) pairs to Timers
    # delete keys when there is a note off
    notes = {}

    def query():
        # force prediction of noto_inst
        # the first time
        insts = [noto_inst]
        if 2*noto_events() > player_events():
            insts.append(player_inst)

        pending.event = noto.query(
            min_time=timer.read(),
            include_inst=insts)
        print('prediction:', pending.event)

    @midi.handle(type='program_change')
    def _(msg):
        """Program change events set instruments"""
        nonlocal player_inst, noto_inst
        if msg.channel == player_channel:
            player_inst = msg.program
        if msg.channel == noto_channel:
            noto_inst = msg.program

    @midi.handle(type='control_change', control=0, channel=player_channel)
    def _(msg):
        """any CC0 message on player channel resets Notochord"""
        noto.reset()

    @midi.handle(type=('note_on', 'note_off'), channel=player_channel)
    def _(msg):
        """MIDI NoteOn events from the player"""
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

    def noto_event():
        # notochord event happens:
        event = pending.event
        inst, pitch, vel = event['inst'], event['pitch'], event['vel']

        # track held notes
        k = (inst, pitch)
        if vel > 0:
            notes[k] = Timer(punch=True)
        else:
            if k in notes:
                del notes[k]
            else:
                # bad prediction: note-off without note-on
                print('REPLACING BAD PREDICTION', pending.event)
                query()
                return

        # send as MIDI
        midi.send('note_on', note=pitch, velocity=int(vel), channel=noto_channel)
        # feed back to Notochord
        noto.feed(inst, pitch, timer.punch(), vel)
        # track total events played
        noto_events.plus()
        # print
        print('NOTOCHORD:', event)

    @repeat(5e-4, lock=True)
    def _():
        """Loop, checking if predicted next event happens"""
        # check if current prediction has passed
        if pending.event is not None and timer.read() > pending.event['time']:
        # if so, check if it is a notochord-controlled instrument
            if pending.event['inst'] == noto_inst:
                # prediction happens -- send and feed
                noto_event()
            # query for new prediction
            query()

    @repeat(1e-2, lock=True)
    def _():
        """Loop, checking if there are any stuck notes"""
        for (inst,pitch),t in notes.items():
            if t.read() > max_note_len:
                pending.event = {'inst':inst, 'pitch':pitch, 'vel':0, 'time':0}
                print('END STUCK NOTE')

if __name__=='__main__':
    run(main)
