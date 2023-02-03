"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from notochord import Notochord
from iipyper import MIDI, run, Timer
# import numpy as np
# from time import time_ns

def main(
        player_channel=0, 
        noto_channel=1,
        player_inst=20,
        noto_inst=20,
        checkpoint="artifacts/notochord-latest.ckpt"):
    midi = MIDI()

    if checkpoint is not None:
        noto = Notochord.from_checkpoint(checkpoint)
        noto.eval()
    else:
        noto = None

    note_map = {}
    timer = Timer()

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

        # NoteOn
        if msg.type=='note_on' and vel > 0:
            # feed in the performed note
            noto.feed(player_inst, pitch, timer.punch(), vel)
            # get the harmonizing note
            r = noto.query(
                next_inst=noto_inst, next_time=0, next_vel=vel,
                include_pitch=range(pitch+1, 128))
            print('NOTO:', r)
            # send it
            midi.send('note_on', 
                note=r['pitch'], velocity=int(r['vel']), channel=noto_channel)
            # feed back
            noto.feed(r['inst'], r['pitch'], r['time'], r['vel'])
            # prepare for later NoteOff
            note_map[pitch] = r['pitch']
        # NoteOff
        else:
            try:
                noto_pitch = note_map.pop(pitch)
            except:
                print('harmonizing NoteOff not found')
                return
            # send harmonizing NoteOff
            midi.send('note_off',
                note=noto_pitch, velocity=vel, channel=noto_channel)
            # feed 
            noto.feed(player_inst, pitch, timer.punch(), 0)
            noto.feed(noto_inst, noto_pitch, 0, 0)

if __name__=='__main__':
    run(main)
