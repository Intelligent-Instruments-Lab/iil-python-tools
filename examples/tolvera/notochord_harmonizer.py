"""
Notochord MIDI harmonizer server.
Each note from the player produces a harmonizing note from Notochord.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""
import taichi as ti
import numpy as np
import math
import tolvera as tol

from notochord import Notochord
from iipyper import MIDI, run, Timer, cleanup, repeat
from iipyper.state import _lock

def main(
        player_channel=0, # MIDI channel numbered from 0
        noto_channel=1,
        player_inst=20, # General MIDI numbered from 1 (see Notochord.feed docstring)
        noto_inst=20,
        midi_in='IAC Driver Tidal', # MIDI port for player input
        midi_out='IAC Driver Tidal', # MIDI port for Notochord output
        checkpoint="/Users/jack/Documents/git/pro/iil/artifacts/notochord/notochord_lakh_50G_deep.pt"
        ):

    ti.init(arch=ti.vulkan)

    x = 1920
    y = 1080
    n = 88
    boids = tol.vera.BoidsMulti(x, y, n, radius=10, colormode='rgb', species=1, size=1)
    window = ti.ui.Window("Boids", (x, y))
    canvas = window.get_canvas()

    midi = MIDI(midi_in, midi_out)

    if checkpoint is not None:
        noto = Notochord.from_checkpoint(checkpoint)
        noto.eval()
    else:
        noto = None

    note_map = {}
    timer = Timer()

    def note_on(pitch, vel):
        # print('PLAYER:', pitch, vel)
        midi.note_on(note=pitch, velocity=vel, channel=noto_channel)
        # feed in the performed note
        noto.feed(player_inst, pitch, timer.punch(), vel)
        # get the harmonizing note
        r = noto.query(
            next_inst=noto_inst, next_time=0, next_vel=vel,
            include_pitch=range(pitch+1, 128))
        # print('NOTO:', r)
        # send it
        midi.note_on(
            note=r['pitch'], velocity=int(r['vel']), channel=noto_channel)
        # feed back
        noto.feed(r['inst'], r['pitch'], r['time'], r['vel'])
        # prepare for later NoteOff
        note_map[pitch] = r['pitch']

    def note_off(pitch, vel):
        try:
            noto_pitch = note_map.pop(pitch)
            print('NoteOff', pitch, noto_pitch)
        except:
            print('NoteOff not found', pitch)
            return
        midi.note_off(
            note=pitch, velocity=vel, channel=noto_channel)
        # send harmonizing NoteOff
        midi.note_off(
            note=noto_pitch, velocity=vel, channel=noto_channel)
        # feed 
        noto.feed(player_inst, pitch, timer.punch(), 0)
        noto.feed(noto_inst, noto_pitch, 0, 0)

    note = None
    prev_note = None
    boid = 0

    @repeat(0.5)
    def _():
        nonlocal note, prev_note, boid
        boid = int(boid/x*127)
        print(note_map, boid)
        note = boid
        if note is not None:
            note_on(note, 64)
        if prev_note is not None:
            if note != prev_note:
                note_off(prev_note, 0)
        prev_note = note

    @cleanup
    def _():
        """end any remaining notes"""
        for pitch in note_map.values():
            midi.note_off(note=pitch, velocity=0, channel=noto_channel)

    counter = 0
    count = 5

    while window.running:
        with _lock:
            if counter == count:
                counter = 0
                boid = boids.boid(0)
                # boids.set_params(boids_params)
            counter +=1
            canvas.set_image(boids.process())
            window.show()


if __name__=='__main__':
    run(main)
