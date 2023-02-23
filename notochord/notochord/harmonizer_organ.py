"""
Notochord MIDI harmonizer server.
Each note from the player produces a harmonizing note from Notochord.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""
import time

from notochord import Notochord
from iipyper import MIDI, run, Stopwatch, cleanup

def main(
        player_channel=0, # MIDI channel numbered from 0
        # noto_channel=1,
        player_inst=20, # General MIDI numbered from 1 (see Notochord.feed docstring)
        noto_inst=None,
        noto_config=None,
        midi_in=None, # MIDI port for player input
        midi_out=None, # MIDI port for Notochord output
        thru=False, # resend player in on output
        checkpoint="artifacts/notochord-latest.ckpt", # Notochord checkpoint
        note_off_delay=2e-3,

        below=False, # harmonize above
        above=True, # harmonize below
        n=1, # number of tones
        ):
    midi = MIDI(midi_in, midi_out)

    if noto_config is None:
        if noto_inst is None:
            noto_inst = player_inst
        noto_config = [ # channel (from 0), inst, min transpose, max transpose (inclusive)
            # (0,noto_inst,-36,-12),
            (1,noto_inst,-36,-12),
            (2,noto_inst,-12,0), 
            (3,noto_inst,3,4), 
            (4,noto_inst,7,36), 
            ]
    for (_,_,lo,hi) in noto_config:
        assert lo <= hi
    ######

    if checkpoint is not None:
        noto = Notochord.from_checkpoint(checkpoint)
        noto.eval()
    else:
        noto = None

    note_map = {} # map from player pitch to notochord pitches
    stopwatch = Stopwatch()

    # @midi.handle(type='program_change')
    # def _(msg):
    #     """
    #     Program change events set instruments
    #     """
    #     nonlocal player_inst, noto_inst
    #     if msg.channel == player_channel:
    #         player_inst = msg.program
    #     if msg.channel == noto_channel:
    #         noto_inst = msg.program

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

        if thru:
            midi.send(msg)

        # NoteOn
        if msg.type=='note_on' and vel > 0:
                # feed in the performed note
            noto.feed(player_inst, pitch, stopwatch.punch(), vel)
            # player_pitches = set(note_map) | {pitch}
            chosen_pitches = set()
            chosen_triples = []
            # get the harmonizing notes
            # for _ in range(n):
            for noto_channel, noto_inst, min_x, max_x in noto_config:

                print(max(36,pitch+min_x), min(94, pitch+max_x+1))

                pitches = (
                    set(range(max(36,pitch+min_x), min(94, pitch+max_x+1)))
                    - {pitch} - chosen_pitches
                )

                if len(pitches)==0:
                    print(f'skipping {noto_channel=}, no pitches available')
                    print(
                        set(range(max(36,pitch+min_x), min(94, pitch+max_x+1))),
                        'minus', {pitch}, 'minus', chosen_pitches)
                    continue

                # pitches = set((
                #     *(range(max(player_pitches)+1, 128) if above else []),
                #     *(range(0, min(player_pitches)) if below else [])
                # )) - chosen_pitches # don't repeat pitches

                print(f'{pitches=}')

                h = noto.query(
                    next_inst=noto_inst, next_time=0, next_vel=vel,
                    include_pitch=pitches)
                print('NOTO:', h)

                h_inst = h['inst'] # noto_inst
                h_pitch = h['pitch']
                h_time = h['time'] # 0
                h_vel = h['vel'] # 0

                # # support legato/polyphonic playing:
                # # check existing harmonizing notes for duplicates with the
                # # new harmonizing note.
                # # if duplicate, just transfer it in the note map 
                new_note = True
                for _,hs in note_map.items():
                    for tr in list(hs):
                        n_chan,n_inst,n_pitch = tr
                        # if noto_inst==n_inst and n_pitch==h_pitch:
                        if n_chan==noto_channel and n_pitch==h_pitch:
                    # if n_inst==noto_inst and h_pitch in hs:
                            hs.remove(tr)
                            new_note = False
                # the model *shouldn't* pick duplicates,
                # but the thinking is that if it wants to,
                # it's better to have fewer notes
                # than to force it to pick something else

                if new_note:
                    # send it
                    midi.note_on(
                        note=h_pitch, velocity=int(h_vel), channel=noto_channel)
                    # feed back
                    noto.feed(h_inst, h_pitch, h_time, h_vel)
                # prepare for later NoteOff
                chosen_pitches.add(h_pitch)
                chosen_triples.append((noto_channel, h_inst, h_pitch))
            # note_map[pitch] = chosen_pitches
            note_map[pitch] = chosen_triples
        # NoteOff
        else:
            try:
                # noto_pitches = note_map.pop(pitch)
                noto_triples = note_map.pop(pitch)
            except:
                print('harmonizing NoteOffs not found')
                return

            # send harmonizing NoteOff
            for noto_channel, noto_inst, noto_pitch in noto_triples:
                midi.note_off(
                    note=noto_pitch, velocity=vel, channel=noto_channel)
                time.sleep(note_off_delay)
            # feed
            for noto_channel, noto_inst, noto_pitch in noto_triples:
                noto.feed(noto_inst, noto_pitch, 0, 0)
            noto.feed(player_inst, pitch, stopwatch.punch(), 0)

    @cleanup
    def _():
        """end any remaining notes"""
        for triples in note_map.values():
            for channel, _, pitch in triples:
                midi.note_off(note=pitch, velocity=0, channel=channel)

if __name__=='__main__':
    run(main)
