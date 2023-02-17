"""
Notochord MIDI harmonizer server.
Each note from the player produces a harmonizing note from Notochord.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

from notochord import Notochord
from iipyper import MIDI, run, Stopwatch, cleanup

def main(
        player_channel=0, # MIDI channel numbered from 0
        noto_channel=1,
        player_inst=20, # General MIDI numbered from 1 (see Notochord.feed docstring)
        noto_inst=20,
        midi_in=None, # MIDI port for player input
        midi_out=None, # MIDI port for Notochord output
        below=False, # harmonize above
        above=True, # harmonize below
        n=1, # number of tones
        checkpoint="artifacts/notochord-latest.ckpt" # Notochord checkpoint
        ):
    midi = MIDI(midi_in, midi_out)

    print(f'{above=}, {below=}')

    if checkpoint is not None:
        noto = Notochord.from_checkpoint(checkpoint)
        noto.eval()
    else:
        noto = None

    note_map = {} # map from player pitch to notochord pitches
    stopwatch = Stopwatch()

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
            noto.feed(player_inst, pitch, stopwatch.punch(), vel)
            player_pitches = set(note_map) | {pitch}
            chosen_pitches = set()
            # get the harmonizing notes
            for _ in range(n):
                pitches = set((
                    *(range(max(player_pitches)+1, 128) if above else []),
                    *(range(0, min(player_pitches)) if below else [])
                )) - chosen_pitches # don't repeat pitches

                print(pitches)

                h = noto.query(
                    next_inst=noto_inst, next_time=0, next_vel=vel,
                    include_pitch=pitches)
                print('NOTO:', h)

                h_pitch = h['pitch']
                h_vel = h['vel']
                h_time = h['time']
                h_inst = h['inst']

                # # support legato/polyphonic playing:
                # # check existing harmonizing notes for duplicates with the
                # # new harmonizing note.
                # # if duplicate, just transfer it in the note map 
                transfer = False
                for p,hs in note_map.items():
                    if h_pitch in hs:
                        hs.remove(h_pitch)
                        transfer=True

                if not transfer:
                    # send it
                    midi.note_on(
                        note=h_pitch, velocity=int(h_vel), channel=noto_channel)
                    # feed back
                    noto.feed(h_inst, h_pitch, h_time, h_vel)
                # prepare for later NoteOff
                chosen_pitches.add(h_pitch)
            note_map[pitch] = chosen_pitches
        # NoteOff
        else:
            try:
                noto_pitches = note_map.pop(pitch)
            except:
                print('harmonizing NoteOffs not found')
                return

            # send harmonizing NoteOff
            for noto_pitch in noto_pitches:
                midi.note_off(
                    note=noto_pitch, velocity=vel, channel=noto_channel)
            # feed
            for noto_pitch in noto_pitches:
                noto.feed(noto_inst, noto_pitch, 0, 0)
            noto.feed(player_inst, pitch, stopwatch.punch(), 0)

    @cleanup
    def _():
        """end any remaining notes"""
        for pitch in note_map.values():
            midi.note_off(note=pitch, velocity=0, channel=noto_channel)

if __name__=='__main__':
    run(main)
