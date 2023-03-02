"""
Notochord MIDI harmonizer server.
Each note from the player produces a harmonizing note from Notochord.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""
from typing import Optional, List, Tuple

from notochord import Notochord, NotoPerformance
from iipyper import MIDI, run, Stopwatch, cleanup

def main(
        player_channel=0, # MIDI channel numbered from 0
        noto_channel=1, # channel for single notochord voice (overriden by noto_config)
        player_inst=257, # General MIDI numbered from 1 (see Notochord.feed docstring)
        noto_inst=257, # instrument for single notochord voice (overridden by noto_config)
        noto_config:Optional[List[Tuple[int,int,int,int]]]=None, # list of tuples of (channel, instrument, min transpose, max transpose)
        midi_in:Optional[str]=None, # MIDI port(s) for player input
        midi_out:Optional[str]=None, # MIDI port(s) for Notochord output
        thru=False, # copy player input to output
        checkpoint="artifacts/notochord-latest.ckpt", # Notochord checkpoint
        below=False, # harmonize above (overridden by noto_config)
        above=True, # harmonize below (overridden by noto_config)
        send_pc=False, # send program change messages to match player and noto_config (useful if using a General MIDI synth like fluidsynth or hardware)
        ):
    midi = MIDI(
        None if midi_in is None else midi_in.split(','), #TODO move to iipyper
        None if midi_out is None else midi_out.split(','))
    
    if noto_config is None:
        if not below and not above:
            raise ValueError
        noto_config = [[
            noto_channel, noto_inst, -128 if below else 1, 128 if above else -1]]
        
    # TODO: per-channel absolute range config

    def warn_inst(i):
        if i > 128:
            if i < 257:
                print(f"WARNING: drum instrument {i} selected, be sure to select a drum bank in your synthesizer")
            else:
                print(f"WARNING: instrument {i} is not General MIDI")

    if send_pc:
        warn_inst(player_inst)
        midi.program_change(channel=player_channel, program=(player_inst-1)%128)
        for (c,i,_,_) in noto_config:
            warn_inst(i)
            midi.program_change(channel=c, program=(i-1)%128)

    
    for (_,_,lo,hi) in noto_config:
        assert lo <= hi, """min transpose should be less than max transpose"""

    noto = Notochord.from_checkpoint(checkpoint)
    noto.eval()
    
    history = NotoPerformance()
    stopwatch = Stopwatch()

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

        print('PLAYER:', msg)

        pitch = msg.note
        vel = msg.velocity

        if thru:
            midi.send(msg)

        noto_range = (0,127) # TODO: config this

        # NoteOn
        if msg.type=='note_on' and vel > 0:
            dt = stopwatch.punch()
            # track
            event = dict(
                channel=player_channel,
                inst=player_inst, pitch=pitch, vel=vel, time=dt)
            history.feed(**event)
            # feed in the performed note
            noto.feed(**event)

            for noto_channel, noto_inst, min_x, max_x in noto_config:

                already_playing = {p for i,p in history.note_pairs if noto_inst==i}

                lo, hi = noto_range
                pitch_range = range(max(lo,pitch+min_x), min(hi, pitch+max_x+1))
                pitches = (
                    set(pitch_range) - {pitch} - already_playing
                )

                if len(pitches)==0:
                    # edge case: no possible pitch
                    print(f'skipping {noto_channel=}, no pitches available')
                    print(pitch_range, 'minus', {pitch}, 'minus', already_playing)
                    continue
                elif len(pitches)==1:
                    # edge case: there is exactly one possible pitch
                    h = dict(
                        inst=noto_inst, pitch=list(pitches)[0], 
                        time=0, vel=vel)
                else:
                    # notochord chooses pitch
                    h = noto.query(
                        next_inst=noto_inst, next_time=0, next_vel=vel,
                        include_pitch=pitches)
                print('NOTO:', h)

                h_inst = h['inst'] # noto_inst
                h_pitch = h['pitch']
                h_time = h['time'] # 0
                h_vel = round(h['vel'])

                # send it
                midi.note_on(
                    note=h_pitch, velocity=h_vel, channel=noto_channel)
                # track
                event = dict(
                    channel=noto_channel,
                    inst=h_inst, pitch=h_pitch, time=h_time, vel=h_vel)
                history.feed(
                    held_note_data=(player_inst, pitch), 
                    **event)
                # feed back
                noto.feed(**event)
        # NoteOff
        else:
            dt = stopwatch.punch()
            event = dict(
                channel=player_channel, 
                inst=player_inst, pitch=pitch, time=dt, vel=0)
            noto.feed(**event)
            history.feed(**event)

            dependents = [
                noto_k
                for noto_k,player_k 
                in history.note_data.items()
                if player_k==(player_inst, pitch)
            ]

            for noto_channel, noto_inst, noto_pitch in dependents:
                # send harmonizing note offs
                midi.note_off(
                    note=noto_pitch, velocity=vel, channel=noto_channel)
                event = dict(
                    channel=noto_channel, 
                    inst=noto_inst, pitch=noto_pitch, time=dt, vel=0)
                # TODO: nominal time option?
                noto.feed(**event)
                history.feed(**event)

    @cleanup
    def _():
        """end any remaining notes"""
        for (c,_,p) in history.note_triples:
            midi.note_off(note=p, velocity=0, channel=c)

if __name__=='__main__':
    run(main)