"""
Notochord MIDI harmonizer server.
Each note from the player produces a harmonizing note from Notochord.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""
import time

from notochord import Notochord, NotoPerformance
from iipyper import MIDI, run, Stopwatch, cleanup

def main(
        player_channel=0, # MIDI channel numbered from 0
        noto_channel=1, # channel for single notochord voice (overriden by noto_config)
        player_inst=257, # General MIDI numbered from 1 (see Notochord.feed docstring)
        noto_inst=257, # instrument for single notochord voice (overridden by noto_config)
        noto_config=None, # list of tuples of (channel, instrument, min transpose, max transpose)
        midi_in=None, # MIDI port(s) for player input
        midi_out=None, # MIDI port(s) for Notochord output
        thru=False, # copy player input to output
        checkpoint="artifacts/notochord-latest.ckpt", # Notochord checkpoint
        note_off_delay=2e-3,
        below=False, # harmonize above (overridden by noto_config)
        above=True, # harmonize below (overridden by noto_config)
        # n=1, # number of tones
        ):
    midi = MIDI(
        None if midi_in is None else midi_in.split(','), #TODO move to iipyper
        None if midi_out is None else midi_out.split(','))
    
    if noto_config is None:
        if not below and not above:
            raise ValueError
        noto_config = [[
            noto_channel, noto_inst, -128 if below else 1, 128 if above else -1]]
    
    # if noto_config is None:
    #     noto_config = [t for t in [ # channel (from 0), inst, min transpose, max transpose (inclusive)
    #         (0,44,-36,-12),
    #         (1,43,-36,-12),
    #         (2,43,-12,0), 
    #         (3,49,3,4), 
    #         (4,41,7,36), 
    #         ] if t[0] != player_channel]
    for (_,_,lo,hi) in noto_config:
        assert lo <= hi, """min transpose should be less than max transpose"""

    noto = Notochord.from_checkpoint(checkpoint)
    noto.eval()
    
    # note_map = {} # map from player pitch to notochord pitches
    history = NotoPerformance()
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
            # player_pitches = set(note_map) | {pitch}
            # chosen_pitches = set()
            # chosen_triples = []
            # get the harmonizing notes
            # for _ in range(n):
            for noto_channel, noto_inst, min_x, max_x in noto_config:

                already_playing = {i for i,p in history.note_pairs}

                lo, hi = noto_range
                pitch_range = range(max(lo,pitch+min_x), min(hi, pitch+max_x+1))
                pitches = (
                    set(pitch_range) - {pitch} - already_playing
                )

                if len(pitches)==0:
                    print(f'skipping {noto_channel=}, no pitches available')
                    print(pitch_range, 'minus', {pitch}, 'minus', already_playing)
                    continue

                # pitches = set((
                #     *(range(max(player_pitches)+1, 128) if above else []),
                #     *(range(0, min(player_pitches)) if below else [])
                # )) - chosen_pitches # don't repeat pitches

                # print(f'{pitches=}')

                h = noto.query(
                    next_inst=noto_inst, next_time=0, next_vel=vel,
                    include_pitch=pitches)
                print('NOTO:', h)

                h_inst = h['inst'] # noto_inst
                h_pitch = h['pitch']
                h_time = h['time'] # 0
                h_vel = round(h['vel'])

                # # support legato/polyphonic playing:
                # # check existing harmonizing notes for duplicates with the
                # # new harmonizing note.
                # # if duplicate, just transfer it in the note map,
                # # don't make a new note
                k = (noto_channel, h_inst, h_pitch) 
                if k in history.note_data:
                    history.note_data[k] = (player_inst, pitch)
                else:

                # new_note = True
                # for _,hs in note_map.items():
                #     for tr in list(hs):
                #         n_chan,n_inst,n_pitch = tr
                #         # if noto_inst==n_inst and n_pitch==h_pitch:
                #         if n_chan==noto_channel and n_pitch==h_pitch:
                #     # if n_inst==noto_inst and h_pitch in hs:
                #             hs.remove(tr)
                #             new_note = False
                # # the model *shouldn't* pick duplicates,
                # # but the thinking is that if it wants to,
                # # it's better to have fewer notes
                # # than to force it to pick something else

                # if new_note:
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
                # prepare for later NoteOff
                # chosen_pitches.add(h_pitch)
                # chosen_triples.append((noto_channel, h_inst, h_pitch))
            # note_map[pitch] = chosen_pitches
            # note_map[pitch] = chosen_triples
        # NoteOff
        else:
            # try:
            #     # noto_pitches = note_map.pop(pitch)
            #     noto_triples = note_map.pop(pitch)
            # except:
            #     print('harmonizing NoteOffs not found')
            #     return
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
                # send
                midi.note_off(
                    note=noto_pitch, velocity=vel, channel=noto_channel)
                event = dict(
                    channel=noto_channel, 
                    inst=noto_inst, pitch=noto_pitch, time=dt, vel=0)
                # TODO: nominal time option?
                noto.feed(**event)
                history.feed(**event)


            # # send harmonizing NoteOff
            # for noto_channel, noto_inst, noto_pitch in noto_triples:
            #     midi.note_off(
            #         note=noto_pitch, velocity=vel, channel=noto_channel)
            #     time.sleep(note_off_delay)
            # # feed
            # for noto_channel, noto_inst, noto_pitch in noto_triples:
            #     noto.feed(noto_inst, noto_pitch, 0, 0)
            # noto.feed(player_inst, pitch, stopwatch.punch(), 0)
            # history.feed(
            #     dict(inst=player_inst, pitch=pitch, vel=0))

    @cleanup
    def _():
        """end any remaining notes"""
        for (c,_,p) in history.note_triples:
            midi.note_off(note=p, velocity=0, channel=c)

        # for triples in note_map.values():
        #     for channel, _, pitch in triples:
        #         midi.note_off(note=pitch, velocity=0, channel=channel)

if __name__=='__main__':
    run(main)
