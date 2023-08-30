"""
Notochord MIDI harmonizer server.
Each note from the player produces a harmonizing note from Notochord.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""
from typing import Optional, List, Tuple

import mido

from notochord import Notochord, NotoPerformance
from iipyper import MIDI, run, Stopwatch, cleanup, TUI

from rich.panel import Panel
from rich.pretty import Pretty
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Button, TextLog

### def TUI components ###
class NotoLog(TextLog):
    value = reactive('')
    def watch_value(self, time: float) -> None:
        self.write(self.value)

class NotoControl(Static):
    def compose(self):
        yield Button("Mute", id="mute", variant="error")
        yield Button("Reset", id="reset", variant='warning')

class NotoTUI(TUI):
    CSS_PATH = 'harmonizer.css'

    BINDINGS = [
        ("m", "mute", "Mute Notochord"),
        ("r", "reset", "Reset Notochord")]

    def compose(self):
        """Create child widgets for the app."""
        yield Header()
        yield self.std_log
        yield NotoLog(id='note')
        yield NotoControl()
        yield Footer()

def main(
    checkpoint="artifacts/notochord-latest.ckpt", # Notochord checkpoint

    player_channel=1, # MIDI channel numbered from 0
    player_inst=1, # General MIDI numbered from 1 (see Notochord.feed docstring)
    noto_config:Optional[List[Tuple[int,int,int,int]]]=None, # list of tuples of (channel, instrument, min transpose, max transpose)
    noto_channel=2, # channel for single notochord voice (overriden by noto_config)
    noto_inst=1, # instrument for single notochord voice (overridden by noto_config)
    below=False, # harmonize above (overridden by noto_config)
    above=True, # harmonize below (overridden by noto_config)

    midi_in:Optional[str]=None, # MIDI port(s) for player input
    midi_out:Optional[str]=None, # MIDI port(s) for Notochord output
    thru=False, # copy player input to output
    send_pc=False, # send program change messages to match player and noto_config (useful if using a General MIDI synth like fluidsynth or hardware)

    use_tui=True,
    ):
    """
    Args:
        checkpoint: path to notochord model checkpoint.

        player_channel: MIDI channels for player input.
        player_inst: Notochord instrument for the player.
        noto_config: list voices played by Notochord. Each voice is a tuple of: (
            MIDI channel indexed from 1,
            General MIDI instrument from 1,
            minimum transpose from the performed pitch,
            maximum transpose
            ).
            For example, [(2,1,-12,0), (3,13,0,12)] would play the grand piano on channel 2 in the octave below, and the marimba on channel 3 in the octave above.
            see https://en.wikipedia.org/wiki/General_MIDI for instrument numbers.
        noto_channel: alternative to using noto_config for a single voice
        noto_inst: alternative to using noto_config for a single voice
        below: alternative to using noto_config for a single voice -- allows
            harmonizing notes below the performed pitch
        above: alternative to using noto_config for a single voice -- allows
            harmonizing notes above the performed pitch

        midi_in: MIDI ports for player input. 
            default is to use all input ports.
            can be comma-separated list of ports.
        midi_out: MIDI ports for Notochord output. 
            default is to use only virtual 'From iipyper' port.
            can be comma-separated list of ports.
        thru: if True, copy incoming MIDI to output ports.
            only makes sense if input and output ports are different.
        send_pc: if True, send MIDI program change messages to set the General MIDI
            instrument according to player_inst, player_channel and noto_config.
            useful when using a General MIDI synthesizer like fluidsynth.

        use_tui: run textual UI.
    """
        # nominal_time: if True, feed Notochord with its own predicted times
        #     instead of the actual elapsed time.
        #     May make Notochord more likely to play chords.
    midi = MIDI(midi_in, midi_out)

    ### Textual UI
    tui = NotoTUI()
    print = tui.print

    def display_event(tag, inst, pitch, vel, channel, **kw):
        """print an event to the terminal"""
        if tag is None:
            return
        s = f'{tag}:\t {inst=:4d}    {pitch=:4d}    {vel=:4d}    {channel=:3d}'
        tui(note=s)
    ###
    
    if noto_config is None:
        if not below and not above:
            raise ValueError
        noto_config = [[
            noto_channel-1, noto_inst, -128 if below else 1, 128 if above else -1]]
    # convert to 0-index
    player_channel = player_channel-1
        
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
    
    class AppState():
        def __init__(self):
            self.muted = False
    state = AppState()

    def noto_mute():
        state.muted = not state.muted
        print('MUTE' if state.muted else 'UNMUTE')

    def noto_reset():
        """reset Notochord and end all of its held notes"""
        print('RESET')

        # end Notochord held notes
        # noto_pairs = {(c,i) for (c,i,_,_) in noto_config}
        # for (c,i,p) in history.note_triples:
        #     if (c,i) in noto_pairs:
        #         midi.note_off(note=p, velocity=0, channel=c)

        # reset stopwatch
        stopwatch.punch()
        # reset notochord state
        noto.reset()
        # # reset history
        # history.push()

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
            display_event('PLAYER', **event)

            if state.muted:
                return

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

                h_inst = h['inst'] # noto_inst
                h_pitch = h['pitch']
                h_time = h['time'] # 0
                h_vel = round(h['vel'])

                # send it
                midi.note_on(note=h_pitch, velocity=h_vel, channel=noto_channel)
                # track
                event = dict(
                    channel=noto_channel,
                    inst=h_inst, pitch=h_pitch, time=h_time, vel=h_vel)
                history.feed(
                    held_note_data=(player_inst, pitch), 
                    **event)
                # feed back
                noto.feed(**event)
                display_event('NOTO', **event)
        # NoteOff
        else:
            dt = stopwatch.punch()
            event = dict(
                channel=player_channel, 
                inst=player_inst, pitch=pitch, time=dt, vel=0)
            noto.feed(**event)
            history.feed(**event)
            display_event('PLAYER', **event)

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
                display_event('NOTO', **event)

    @cleanup
    def _():
        """end any remaining notes"""
        for (c,_,p) in history.note_triples:
            midi.note_off(note=p, velocity=0, channel=c)

    @tui.set_action
    def mute():
        noto_mute()
    
    @tui.set_action
    def reset():
        noto_reset()

    if use_tui:
        tui.run()

if __name__=='__main__':
    run(main)