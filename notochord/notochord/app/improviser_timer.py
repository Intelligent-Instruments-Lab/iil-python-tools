"""
Notochord MIDI co-improviser server.
Notochord plays different instruments along with the player.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

# TODO: make key bindings visibly click corresponding buttons
# TODO: make Mute a toggle but Reset a momentary

# TODO: color note by pitch class + register
# TODO: color instrument 1-128, MEL, DRUM, ANON, ANONDRUM
# TODO: color time
# TODO: id prediction as player / noto
# TODO: unify note log / prediction format
#   player/noto, channel, inst for both

from notochord import Notochord, MIDIConfig
from iipyper import MIDI, run, Stopwatch, Timer, cleanup, TUI, profile, _lock
from typing import Dict

import mido
import time

from rich.align import Align
from rich.panel import Panel
from rich.pretty import Pretty
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Button, TextLog

### def TUI components ###
class NotoLog(TextLog):
    value = reactive('')
    def watch_value(self, time: float) -> None:
        self.write(self.value)

class NotoPrediction(Static):
    value = reactive(None)
    def watch_value(self, time: float) -> None:
        evt = self.value
        if evt is None:
            return
        s = f"\tinstrument: {evt['inst']:03d}    pitch: {evt['pitch']:03d}    time: {int(evt['time']*1000):03d} ms    velocity:{int(evt['vel']):03d}"
        self.update(Panel(s, title='prediction'))

class NotoControl(Static):
    def compose(self):
        yield Button("Mute", id="mute", variant="error")
        yield Button("Reset", id="reset", variant='warning')

class NotoTUI(TUI):
    CSS_PATH = 'improviser.css'

    BINDINGS = [
        ("m", "mute", "Mute Notochord"),
        ("r", "reset", "Reset Notochord")]

    def compose(self):
        """Create child widgets for the app."""
        yield Header()
        yield self.std_log
        yield NotoLog(id='note')
        yield NotoPrediction(id='prediction')
        yield NotoControl()
        yield Footer()
### end def TUI components###

def main(
        player_config:Dict[int,int]=None, # map MIDI channel : instrument
        noto_config:Dict[int,int]=None, # map MIDI channel : instrument
        max_note_len=5, # in seconds, to auto-release stuck Notochord notes
        midi_in=None, # MIDI port for player input
        midi_out=None, # MIDI port for Notochord output
        n_recent=50,
        predict_player=True,
        testing=False,
        checkpoint="artifacts/notochord-latest.ckpt" # Notochord checkpoint
        ):
    """
    Args:
        player_config: mapping from MIDI channels to MIDI instruments controlled
            by the player.
        noto_config: mapping from MIDI channels to MIDI instruments controlled
            by notochord. Both indexed from 1.
            instruments should be different from the player instruments.
            channels should be different unless different ports are used.
            MIDI channels and General MIDI instruments are indexed from 1.
        max_note_len: time in seconds after which to force-release sustained
            notochord notes.
        midi_in: MIDI port for player input
        midi_out: MIDI port for Notochord output
        checkpoint: path to notochord model checkpoint
    """
    midi = MIDI(midi_in, midi_out)

    ### Textual UI
    tui = NotoTUI()
    print = tui.print
    def print_note(*args):
        tui(note=' '.join(str(a) for a in args))
    ###

    # default channel:instrument mappings
    if player_config is None:
        player_config = {1:257} # channel 1: anon 1
    if noto_config is None:
        noto_config = {2:258} # channel 2: anon 2

    # convert 1-indexed MIDI channels to 0-indexed here
    player_map = MIDIConfig({k-1:v for k,v in player_config.items()})
    noto_map = MIDIConfig({k-1:v for k,v in noto_config.items()})

    if len(player_map.insts & noto_map.insts):
        print("WARNING: Notochord and Player instruments shouldn't overlap")
        print('setting to an anonymous instrument')
        # TODO: set to anon insts without changing mel/drum
        # respecting anon insts selected for player
        raise NotImplementedError
    # TODO:
    # check for repeated insts/channels

    # load notochord model
    if checkpoint is not None:
        noto = Notochord.from_checkpoint(checkpoint)
        noto.eval()
        noto.reset()
    else:
        noto = None

    # main stopwatch to track time difference between MIDI events
    stopwatch = Stopwatch()

    # track recent instruments played for part balancing
    recent_insts = []
    def track_recent_insts(inst):
        recent_insts.append(inst)
        if len(recent_insts) > n_recent:
            recent_insts.pop(0)

    # class to track pending event prediction
    class Prediction:
        def __init__(self):
            self.event = None
            self.gate = True
            self.timer = None

        def update(self, event):
            with _lock:
                self.cancel()
                self.event = event
                print(event['time'] - stopwatch.read())
                self.timer = Timer(event['time'] - stopwatch.read(), do_event, lock=False)

        def cancel(self):
            if self.timer is not None:
                self.timer.cancel()              
            self.event = None

    pending = Prediction()

    # mapping from (inst, pitch) pairs to Stopwatches
    # to track sustained notochord notes
    notes = {}

    def noto_reset():
        noto.reset()
        for (inst,pitch) in notes:
            midi.note_off(note=pitch, velocity=0, channel=noto_map.inv(inst))
        notes.clear()
        recent_insts.clear()
        print('RESET')

    # query Notochord for a new next event
    def query():
        # check for stuck notes
        # and prioritize ending those
        for (inst, pitch), sw in notes.items():
            if sw.read() > max_note_len:
                # query for the end of a note with flexible timing
                with profile('query', print=print):
                    t = stopwatch.read()
                    pending.update(noto.query(
                        next_inst=inst, next_pitch=pitch,
                        next_vel=0, min_time=t, max_time=t+0.5))
                print('END STUCK NOTE')
                return

        # force sampling a notochord instrument which hasn't played recently
        insts = noto_map.insts - set(recent_insts)
        # if there is one
        if not len(insts):
            insts = noto_map.insts
            if predict_player:
                insts = insts | player_map.insts
        # print(f'considering {insts}')

        with profile('query', print=print):
            pending.update(noto.query(
                min_time=stopwatch.read(), # event can't happen sooner than now
                include_inst=insts,
                # steer_pitch=0.6,
                # steer_time=0.6,
                # steer_vel=0.5,
                ))
        # display the predicted event
        tui(prediction=pending.event)

    @midi.handle(type='program_change')
    def _(msg):
        """Program change events set instruments"""
        if msg.channel in player_map:
            player_map[msg.channel] = msg.program
        if msg.channel in noto_map:
            noto_map[msg.channel] = msg.program

    @midi.handle(type='control_change')
    def _(msg):
        """any CC1 message on player channel resets Notochord"""
        if msg.channel in player_map.channels:
            # print(msg)
            if msg.control==1:
                noto_reset()

    @midi.handle(type=('note_on', 'note_off'))
    def _(msg):
        """MIDI NoteOn events from the player"""
        if msg.channel not in player_map.channels:
            return

        if noto is None:
            print('Notochord model not loaded')
            return

        print_note('PLAYER:\t', msg)

        inst = player_map[msg.channel]
        pitch = msg.note
        vel = msg.velocity

        # feed event to Notochord
        with profile('feed', print=print):
            noto.feed(inst, pitch, stopwatch.punch(), vel)

        track_recent_insts(inst)

        # query for new prediction
        query()

        # for latency testing:
        if testing: midi.cc(control=3, value=msg.note, channel=15)

    def do_event():
        """
        The pending Notochord event happens: 
        send as MIDI, feedback to Notochord, track held notes
        """
        if testing or not pending.gate or pending.event is None: return

        event = pending.event
        inst, pitch, vel = event['inst'], event['pitch'], event['vel']

        if pending.event['inst'] in noto_map.insts:
            # notochord event happens

            # track held notes
            k = (inst, pitch)
            if vel > 0:
                notes[k] = Stopwatch(punch=True)
            else:
                if k in notes:
                    del notes[k]
                else:
                    # bad prediction: note-off without note-on
                    print('REPLACING INVALID PREDICTION')
                    query()
                    return

            # send as MIDI
            msg = mido.Message(
                type='note_on' if vel > 0 else 'note_off', 
                note=pitch, 
                velocity=int(vel), 
                channel=noto_map.inv(inst))
            midi.send(msg)

            # feed back to Notochord
            with profile('feed', print=print):
                noto.feed(inst, pitch, stopwatch.punch(), vel)
            # track total events played
            track_recent_insts(inst)
            # print
            print_note('NOTO:\t', msg)
        else:
            print('PLAYER PREDICTION ELAPSED')

        query()

    @cleanup
    def _():
        """end any remaining notes"""
        for (inst,pitch) in notes:
            midi.note_on(note=pitch, velocity=0, channel=noto_map.inv(inst))

    @tui.set_action
    def mute():
        pending.gate = not pending.gate
        # end+feed all held notes
        for (inst,pitch) in notes:
            midi.note_off(note=pitch, velocity=0, channel=noto_map.inv(inst))
            noto.feed(inst=inst, pitch=pitch, time=stopwatch.punch(), vel=0)
        notes.clear()
        # if unmuting make sure there is a pending event
        if pending.gate and pending.event is None:
            query()
    
    @tui.set_action
    def reset():
        # end all held notes
        noto_reset()

    # tui.run()


if __name__=='__main__':
    run(main)
