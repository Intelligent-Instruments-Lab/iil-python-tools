"""
Notochord MIDI co-improviser server.
Notochord plays a different instrument along with the player.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

from notochord import Notochord, MIDIConfig
from iipyper import MIDI, run, Timer, repeat, cleanup
from typing import Dict

import mido
# import time

from rich.panel import Panel
from rich.pretty import Pretty

from textual.app import App, ComposeResult
from textual.reactive import reactive
# from textual import events
from textual.widgets import Header, Footer, Static, Button, TextLog, Label

class NotoPrediction(Static):
    value = reactive({})

    def watch_value(self, time: float) -> None:
        """Called when the value attribute changes."""
        self.update(Panel(Pretty(self.value), title='prediction'))

class NotoControl(Static):
    def compose(self) -> ComposeResult:
        """"""
        yield Button("Mute", id="mute", variant="error")
        yield Button("Reset", id="reset")

class NotoApp(App):
    """A Textual app to manage stopwatches."""

    CSS_PATH = 'improviser.css'

    BINDINGS = [
        ("m", "mute", "Mute Notochord"),
        ("r", "reset", "Reset Notochord")]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        getattr(self, f'action_{event.button.id}')()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        self.note_log = TextLog()
        yield self.note_log

        self.prediction = NotoPrediction()
        yield self.prediction

        yield NotoControl()

        yield Footer()


    def set_action(self, f):
        # print(f'action_{f.__name__}')
        setattr(self, f'action_{f.__name__}', f)

    def __call__(self, *a, **kw):
        if self.is_running:
            self.call_from_thread(self.do_call, *a, **kw)

    def do_call(self, prediction=None, note=None):
        if not self.is_running:
            return
        if prediction is not None:
            self.prediction.value = prediction
        if note is not None:
            self.note_log.write(note)

def main(
        player_config:Dict[int,int]=None, # map MIDI channel : instrument
        noto_config:Dict[int,int]=None, # map MIDI channel : instrument
        max_note_len=5, # in seconds, to auto-release stuck Notochord notes
        midi_in=None, # MIDI port for player input
        midi_out=None, # MIDI port for Notochord output
        checkpoint="artifacts/notochord-latest.ckpt" # Notochord checkpoint
        ):
    midi = MIDI(midi_in, midi_out)

    ### Textual UI
    ui = NotoApp()
    def print_note(*args):
        print(*args)
        ui(note=' '.join(str(a) for a in args))
    ###

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
            self.gate = True
    pending = Prediction(None)

    noto_events = Counter()
    player_events = Counter()

    # map (inst, pitch) pairs to Timers
    # delete keys when there is a note off
    notes = {}

    def query_end(inst, pitch):
        pending.event = noto.query(
            next_inst=inst,
            next_pitch=pitch,
            next_vel=0,
            max_time=0.5)

    def query():
        # check for stuck notes
        # and prioritize ending those
        for (inst,pitch),t in notes.items():
            if t.read() > max_note_len:
                query_end(inst,pitch)
                print('END STUCK NOTE')
                return

        # TODO: controls for this
        # force sampling Notochord-controlled instruments
        # when it has played much less than the player 
        insts = noto_map.insts
        if noto_events() > 2*player_events():
            insts = insts | player_map.insts

        pending.event = noto.query(
            min_time=timer.read(),
            include_inst=insts)
        ui(prediction=pending.event)

    @midi.handle(type='program_change')
    def _(msg):
        """Program change events set instruments"""
        if msg.channel in player_map:
            player_map[msg.channel] = msg.program
        if msg.channel in noto_map:
            noto_map[msg.channel] = msg.program

    @midi.handle(type='control_change', control=0)
    def _(msg):
        """any CC0 message on player channel resets Notochord"""
        if msg.channel in player_map.channels:
            noto.reset()

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
        # query for new prediction
        noto.feed(inst, pitch, timer.punch(), vel)
        query()
        player_events.plus()
        # print(timer.read())

    def noto_event():
        """
        The pending Notochord event happens: 
        send as MIDI, feedback to Notochord, track held notes
        """
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
                # print_right('REPLACING BAD PREDICTION', pending.event)
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
        noto.feed(inst, pitch, timer.punch(), vel)
        # track total events played
        noto_events.plus()
        # print
        print_note('NOTO:\t', msg)

    @repeat(5e-4, lock=True)
    def _():
        """Loop, checking if predicted next event happens"""
        # check if current prediction has passed
        if pending.gate and pending.event is not None and timer.read() > pending.event['time']:
        # if so, check if it is a notochord-controlled instrument
            if pending.event['inst'] in noto_map.insts:
                # prediction happens -- send and feed
                noto_event()
            # query for new prediction
            query()

    @cleanup
    def _():
        """end any remaining notes"""
        for (inst,pitch) in notes:
            midi.note_on(note=pitch, velocity=0, channel=noto_map.inv(inst))

    @ui.set_action
    def mute():
        pending.gate = not pending.gate
        # end+feed all held notes
        for (inst,pitch) in notes:
            midi.note_off(note=pitch, velocity=0, channel=noto_map.inv(inst))
            noto.feed(inst=inst, pitch=pitch, time=timer.punch(), vel=0)
        notes.clear()
        # if unmuting make sure there is a pending event
        if pending.gate and pending.event is None:
            query()
    
    @ui.set_action
    def reset():
        # end all held notes
        noto.reset()
        for (inst,pitch) in notes:
            midi.note_off(note=pitch, velocity=0, channel=noto_map.inv(inst))
        notes.clear()

    ui.run()


if __name__=='__main__':
    run(main)
