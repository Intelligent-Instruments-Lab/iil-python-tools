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
# TODO: grey out predictions when player or muted notochord

from notochord import Notochord, MIDIConfig, NotoPerformance
from iipyper import OSC, MIDI, run, Stopwatch, repeat, cleanup, TUI, profile, lock
from typing import Dict

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
            s = ''
        else:
            s = f"\tinstrument: {evt['inst']:3d}    pitch: {evt['pitch']:3d}    time: {int(evt['time']*1000):4d} ms    velocity:{int(evt['vel']):3d}"
        self.update(Panel(s, title='prediction'))

class NotoControl(Static):
    def compose(self):
        yield Button("Mute", id="mute", variant="error")
        yield Button("Query", id="query")
        yield Button("Reset", id="reset", variant='warning')

class NotoTUI(TUI):
    CSS_PATH = 'improviser.css'

    BINDINGS = [
        ("m", "mute", "Mute Notochord"),
        ("q", "query", "Re-query Notochord"),
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
        player_config:Dict[int,int]=None, # map MIDI channel : GM instrument
        noto_config:Dict[int,int]=None, # map MIDI channel : GM instrument
        max_note_len=5, # in seconds, to auto-release stuck Notochord notes
        midi_in=None, # MIDI port for player input
        midi_out=None, # MIDI port for Notochord output
        balance_sample=False, # Notochord won't play more notes than the player
        force_sample=True, # force prediction of not-recently seen instruments
        n_recent=24, # number of recent events to consider for above
        osc_host='',
        osc_port=None,
        predict_player=True, # forecasted next events can be for player (preserves model distribution, but can lead to Notochord deciding not to play)
        use_tui=True, # run textual UI
        nominal_time=False, #feed Notochord with nominal dt instead of actual
        thru=False, # copy player input to output
        initial_mute=False, # start with Notochord muted
        initial_query=False, # let Notochord start playing immediately
        checkpoint="artifacts/notochord-latest.ckpt", # Notochord checkpoint
        testing=False,
        dump_midi=False,
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
    if osc_port is not None:
        osc = OSC(osc_host, osc_port)
    midi = MIDI(midi_in, midi_out)

    ### Textual UI
    tui = NotoTUI()
    print = tui.print
    ###

    # default channel:instrument mappings
    if player_config is None:
        player_config = {1:257} # channel 1: anon
    if noto_config is None:
        noto_config = {2:258} # channel 2: anon

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

    # TODO: add arguments for this,
    # and sensible defaults for drums etc
    inst_pitch_map = {i: range(128) for i in noto_map.insts | player_map.insts}

    # print(inst_pitch_map)

    # load notochord model
    try:
        noto = Notochord.from_checkpoint(checkpoint)
        noto.eval()
        noto.reset()
    except Exception:
        print("""error loading notochord model""")
        raise

    # main stopwatch to track time difference between MIDI events
    stopwatch = Stopwatch()

    # simple class to hold pending event prediction
    class Prediction:
        def __init__(self):
            self.event = None
            self.gate = not initial_mute
    pending = Prediction()

    # query parameters controlled via MIDI / OSC
    controls = {}

    # tracks held notes, recently played instruments, etc
    history = NotoPerformance()

    def display_event(tag, memo, inst, pitch, vel, channel, **kw):
        """print an event to the terminal"""
        if tag is None:
            return
        s = f'{tag}:\t {inst=:4d}    {pitch=:4d}    {vel=:4d}    {channel=:3d}'
        if memo is not None:
            s += f'    ({memo})'
        tui(note=s)

    def play_event(event, channel, feed=True, send=True, tag=None, memo=None):
        """realize an event as MIDI, terminal display, and Notochord update"""
        # normalize values
        vel = event['vel'] = round(event['vel'])
        dt = stopwatch.punch()
        if 'time' not in event or not nominal_time:
            event['time'] = dt

        # send out as MIDI
        if send:
            midi.send(
                'note_on' if vel > 0 else 'note_off', 
                note=event['pitch'], velocity=vel, channel=channel)

        # feed to NotoPerformance
        # put a stopwatch in the held_note_data field for tracking note length
        history.feed(held_note_data=Stopwatch(), channel=channel, **event)

        # print
        display_event(tag, memo=memo, channel=channel, **event)

        # feed to Notochord
        if feed:
            noto.feed(**event)

    # @lock
    def noto_reset():
        """reset Notochord and end all of its held notes"""
        print('RESET')

        # cancel pending predictions
        pending.event = None
        tui(prediction=pending.event)
        
        # end Notochord held notes
        for (inst,pitch) in history.note_pairs:
            if inst in noto_map.insts:
                play_event(
                    dict(inst=inst, pitch=pitch, vel=0),
                    channel=noto_map.inv(inst), 
                    feed=False, # skip feeding Notochord since we are resetting it
                    tag='NOTO', memo='reset')
        # reset stopwatch
        stopwatch.punch()
        # reset notochord state
        noto.reset()
        # reset history
        history.push()
        # query the fresh notochord for a new prediction
        if pending.gate:
            noto_query()

    # @lock
    def noto_mute():
        pending.gate = not pending.gate
        print('UNMUTE' if pending.gate else 'MUTE')
        # if unmuting, we're done
        if pending.gate:
            return
        # cancel pending predictions
        pending.event = None
        tui(prediction=pending.event)
        # end+feed all held notes
        for (inst,pitch) in history.note_pairs:
            try:
                channel = noto_map.inv(inst)
            except KeyError:
                channel = player_map.inv(inst)
            play_event(
                dict(inst=inst, pitch=pitch, vel=0), 
                channel=channel, tag='NOTO', memo='mute')

    def query_steer_time(insts):
        # with profile('query_ivtp', print=print):
        inst_range_map = {k:inst_pitch_map[k] for k in insts}
        held_note_map = history.held_inst_pitch_map(insts)
        pending.event = noto.query_ivtp(
            inst_range_map, 
            held_note_map, 
            min_time=stopwatch.read(), # event can't happen sooner than now
            steer_duration=controls.get('steer_duration', None),
            steer_density=controls.get('steer_density', None),
            steer_pitch=controls.get('steer_pitch', None)
        )

    # query Notochord for a new next event
    # @lock
    def noto_query():
        # check for stuck notes
        # and prioritize ending those
        for (inst, pitch), sw in history.note_data.items():
            if (
                inst in noto_map.insts 
                and sw.read() > max_note_len*controls.get('steer_duration', 1)
                ):
                # query for the end of a note with flexible timing
                # with profile('query', print=print):
                t = stopwatch.read()
                pending.event = noto.query(
                    next_inst=inst, next_pitch=pitch,
                    next_vel=0, min_time=t, max_time=t+0.5)
                print(f'END STUCK NOTE {inst=},{pitch=}')
                return

        counts = history.inst_counts(
            n=n_recent, insts=noto_map.insts | player_map.insts)
        print(counts)
        # force sampling a notochord instrument which hasn't played recently
        if force_sample:
            # insts = noto_map.insts - set(recent_insts)
            recent_insts = set(counts[counts > 0])
            insts = noto_map.insts - recent_insts
        else:
            insts = []

        if balance_sample:
            min_count = counts.min()
            insts = set(counts.index[counts <= min_count+8])
            # always allow instruments which have a held note
            insts |= set(i for i,p in history.note_pairs)
            print(insts)
            # player_count = sum(v for k,v in counts.items() if k in player_map.insts)
            # noto_count = sum(v for k,v in counts.items() if k in noto_map.insts)
            # print(f'{player_count=}, {noto_count=}')
            # print(counts)
            # if noto_count > player_count - 3:
                # insts |= player_map.insts
            # if player_count > noto_count - 3:
                # insts |= noto_map.insts
            # if counts['noto'] > counts['player'] - 3:
            #     insts |= player_map.insts
            # if counts['player'] > counts['noto'] - 3:
            #     insts |= noto_map.insts

        # if there is one
        if not len(insts):
            insts = noto_map.insts
            if predict_player:
                insts = insts | player_map.insts
        # print(f'considering {insts}')

        # query_steer_time(insts)
        if 'steer_pitch' in controls or 'steer_density' in controls or 'steer_duration' in controls:
            query_steer_time(insts)
        else:
            # with profile('query', print=print):
            pending.event = noto.query(
                min_time=stopwatch.read(), # event can't happen sooner than now
                include_inst=insts,
                # steer_pitch=controls.get('steer_pitch', None),
                )
        # display the predicted event
        tui(prediction=pending.event)


    #### MIDI handling

    # print all incoming MIDI for debugging
    if dump_midi:
        @midi.handle
        def _(msg):
            print(msg)

    @midi.handle(type='program_change')
    def _(msg):
        """Program change events set instruments"""
        if msg.channel in player_map:
            player_map[msg.channel] = msg.program
        if msg.channel in noto_map:
            noto_map[msg.channel] = msg.program

    @midi.handle(type='pitchwheel')
    def _(msg):
        # print(msg.pitch)
        controls['steer_pitch'] = (msg.pitch+8192)/16384
        # print(controls)

    # very basic CC handling for controls
    @midi.handle(type='control_change')
    def _(msg):
        """CC messages on any channel"""

        if msg.control==4:
            noto_reset()
        if msg.control==5:
            noto_query()
        if msg.control==5:
            mute()

        if msg.control==1:
            controls['steer_pitch'] = msg.value/127
            print(f"{controls['steer_pitch']=}")
        if msg.control==2:
            controls['steer_density'] = msg.value/127
            print(f"{controls['steer_density']=}")
        if msg.control==3:
            controls['steer_duration'] = msg.value/127
            print(f"{controls['steer_duration']=}")

    # very basic OSC handling for controls
    if osc_port is not None:
        @osc.args('/*')
        def _(route, *a):
            print('OSC:', route, *a)
            ctrl = route.split['/'][1]
            if ctrl=='reset':
                noto_reset()
            if ctrl=='query':
                noto_query()
            if ctrl=='mute':
                mute()

            controls[ctrl] = a[0]
            print(controls)

    @midi.handle(type=('note_on', 'note_off'))
    def _(msg):
        """MIDI NoteOn events from the player"""
        # if thru and msg.channel not in noto_map.channels:
            # midi.send(msg)

        if msg.channel not in player_map.channels:
            return
        
        inst = player_map[msg.channel]
        pitch = msg.note
        vel = msg.velocity if msg.type=='note_on' else 0

        # feed event to Notochord
        # with profile('feed', print=print):
        play_event(
            {'inst':inst, 'pitch':pitch, 'vel':vel}, 
            channel=msg.channel, send=thru, tag='PLAYER')

        # query for new prediction
        noto_query()

        # for latency testing:
        # if testing: midi.cc(control=3, value=msg.note, channel=15)

    def noto_event():
        # notochord event happens:
        event = pending.event
        inst, pitch, vel = event['inst'], event['pitch'], round(event['vel'])

        # note on which is already playing or note off which is not
        if (vel>0) == ((inst, pitch) in history.note_pairs): 
            print(f're-query for invalid {vel=}, {inst=}, {pitch=}')
            noto_query()
            return
        
        chan = noto_map.inv(inst)
        play_event(event, channel=chan, tag='NOTO')

    @repeat(1e-3, lock=True)
    def _():
        """Loop, checking if predicted next event happens"""
        # check if current prediction has passed
        if (
            not testing and
            pending.gate and
            pending.event is not None and
            stopwatch.read() > pending.event['time']
            ):
            # if so, check if it is a notochord-controlled instrument
            if pending.event['inst'] in noto_map.insts:
                # prediction happens
                noto_event()
            # query for new prediction
            noto_query()

    @cleanup
    def _():
        """end any remaining notes"""
        # print(f'cleanup: {notes=}')
        for (inst,pitch) in history.note_pairs:
        # for (inst,pitch) in notes:
            if inst in noto_map.insts:
                midi.note_on(note=pitch, velocity=0, channel=noto_map.inv(inst))

    @tui.set_action
    def mute():
        noto_mute()
    
    @tui.set_action
    def reset():
        noto_reset()
    
    @tui.set_action
    def query():
        noto_query()

    if initial_query:
        noto_query()

    if use_tui:
        tui.run()


if __name__=='__main__':
    run(main)
