"""
Notochord MIDI co-improviser server.
Notochord plays different instruments along with the player.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

from typing import Optional, Dict
from numbers import Number
import random

from notochord import Notochord, MIDIConfig, NotoPerformance
from iipyper import OSC, MIDI, run, Stopwatch, repeat, cleanup, TUI, profile, lock

from rich.panel import Panel
from rich.pretty import Pretty
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
            s = f"\tinstrument: {evt['inst']:3d}    pitch: {evt['pitch']:3d}    time: {int(evt['time']*1000):4d} ms    velocity:{int(evt['vel']):3d}     end: {evt['end']:4f}"
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
    checkpoint="artifacts/noto-txala-011-0020.ckpt", # Notochord checkpoint
    player_config:Dict[int,int]=None, # map MIDI channel : GM instrument
    noto_config:Dict[int,int]=None, # map MIDI channel : GM instrument
    pitch_set=(41,43,45), # MIDI pitches used

    initial_mute=False, # start with Notochord muted
    initial_query=False, # let Notochord start playing immediately

    midi_in:Optional[str]=None, # MIDI port for player input
    midi_out:Optional[str]=None, # MIDI port for Notochord output
    thru=False, # copy player input to output
    send_pc=False, # send program change messages
    dump_midi=False, # print all incoming MIDI

    input_latency=0.02,
    rhythm_temp=0.9,
    timing_temp=0.2,
    steer_rate=None,

    auto_reset=True,

    start_after=2, # don't sample notochord until this many total events
    max_run=5,
    # balance_sample=True, # choose instruments which have played less recently
    n_recent=16, # number of recent note-on events to consider for above
    n_margin=5, # amount of 'slack' in the balance_sample calculation
    
    max_time=None, # max time between events
    nominal_time=False, #feed Notochord with nominal dt instead of actual
    backoff_time=1e-3, #time to wait when a predicted player event doesn't happen

    osc_port=None, # if supplied, listen for OSC to set controls on this port
    osc_host='', # leave this as empty string to get all traffic on the port

    use_tui=True, # run textual UI
    predict_player=True, # forecasted next events can be for player (preserves model distribution, but can lead to Notochord deciding not to play)
    auto_query=True, # query notochord whenever it is unmuted and there is no pending event. generally should be True except for testing purposes.
    testing=False
    ):
    """
    Args:
        checkpoint: path to notochord model checkpoint.

        player_config: mapping from MIDI channels to MIDI instruments controlled
            by the player.
        noto_config: mapping from MIDI channels to MIDI instruments controlled
            by notochord. Both indexed from 1.
            instruments should be different from the player instruments.
            channels should be different unless different ports are used.
            MIDI channels and General MIDI instruments are indexed from 1.

        pitch_set: collection of MIDI pitches for the txalaparta boards

        initial_mute: start Notochord muted so it won't play with input.
        initial_query: query Notochord immediately so it plays even without input.

        midi_in: MIDI ports for player input. 
            default is to use all input ports.
            can be comma-separated list of ports.
        midi_out: MIDI ports for Notochord output. 
            default is to use only virtual 'From iipyper' port.
            can be comma-separated list of ports.
        thru: if True, copy incoming MIDI to output ports.
            only makes sense if input and output ports are different.
        send_pc: if True, send MIDI program change messages to set the General MIDI
            instrument on each channel according to player_config and noto_config.
            useful when using a General MIDI synthesizer like fluidsynth.
        dump_midi: if True, print all incoming MIDI for debugging purposes

        balance_sample choose instruments which have played less recently
            ensures that all configured instruments will play.
        n_recent: number of recent note-on events to consider for above
        n_margin: amount of 'slack' in the balance_sample calculation

        max_time: maximum time in seconds between predicted events.
            default is the Notochord model's maximum (usually 10 seconds).
        nominal_time: if True, feed Notochord with its own predicted times
            instead of the actual elapsed time.
            May make Notochord more likely to play chords.

        osc_port: optional. if supplied, listen for OSC to set controls
        osc_host: hostname or IP of OSC sender.
            leave this as empty string to get all traffic on the port

        use_tui: run textual UI.
        predict_player: forecasted next events can be for player.
            generally should be True;
            instead use balance_sample to force Notochord to play.
        auto_query: query notochord whenever it is unmuted and there is no pending event. generally should be True unless debugging.
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
        # player_config = {1:265,2:266}
        player_config = {1:290,2:291}
    if noto_config is None:
        # noto_config = {3:267,4:268}
        noto_config = {3:292,4:293}

    state = {
        'run_length': 0,
        'last_side': None
    }

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

    def warn_inst(i):
        if i > 128:
            if i < 257:
                print(f"WARNING: drum instrument {i} selected, be sure to select a drum bank in your synthesizer")
            else:
                print(f"WARNING: instrument {i} is not General MIDI")

    if send_pc:
        for c,i in (player_map | noto_map).items():
            warn_inst(i)
            midi.program_change(channel=c, program=(i-1)%128)
    
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
    if steer_rate is not None:
        controls['steer_rate'] = steer_rate

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
        dt = stopwatch.punch(latency=input_latency if tag=='PLAYER' else 0)
        # dt = stopwatch.punch()
        # if tag=='PLAYER':
            # dt = max(0.001, dt - input_latency)
            # print(f'latency corrected: {dt}')
        if 'time' not in event or not nominal_time:
            event['time'] = dt

        side = int(event['inst']//2)
        if side==state['last_side']:
            state['run_length'] += 1
        else:
            state['run_length'] = 1

        state['last_side'] = side
        print(f'{state["run_length"]=}')

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
    def noto_reset(query=True):
        """reset Notochord and end all of its held notes"""
        print('RESET')

        # cancel pending predictions
        pending.event = None
        tui(prediction=pending.event)
        
        # reset stopwatch
        stopwatch.punch()
        # reset notochord state
        noto.reset()
        # reset history
        history.push()
        # query the fresh notochord for a new prediction
        if pending.gate and query:
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

    # query Notochord for a new next event
    # @lock
    def noto_query(delay=0):
        counts = history.inst_counts(
            n=n_recent, insts=noto_map.insts | player_map.insts)
        # print(counts)
        player_count = sum(counts[i] for i in player_map.insts)
        noto_count = sum(counts[i] for i in noto_map.insts)
        print(f'player: {player_count}')
        print(f'noto: {noto_count}')
        total_count = player_count + noto_count

        all_insts = noto_map.insts 
        if predict_player:
            all_insts = all_insts | player_map.insts

        steer_time = 1-controls.get('steer_rate', 0.5)
        tqt = (max(0,steer_time-0.5), min(1, steer_time+0.5))

        print(tqt)

        # if using nominal time,
        # *subtract* estimated feed latency to min_time; (TODO: really should
        #   set no min time when querying, use stopwatch when re-querying...)
        # if using actual time, *add* estimated query latency
        time_offset = -5e-3 if nominal_time else 10e-3
        min_time = stopwatch.read()+time_offset+delay

        if total_count < start_after:
            insts = player_map.insts
        else:
            insts = all_insts
            if state['run_length'] >= max_run:
                s = int(state['last_side']*2)
                insts = insts - {s, s+1}
            # balance_sample: note-ons only from entity which has played less
            # if balance_sample:                
            #     if player_count <= noto_count - n_margin:
            #         insts = player_map.insts
            #     elif noto_count <= player_count - n_margin:
            #         insts = noto_map.insts

        if len(insts)==0:
            insts = all_insts

        print(f'{insts=}')

        # bal_insts = set(counts.index[counts <= counts.min()+n_margin])
        # if balance_sample and len(bal_insts)>0:
        #     insts = bal_insts
        # else:
        #     insts = all_insts

        # query_method = noto.query_tipv_onsets

        if len(insts) > 2:
            # in this case *only* time is messed with,
            # so if we sample time first,
            # the rest can be properly conditioned on it
            query_method = noto.query_tipv_onsets
        else:
            # in this case, instrument and time have both been constrained,
            # and we can't sample the true joint distribution,
            # but we sample instrument first
            # under the assumption that the instrument constraint is
            # 'stronger' than the time constraint
            query_method = noto.query_itpv_onsets

        max_t = None if max_time is None else max(max_time, min_time+0.2)

        print(min_time, max_t)

        pending.event = query_method(
            include_pitch=pitch_set,
            include_inst=list(insts),
            min_time=min_time, max_time=max_t,
            truncate_quantile_time=tqt,
            min_vel=80, max_vel=120,
            rhythm_temp=rhythm_temp,
            timing_temp=timing_temp,
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
        controls['steer_pitch'] = (msg.pitch+8192)/16384
        # print(controls)

    # very basic CC handling for controls
    @midi.handle(type='control_change')
    def _(msg):
        """CC messages on any channel"""

        # if msg.control==1:
        #     controls['steer_pitch'] = msg.value/127
        #     print(f"{controls['steer_pitch']=}")
        # if msg.control==2:
        #     controls['steer_density'] = msg.value/127
        #     print(f"{controls['steer_density']=}")
        if msg.control==3:
            controls['steer_rate'] = msg.value/127
            print(f"{controls['steer_rate']=}")

        if msg.control==4:
            noto_reset()
        if msg.control==5:
            noto_query()
        if msg.control==6:
            noto_mute()

    # very basic OSC handling for controls
    if osc_port is not None:
        @osc.args('/notochord/improviser/*')
        def _(route, *a):
            print('OSC:', route, *a)
            ctrl = route.split['/'][3]
            if ctrl=='reset':
                noto_reset()
            elif ctrl=='query':
                noto_query()
            elif ctrl=='mute':
                noto_mute()
            else:
                assert len(a)==0
                arg = a[0]
                assert isinstance(arg, Number)
                controls[ctrl] = arg
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

        # send a MIDI reply for latency testing purposes:
        # if testing: midi.cc(control=3, value=msg.note, channel=15)

    def noto_event():
        # notochord event happens:
        event = pending.event
        inst, pitch, vel = event['inst'], event['pitch'], round(event['vel'])

        # note on which is already playing or note off which is not
        # if (vel>0) == ((inst, pitch) in history.note_pairs): 
        #     print(f're-query for invalid {vel=}, {inst=}, {pitch=}')
        #     noto_query()
        #     return
        
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
            e = pending.event
            if e['time'] >= noto.time_dist.hi.item():
                if auto_reset:
                    noto_reset(query=False)
            # if so, check if it is a notochord-controlled instrument
            if e['inst'] in noto_map.insts:
                # prediction happens
                noto_event()
                delay = 0
            else:
                delay = backoff_time
            # query for new prediction
            # print(pending.event)
            pending.event = None
            if ('end' in e and random.random() < e['end']):
                print('END')
                if auto_reset:
                    noto_reset(query=False)
            elif auto_query:
                noto_query(delay=delay)

    @cleanup
    def _():
        """"""
        pass
        # print(f'cleanup: {notes=}')
        # for (chan,inst,pitch) in history.note_triples:
        # # for (inst,pitch) in notes:
        #     if inst in noto_map.insts:
        #         midi.note_on(note=pitch, velocity=0, channel=chan)

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
