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

# TODO: controls display panel
# TODO: MIDI learn

# TODO: held notes display panel

from typing import Optional, Dict, Any
from numbers import Number

import iipyper, notochord
from notochord import Notochord, MIDIConfig, NotoPerformance
from iipyper import OSC, MIDI, run, Stopwatch, repeat, cleanup, TUI, profile, lock

from rich.panel import Panel
from rich.pretty import Pretty
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Button, TextLog, Switch, Checkbox

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

# class NotoToggle(Static):
#     def compose(self):
#         yield Button("Mute", id="mute", variant="error")
#         yield Switch()

class NotoControl(Static):
    def compose(self):
        # yield NotoToggle()
        # yield Checkbox("Mute", id="mute")
        yield Button("Mute", id="mute", variant="error")
        yield Button("Sustain", id="sustain", variant="primary")
        yield Button("Query", id="query")
        yield Button("Reset", id="reset", variant='warning')

class NotoTUI(TUI):
    CSS_PATH = 'improviser.css'

    BINDINGS = [
        ("m", "mute", "Mute Notochord"),
        ("s", "sustain", "Mute without ending notes"),
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
    checkpoint="artifacts/notochord-latest.ckpt", # Notochord checkpoint
    config:Dict[int,Dict[str,Any]]=None, # map MIDI channel : GM instrument

    initial_mute=False, # start with Notochord muted
    initial_query=False, # let Notochord start playing immediately

    midi_in:Optional[str]=None, # MIDI port for player input
    midi_out:Optional[str]=None, # MIDI port for Notochord output
    thru=False, # copy player input to output
    send_pc=False, # send program change messages
    dump_midi=False, # print all incoming MIDI

    balance_sample=False, # choose instruments which have played less recently
    n_recent=64, # number of recent note-on events to consider for above
    n_margin=8, # amount of 'slack' in the balance_sample calculation
    
    max_note_len=5, # in seconds, to auto-release stuck Notochord notes
    max_time=None, # max time between events
    nominal_time=False, #feed Notochord with nominal dt instead of actual

    osc_port=None, # if supplied, listen for OSC to set controls on this port
    osc_host='', # leave this as empty string to get all traffic on the port

    use_tui=True, # run textual UI
    predict_input=True, # forecasted next events can be for input (preserves model distribution, but can lead to Notochord deciding not to play)
    predict_follow=False,
    debug_query=False, # don't query notochord when there is no pending event.
    testing=False,
    estimated_latency=1e-2
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


        config: mapping from MIDI channels to voice specs.
            MIDI channels and General MIDI instruments are indexed from 1.
            see https://en.wikipedia.org/wiki/General_MIDI for instrument numbers.
            There are 3 types of voice: 'auto', 'follow' and 'input'.
            Example:
            {
                1:{
                    'type':'input', 'inst':1
                }, # input grand piano on MIDI channel 1
                2:{
                    'type':'follow', 'source':1, 'inst':1, 
                    'transpose':(-12,12)
                }, # harmonize the channel within 1 octave 1 with more piano
                3:{
                    'type':'auto', 'inst':12, 'range':(36,72)
                }, # autonomous vibraphone voice in the MIDI pitch 36-72 range
                10:{
                    'type':'auto', 'inst':129,
                }, # autonomous drums voice
                4:{
                    'type':'follow', 'source':3, 'inst':10, 'range':(72,96)
                }, # harmonize channel 3 within upper registers of the glockenspiel
            }
            Notes:
            no 'input' or 'auto' channels should use the same instrument,
            but 'follow' channels may have the same as an 'input' or 'auto'

        initial_mute: start 'auto' voices muted so it won't play with input.
        initial_query: query Notochord immediately,
            so 'auto' voices begin playing  without input.

        midi_in: MIDI ports for input. 
            default is to use all input ports.
            can be comma-separated list of ports.
        midi_out: MIDI ports for output. 
            default is to use only virtual 'From iipyper' port.
            can be comma-separated list of ports.
        thru: if True, copy input MIDI to output ports.
            only makes sense if input and output ports are different.
        send_pc: if True, send MIDI program change messages to set the General MIDI
            instrument on each channel according to player_config and noto_config.
            useful when using a General MIDI synthesizer like fluidsynth.
        dump_midi: if True, print all incoming MIDI for debugging purposes

        balance_sample choose 'auto' voices which have played less recently,
            ensures that all configured instruments will play.
        n_recent: number of recent note-on events to consider for above
        n_margin: amount of 'slack' in the balance_sample calculation

        max_note_len: time in seconds after which to force-release sustained
            'auto' notes.
        max_time: maximum seconds between predicted events for 'auto' voices.
            default is the Notochord model's maximum (usually 10 seconds).
        nominal_time: if True, feed Notochord with its own predicted times
            instead of the actual elapsed time.
            May make Notochord more likely to play chords.

        osc_port: optional. if supplied, listen for OSC to set controls
        osc_host: hostname or IP of OSC sender.
            leave this as empty string to get all traffic on the port

        use_tui: run textual UI.
        predict_input: forecasted next events can be for 'input' or 'follow' voices.
            generally should be true, use balance_sample to force 'auto' voices to
            play.
        debug_query=False, # don't query notochord when there is no pending event.
    """
    if osc_port is not None:
        osc = OSC(osc_host, osc_port)
    midi = MIDI(midi_in, midi_out)

    ### Textual UI
    tui = NotoTUI()
    print = notochord.print = iipyper.print = tui.print
    ###

    # default channel:instrument mappings
    if config is None:
        config = {
            1:{'type':'input', 'inst':1},
            2:{'type':'follow', 'inst':1, 'source':1, 'transpose':(3,15)},
            3:{'type':'follow', 'inst':10, 'source':2, 'range':(72,96)},
            4:{'type':'auto', 'inst':12},
            5:{'type':'follow', 'source':4, 'inst':12, 'transpose':(-15,-3)},
            10:{'type':'auto', 'inst':129},
        }

    def validate_config():
        assert all(
            v['source'] in config for v in config.values() if v['type']=='follow'
            ), 'ERROR: no source given for follow voice'
        # TODO: check for follow cycles
    validate_config()

    def type_insts(t):
        if isinstance(t, str):
            t = t,
        # set of instruments with given type(s)
        return {v['inst'] for v in config.values() if v['type'] in t}
    def type_chans(t):
        if isinstance(t, str):
            t = t,
        # list of channels with given type
        return [k for k,v in config.items() if v['type'] in t]
    def channel_inst(c):
        return config[c]['inst']
    def channel_insts():
        # list of channel,instrument pairs
        return [(c,channel_inst(c)) for c in config]
    def inst_ranges(insts):
        # instruments to sets of allowed MIDI numbers
        # TODO: better defaults here
        r = {}
        for v in config.values():
            i = v['inst']
            if i in insts:
                s = set(v.get('range', range(128)))
                if i in r:
                    r[i] |= s
                else:
                    r[i] = s
        return r
    def auto_inst_channel(i):
        for k,v in config.items():
            if v['inst']==i:
                return k
        raise ValueError
    def channel_followers(chan):
        # return channel of all 'follow' voices with given source
        return [k for k,v in config.items() if v.get('source', None)==chan]
    
    if len(type_insts('input') & type_insts('auto')):
        print("WARNING: auto and input instruments shouldn't overlap")
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

    def do_send_pc():
        for c,i in channel_insts():
            warn_inst(i)
            # convert to 0-index
            midi.program_change(channel=c-1, program=(i-1)%128)

    if send_pc:
        do_send_pc()
    
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

    def play_event(
            event, channel, 
            parent=None, # parent note as (channel, inst, pitch)
            feed=True, 
            send=True, 
            tag=None, memo=None):
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
                note=event['pitch'], velocity=vel, channel=channel-1)

        # feed to NotoPerformance
        # put a stopwatch in the held_note_data field for tracking note length
        history.feed(held_note_data={
            'duration':Stopwatch(),
            'parent':parent
            }, channel=channel, **event)

        # print
        display_event(tag, memo=memo, channel=channel, **event)

        # feed to Notochord
        if feed:
            noto.feed(**event)

        follow_event(event, channel)

    def follow_event(source_event, source_channel):
        source_vel = source_event['vel']
        source_pitch = source_event['pitch']
        source_inst = source_event['inst']
        source_k = (source_channel, source_inst, source_pitch)

        # TODO: process events from 'follow' channels

        dt = 0 if nominal_time else estimated_latency

        if source_vel > 0:
            # NoteOn
            for noto_channel in channel_followers(source_channel):
                cfg = config[noto_channel]
                
                if cfg.get('muted', False): continue

                noto_inst = cfg['inst']
                min_x, max_x = cfg.get('transpose', (-128,128))
                lo, hi = cfg.get('range', (0,127))

                already_playing = {p for i,p in history.note_pairs if noto_inst==i}
                # print(f'{already_playing=}')

                pitch_range = range(
                    max(lo,source_pitch+min_x), min(hi, source_pitch+max_x+1))
                pitches = (
                    set(pitch_range) - {source_pitch} - already_playing
                )

                if len(pitches)==0:
                    # edge case: no possible pitch
                    print(f'skipping {noto_channel=}, no pitches available')
                    print(f'{pitch_range} minus {{source_pitch}} minus {already_playing}')
                    continue
                elif len(pitches)==1:
                    # edge case: there is exactly one possible pitch
                    h = dict(
                        inst=noto_inst, pitch=list(pitches)[0], 
                        time=0, vel=source_vel)
                else:
                    # notochord chooses pitch
                    h = noto.query(
                        next_inst=noto_inst, next_time=dt, next_vel=source_vel,
                        include_pitch=pitches)
                    
                play_event(h, noto_channel, parent=source_k, tag='NOTO', memo='follow')
        # NoteOff
        else:
            # print(f'{history.note_data=}')
            dependents = [
                noto_k # chan, inst, pitch
                for noto_k, note_data
                in history.note_data.items()
                if note_data['parent']==source_k
            ]

            for noto_channel, noto_inst, noto_pitch in dependents:
                h = dict(inst=noto_inst, pitch=noto_pitch, time=dt, vel=0)
                play_event(h, noto_channel,tag='NOTO', memo='follow')

    # @lock
    def noto_reset():
        """reset Notochord and end all of its held notes"""
        print('RESET')

        # cancel pending predictions
        pending.event = None
        tui(prediction=pending.event)
        
        # end Notochord held notes
        for (chan,inst,pitch) in history.note_triples:
            if inst in type_insts('auto'):
                play_event(
                    dict(inst=inst, pitch=pitch, vel=0),
                    channel=chan, 
                    feed=False, # skip feeding Notochord since we are resetting it
                    tag='NOTO', memo='reset')
        # reset stopwatch
        stopwatch.punch()
        # reset notochord state
        noto.reset()
        # reset history
        history.push()

        # TODO: feed note-ons from any held input/follower notes?

        # query the fresh notochord for a new prediction
        if pending.gate:
            auto_query()

    # @lock
    def noto_mute(sustain=False):
        tui.query_one('#mute').label = 'UNMUTE' if pending.gate else 'MUTE'
        # if sustain:
        tui.query_one('#sustain').label = 'END SUSTAIN' if pending.gate else 'SUSTAIN'

        pending.gate = not pending.gate

        if sustain:
            print('END SUSTAIN' if pending.gate else 'SUSTAIN')
        else:
            print('UNMUTE' if pending.gate else 'MUTE')
        # if unmuting, we're done
        if pending.gate:
            if sustain:
                auto_query()
            return
        # cancel pending predictions
        pending.event = None
        tui(prediction=pending.event)

        if sustain:
            return
        
        # end+feed all held notes
        for (chan,inst,pitch) in history.note_triples:
            if chan in type_chans('auto'):
                play_event(
                    dict(inst=inst, pitch=pitch, vel=0), 
                    channel=chan, tag='AUTO', memo='mute')

    # query Notochord for a new next event
    # @lock
    def auto_query(predict_input=predict_input, predict_follow=predict_follow):
        # check for stuck notes
        # and prioritize ending those
        for (_, inst, pitch), note_data in history.note_data.items():
            dur = note_data['duration'].read()
            if (
                inst in type_insts('auto') 
                and dur > max_note_len*(.1+controls.get('steer_duration', 1))
                ):
                # query for the end of a note with flexible timing
                # with profile('query', print=print):
                t = stopwatch.read()
                pending.event = noto.query(
                    next_inst=inst, next_pitch=pitch,
                    next_vel=0, min_time=t, max_time=t+0.5)
                print(f'END STUCK NOTE {inst=},{pitch=}')
                return


        all_insts = type_insts(('auto', 'input', 'follow'))
        counts = history.inst_counts(n=n_recent, insts=all_insts)
        print(counts)

        inst_types = ['auto']
        if predict_follow:
            inst_types.append('follow')
        if predict_input:
            inst_types.append('input')
        allowed_insts = type_insts(inst_types)

        held_notes = history.held_inst_pitch_map(all_insts)
        print(f'{held_notes=}')

        steer_time = 1-controls.get('steer_rate', 0.5)
        steer_pitch = controls.get('steer_pitch', 0.5)
        steer_density = controls.get('steer_density', 0.5)
        
        tqt = (max(0,steer_time-0.5), min(1, steer_time+0.5))
        tqp = (max(0,steer_pitch-0.5), min(1, steer_pitch+0.5))

        # if using nominal time,
        # *subtract* estimated feed latency to min_time; (TODO: really should
        #   set no min time when querying, use stopwatch when re-querying...)
        # if using actual time, *add* estimated query latency
        time_offset = -5e-3 if nominal_time else 10e-3
        min_time = stopwatch.read()+time_offset

        # balance_sample: note-ons only from instruments which have played less
        bal_insts = allowed_insts & set(counts.index[counts <= counts.min()+n_margin])
        if balance_sample and len(bal_insts)>0:
            allowed_insts = bal_insts

        # VTIP is better for time interventions,
        # VIPT is better for instrument interventions
        # could decide probabilistically based on value of controls + insts...
        if allowed_insts==all_insts:
            query_method = noto.query_vtip
        else:
            query_method = noto.query_vipt

        # print(f'considering {insts} for note_on')
        # use only currently selected instruments
        inst_pitch_map = inst_ranges(allowed_insts)
        note_on_map = {
            i: set(inst_pitch_map[i])-set(held_notes[i]) # exclude held notes
            for i in allowed_insts
        }
        # use any instruments which are currently holding notes
        note_off_map = {
            i: set(ps)&set(held_notes[i]) # only held notes
            for i,ps in inst_pitch_map.items()
        }

        max_t = None if max_time is None else max(max_time, min_time+0.2)

        try:
            pending.event = query_method(
                note_on_map, note_off_map,
                min_time=min_time, max_time=max_t,
                truncate_quantile_time=tqt,
                truncate_quantile_pitch=tqp,
                steer_density=steer_density,
            )
        except Exception:
            print(f'ERROR: query failed. {allowed_insts=} {note_on_map=} {note_off_map=}')

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
        raise NotImplementedError

    @midi.handle(type='pitchwheel')
    def _(msg):
        controls['steer_pitch'] = (msg.pitch+8192)/16384
        # print(controls)

    # very basic CC handling for controls
    @midi.handle(type='control_change')
    def _(msg):
        """CC messages on any channel"""

        if msg.control==1:
            controls['steer_pitch'] = msg.value/127
            print(f"{controls['steer_pitch']=}")
        if msg.control==2:
            controls['steer_density'] = msg.value/127
            print(f"{controls['steer_density']=}")
        if msg.control==3:
            controls['steer_rate'] = msg.value/127
            print(f"{controls['steer_rate']=}")

        if msg.control==4:
            noto_reset()
        if msg.control==5:
            auto_query()
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
                auto_query()
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
        # convert from 0-index
        channel = msg.channel+1

        if thru and channel in type_chans('input'):
            midi.send(msg)

        if channel not in type_chans('input'):
            print(f'WARNING: ignoring MIDI {msg} on non-input channel')
            return
        
        inst = channel_inst(channel)
        pitch = msg.note
        vel = msg.velocity if msg.type=='note_on' else 0

        # feed event to Notochord
        # with profile('feed', print=print):
        play_event(
            {'inst':inst, 'pitch':pitch, 'vel':vel}, 
            channel=channel, send=thru, tag='PLAYER')

        # query for new prediction
        auto_query()

        # send a MIDI reply for latency testing purposes:
        # if testing: midi.cc(control=3, value=msg.note, channel=15)

    def auto_event():
        # 'auto' event happens:
        event = pending.event
        inst, pitch, vel = event['inst'], event['pitch'], round(event['vel'])

        # note on which is already playing or note off which is not
        if (vel>0) == ((inst, pitch) in history.note_pairs): 
            print(f're-query for invalid {vel=}, {inst=}, {pitch=}')
            auto_query()
            return
        
        chan = auto_inst_channel(inst)
        play_event(event, channel=chan, tag='NOTO')

    @repeat(1e-3, lock=True)
    def _():
        """Loop, checking if predicted next event happens"""
        # check if current prediction has passed
        dt = pending.event['time'] if pending.event is not None else float('inf')
        if (
            not testing and
            pending.gate and
            stopwatch.read() > dt
            ):
            # if so, check if it is a notochord-controlled instrument
            if pending.event['inst'] in type_insts('auto'):
                # prediction happens
                auto_event()
            # query for new prediction
            if dt < noto.max_dt and not debug_query:
                auto_query()

    @cleanup
    def _():
        """end any remaining notes"""
        # print(f'cleanup: {notes=}')
        for (chan,inst,pitch) in history.note_triples:
        # for (inst,pitch) in notes:
            if inst in type_insts(('auto', 'follow')):
                midi.note_on(note=pitch, velocity=0, channel=chan-1)

    @tui.set_action
    def mute():
        noto_mute()

    @tui.set_action
    def sustain():
        noto_mute(sustain=True)
    
    @tui.set_action
    def reset():
        noto_reset()
    
    @tui.set_action
    def query():
        auto_query()

    if initial_query:
        auto_query(predict_input=False, predict_follow=False)

    if use_tui:
        tui.run()


if __name__=='__main__':
    run(main)
