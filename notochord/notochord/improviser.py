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

def main(
        player_config:Dict[int,int]=None, # map MIDI channel : instrument
        noto_config:Dict[int,int]=None, # map MIDI channel : instrument
        max_note_len=5, # in seconds, to auto-release stuck Notochord notes
        midi_in=None, # MIDI port for player input
        midi_out=None, # MIDI port for Notochord output
        checkpoint="artifacts/notochord-latest.ckpt" # Notochord checkpoint
        ):
    midi = MIDI(midi_in, midi_out)

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
        print('prediction:', pending.event)


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

        print('PLAYER:', msg)

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
                print('REPLACING BAD PREDICTION', pending.event)
                query()
                return

        # send as MIDI
        midi.note_on(note=pitch, velocity=int(vel), channel=noto_map.inv(inst))
        # feed back to Notochord
        noto.feed(inst, pitch, timer.punch(), vel)
        # track total events played
        noto_events.plus()
        # print
        print('NOTOCHORD:', event)

    @repeat(5e-4, lock=True)
    def _():
        """Loop, checking if predicted next event happens"""
        # check if current prediction has passed
        if pending.event is not None and timer.read() > pending.event['time']:
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



if __name__=='__main__':
    run(main)
