## WIP realtime audio generation from notochord checkpoint

from notochord import Notochord

import fire
import fluidsynth
from mido import second2tick
from multiprocessing import Process, Queue
import time


def sequence_event(event: dict) -> None:
    """
    :param nc_notes: list of note events returned from querying notochord
    """
    ticks_per_beat = 1000
    tempo = 120

    seq = fluidsynth.Sequencer(time_scale=ticks_per_beat)
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload("example.sf2")
    fs.program_select(0, sfid, 0, 0)
    synthID = seq.register_fluidsynth(fs)

    try:
        while True:
            if q.empty():
                time.sleep(0.0005)
                continue

            r = q.get()
            inst, pitch, nctime, velocity, end, step = r['instrument'], r['pitch'], r['time'], r['velocity'], r['end'], r['step']

            # process notochord events for MIDI
            # program = inst - 1  # instruments in nc are shifted by 1 to allow for start token
            # TODO: convert inst to prog/ channel
            # TODO: ignore start token 0

            # measure fluidsynth delay - bonus points for latency to actual sound production
            # measure notochord latency
            # build in 100ms buffer like tidal?

            # first- dumb version - time.sleep nctime
            # second version- compensate for drift
            # third version- ?

            note_off = velocity == 0
            velocity = round(velocity)  # discretize velocity
            # can increase the time resolution by increasing MidiFile.ticks_per_beat- typical range is [96-480] but can go higher
            midi_time = round(second2tick(nctime, ticks_per_beat, tempo))

            # TODO: handle drum tracks - should be mapped to channel 10 in 1-128 or channel 9 in 0-127
            if note_off:
                seq.note_off(time=midi_time, absolute=False, channel=0, key=pitch, velocity=velocity, dest=synthID)
            else:
                seq.note_on(time=midi_time, absolute=False, channel=0, key=pitch, velocity=velocity, dest=synthID)


    except KeyboardInterrupt:
        print('sequence_event done')
    # finally:


def generate_from_notochord(q, checkpoint: str) -> None:
    predictor = Notochord.from_checkpoint(checkpoint)
    predictor.reset()  # TODO: look at reset logic
    predictor.eval()

    unique_inst = set()
    notes = []

    # optional feed initial note
    r = predictor.feed(inst=1, pitch=60, nctime=0, velocity=100, include_instrument=None)

    try:
        while True:
            # Only allow sampling up to 16 instruments- take the first 16 unique instruments
            if len(unique_inst) >= 16:
                include_instrument = list(unique_inst)
            else:
                unique_inst.add(inst)
            # use query feed - blacklist certain instruments, set temperature - absolute simplest way
            start = time.time()
            r = predictor.query_feed(include_instrument=include_instrument)
            notochord_latency = time.time() - start
            time.sleep(min(nctime - notochord_latency, 0))  # TODO: test logic for factoring in notochord_latency
            q.put(r)  # add event to queue to be played
    except KeyboardInterrupt:
        print('generate_from_notochord done')
    finally:
        predictor.reset()


def main(checkpoint: str = None) -> None:

    if checkpoint is None:
        raise ValueError(f'Checkpoint required but is None. Use --checkpoint to provide the path to the .ckpt file.')
        
    p1 = Process(target=generate_from_notochord, args=(q, checkpoint,))
    p2 = Process(target=sequence_event, args=(q,))
    p1.start()
    p2.start()


if __name__=='__main__':
    q = Queue()
    fire.Fire(main)
