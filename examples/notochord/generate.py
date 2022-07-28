## WIP!! offline generate midi and audiofile from notochord

from notochord import Notochord

import fire
from typing import List
from mido import Message, MidiFile, MidiTrack, second2tick
import time


def save_midifile(nc_notes: List, filename: str='generated.midi') -> bool:
    """
    :param nc_notes: list of note events returned from querying notochord
    """
    mid = MidiFile(type=1)
    inst_track = {}

    for r in nc_notes:
        print(r)
        inst, pitch, nctime, velocity, end, step = r['instrument'], r['pitch'], r['time'], r['velocity'], r['end'], r['step']

        # process notochord events for MIDI
        program = inst - 1  # instruments in nc are shifted by 1 to allow for start token?
        velocity = round(velocity)  # discretize velocity
        # can increase the time resolution by increasing MidiFile.ticks_per_beat- typical range is [96-480] but can go higher
        midi_time = round(second2tick(nctime))  # TODO: specify ticks_per_beat, tempo

        # TODO does fluid synth handle multi tracks?
        track = inst_track.get(inst)
        if track is None:
            # add a new track for each new instrument
            track = MidiTrack()
            inst_track[inst] = track
            mid.tracks.append(track)

        # TODO: handle drum tracks - should be mapped to channel 10 in 1-128 or channel 9 in 0-127
        # have to synchronize to start at same time or try async type=2? maybe program change does this for us
        
        track.append(Message('program_change', program=inst, time=0))
        note_on_off = 'note_on' if velocity else 'note_off'
        track.append(Message(note_on_off, note=pitch, velocity=velocity, time=midi_time))

    mid.save(filename)
    return True


def main(checkpoint: str = None) -> None:

    if checkpoint is None:
        raise ValueError(f'Checkpoint required but is None. Use --checkpoint to provide the path to the .ckpt file.')
        
    predictor = Notochord.from_checkpoint(checkpoint)
    predictor.eval()

    include_instrument = None
    inst = 1  # grand piano?
    pitch = 60  # C4
    nctime = 0
    velocity = 100
    unique_inst = set()
    notes = []

    start = time.time()
    # r = predictor.feed(inst, pitch, nctime, velocity)
    while time.time() - start < 10:

        # Only allow sampling up to 16 instruments- take the first 16 unique instruments
        if len(unique_inst) >= 16:
            include_instrument = list(unique_inst)
        else:
            unique_inst.add(inst)

        # r = predictor.query()
        r = predictor.feed_query(inst, pitch, nctime, velocity, include_instrument=include_instrument)
        # print(r)
        inst, pitch, nctime, velocity, end, step = r['instrument'], r['pitch'], r['time'], r['velocity'], r['end'], r['step']
        notes.append(r)

    end = time.time()
    print(end - start)
    print((end - start) / step)
    print(unique_inst)
    print(len(unique_inst))
    print(include_instrument)
    predictor.reset()

    save_midifile(notes)


if __name__=='__main__':
    fire.Fire(main)
