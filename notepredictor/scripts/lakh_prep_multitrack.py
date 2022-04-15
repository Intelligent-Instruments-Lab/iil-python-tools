from pathlib import Path
from multiprocessing import Pool
import functools as ft
from collections import defaultdict
import random

from tqdm import tqdm
import fire
from pretty_midi import PrettyMIDI
import torch

def process(fnames):
    f,g = fnames

    try:
        mid = PrettyMIDI(str(f))
    except Exception:
        return

    # fix overlapping notes and add a margin for 
    # dequantization at data loading time
    time_margin = 1e-3

    events = []
    # for each instrument
    for inst in mid.instruments:
        inst.remove_invalid_notes()
        program = inst.program + 128*inst.is_drum
        
        # break out by pitch
        nbp = defaultdict(list)
        for n in inst.notes:
            nbp[n.pitch].append(n)
            
        # shorten all notes so they end 2*$margin before next (within pitch)
        for seq in nbp.values():
            for i,n in enumerate(seq[:-1]):
                max_end = seq[i+1].start-2*time_margin
                if n.end > max_end:
                    n.end = max_end
                # and flatten again
                # converting note offs to 0 velocity
                # also prevent any note ons from having 0 velocity
                events.append((n.start, n.pitch, max(1, n.velocity), program))
                events.append((n.end, n.pitch, 0, program))

    if len(events) < 64:
        return

    time, pitch, vel, prog = zip(*events)
    torch.save(dict(
        time=torch.FloatTensor(time), 
        pitch=torch.LongTensor(pitch), 
        velocity=torch.LongTensor(vel),
        program=torch.LongTensor(prog)
    ), g.with_suffix('.pkl'))


    #     # NOTE: this will sort concurrent events by pitch
    #     # which will introduce some bias when interacting with the model?
    #     # e.g. if user plays a note, it will never be harmonized below (or only)
    #     # with inexact timing, less frequently
    #     # similarly the pitch order would correlate with instrument, i.e. bass 
    #     # would usually play first
    #     # if anything descending pitch might sound better
    #     # could randomize -- even better would be to randomize in dataloader
    #     # might be expensive though
    #     note_ons = [(n.start, n.pitch, n.velocity, program) for n in inst.notes]
    #     note_offs = [(n.end, n.pitch, 0, program) for n in inst.notes]
    #     inst_events.extend(note_ons+note_offs)
    # if len(inst_events) < 64:
    #     return
    # time, pitch, vel, prog = zip(*sorted(inst_events))
    # delta = torch.FloatTensor([0, *time]).diff(1)
    # torch.save(dict(
    #     time=delta, 
    #     pitch=torch.LongTensor(pitch), 
    #     velocity=torch.LongTensor(vel),
    #     program=torch.LongTensor(prog)
    # ), g.with_suffix('.pkl'))

def main(data_path, dest_path, n_jobs=4):
    data_dir = Path(data_path)
    files = list(data_dir.glob('**/*.mid'))
    files_out = [
        (Path(dest_path) / f.relative_to(data_path))
        for f in files]
    parents = {g.parent for g in files_out}
    for parent in list(parents):
        parent.mkdir(parents=True, exist_ok=True)

    # files = files[:1000]

    with Pool(n_jobs) as pool:
        for _ in tqdm(pool.imap_unordered(process, zip(files, files_out), 32)):
            pass
       

if __name__=="__main__":
    fire.Fire(main)
    