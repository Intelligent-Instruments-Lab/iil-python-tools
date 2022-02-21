from pathlib import Path
from multiprocessing import Pool
import functools as ft

from tqdm import tqdm
import fire
import mido
import torch

def process(fnames, min_len=64):
    f,g = fnames

    try:
        mid = mido.MidiFile(f)
    except Exception:
        return

    tempos = {m.tempo for tr in mid.tracks for m in tr if m.type=='set_tempo'}
    if len(tempos)!=1:
        return
    micros_per_beat = tuple(tempos)[0]

    s_per_tick = micros_per_beat / mid.ticks_per_beat / 1e6

    for i,tr in enumerate(mid.tracks):
        seq = [m for m in tr if m.type=='note_on' and m.velocity]
        if len(seq) < min_len:
            continue
        torch.save(dict(
            pitch=torch.LongTensor([m.note for m in seq]),
            velocity=torch.LongTensor([m.velocity for m in seq]),
            # ticks=torch.LongTensor([m.time for m in seq]),
            time=torch.Tensor([m.time for m in seq])*s_per_tick,
            # src_file=fname,
            src_track=i,
            # tempo=micros_per_beat,
        ), g.with_suffix(f'.{i}.pkl') )

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
        # p.imap_unordered(ft.partial(process, data_path, dest_path), tqdm(files), 32)
        for _ in tqdm(pool.imap_unordered(process, zip(files, files_out), 32)):
            pass
       

if __name__=="__main__":
    fire.Fire(main)
    