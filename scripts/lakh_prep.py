from pathlib import Path
import itertools as it

from tqdm import tqdm
import fire
import mido
import torch

def main(data_path, dest_path, min_len=64):
    data_dir = Path(data_path)
    files = list(data_dir.glob('**/*.mid'))
    # files = files[:1000]

    for f in tqdm(files):
        try:
            mid = mido.MidiFile(f)
        except Exception:
            continue
        
        g = (Path(dest_path) / f.relative_to(data_path)).with_suffix('.pkl')
        g.parent.mkdir(parents=True, exist_ok=True)

        tempos = {m.tempo for tr in mid.tracks for m in tr if m.type=='set_tempo'}
        if len(tempos)!=1:
            continue
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
            ), g)

if __name__=="__main__":
    fire.Fire(main)
    