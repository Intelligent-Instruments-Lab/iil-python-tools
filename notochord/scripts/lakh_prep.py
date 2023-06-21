from pathlib import Path
from multiprocessing import Pool
import functools as ft
import itertools as it
from collections import defaultdict

from tqdm import tqdm
import fire
import mido
import torch

def pitch_collision(p1, p2):
    return len(p1['pitches'] & p2['pitches']) > 0

def time_collision(p1, p2):
    # there is a collision unless one ends before the other starts
    if p1['bounds'][1] < p2['bounds'][0]:
        return False
    if p2['bounds'][1] < p1['bounds'][0]:
        return False
    return True
    
def note_collision(p1, p2):
    parts = {1:p1, 2:p2}
    if not pitch_collision(p1, p2):
        # print('early out: pitch')
        return False
    if not time_collision(p1, p2):
        # print('early out: time')
        return False
    events = [
        *zip(p1['events']['time'], p1['events']['vel'], p1['events']['pitch'], it.repeat(1)),
        *zip(p2['events']['time'], p2['events']['vel'], p2['events']['pitch'], it.repeat(2))]
    events.sort()
    # print(events)
    held = {1:set(), 2:set()}
    for (t,v,p,part) in events:
        if v > 0:
            held[part].add(p)
        else:
            held[part].discard(p)
        if p in held[1] and p in held[2]:
            return True
    return False
        
def repair_events(part):
    events = zip(part['events']['time'], part['events']['vel'], part['events']['pitch'])
    i = part['inst']
    new_events = []
    # held = defaultdict(int)
    held = set()
    n_double_off = 0
    n_double_on = 0
    for (t,v,p) in events:
        if v > 0:
            if p in held:
                # add note-offs for extra note-ons
                n_double_on += 1
                # print(f'double note on: part {part["part"]} pitch {p} time {t}')
                new_events.append((t,0,p,i))
            new_events.append((t,v,p,i))
            held.add(p)
        else:
            if p in held:
                new_events.append((t,v,p,i))
                held.remove(p)
            else:
                # delete extra note-offs
                n_double_off += 1
                # print(f'double note off: part {part["part"]} pitch {p} time {t}')
    # if n_double_off:
    #     print(f'double note off: part {part["part"]} count {n_double_off}')
    # if n_double_on:
    #     print(f'double note on: part {part["part"]} count {n_double_on}')
    new_events.sort()
    return new_events

# number of channels with no program change
class AnonTracks:
    def __init__(self):
        self.n = 0
    def __call__(self):
        self.n += 1
        return 256+self.n

def process(fnames):
    f,g = fnames

    # fix overlapping notes and add a margin for 
    # dequantization at data loading time
    # time_margin = 1e-3
 
    try:
        mid = mido.MidiFile(f)
    except Exception:
        tqdm.write(f'error opening {f}')
        return

    if mid.type==2:
        tqdm.write(f'type 2 file {f}')
        return
           
    release_velocity_counts = defaultdict(int)
    channel_counts = defaultdict(int)
    note_ons = 0
    pseudo_note_offs = 0
    note_offs = 0
    
    tempo_changes = 0
        
    parts = defaultdict(lambda: defaultdict(list))
    
    ticks_per_beat = mid.ticks_per_beat
    
    next_anon = AnonTracks()

    # apparently delta times are within-track,
    # but tempo changes affect all tracks?
    # 500_000 is default us/beat
    tick_tempos = [(0, 500_000)]
    for track_idx, track in enumerate(mid.tracks):
        channel_instruments = defaultdict(next_anon)
        time_ticks = 0
        
        for msg in track:
            # time_seconds += mido.tick2second(msg.time, ticks_per_beat, tempo)
            time_ticks += msg.time
                        
            if msg.type=='program_change':
                channel_instruments[msg.channel] = msg.program + 1 + 128*int(msg.channel==9)
                
            elif msg.type=='set_tempo':
                tick_tempos.append((time_ticks, msg.tempo))
                # print(f'tempo: {msg.tempo} at tick {time_ticks}')
                tempo_changes += 1
                
            elif msg.type in ('note_on', 'note_off'):

                # triple of track number, channel number, current instrument
                if msg.channel not in channel_instruments and msg.channel==9:
                    channel_instruments[msg.channel] = 129
                part = (track_idx, msg.channel, channel_instruments[msg.channel])
                pitch = msg.note
                
                if msg.type=='note_on':
                    vel = msg.velocity
                    if msg.velocity==0:
                        pseudo_note_offs += 1
                    else:
                        note_ons += 1
                else:
                    vel = 0
                    note_offs += 1
                    release_velocity_counts[msg.velocity] += 1
                    
                channel_counts[msg.channel] += 1
                
                # absolute time, pitch, vel=0 for note-off
                parts[part]['pitch'].append(pitch)
                parts[part]['time'].append(time_ticks)#time_seconds)
                parts[part]['vel'].append(vel)
         
            else:
                continue
                
    # abs ticks -> abs seconds:
    # find tempo below ticks
    # abs seconds = convert(ticks-last_tempo_change_ticks) + abs_seconds(last_tempo_change_ticks)
    @ft.lru_cache(4096)
    def abs_ticks_to_seconds(ticks):
        if ticks==0: return 0
        last_change_ticks, tempo = next(filter(
            lambda x: x[0]<ticks, 
            reversed(tick_tempos)))
        return (
            mido.tick2second(ticks-last_change_ticks, ticks_per_beat, tempo) 
            + abs_ticks_to_seconds(last_change_ticks))
    
    try:
        for part, events in parts.items():
            events['time'] = [abs_ticks_to_seconds(t) for t in events['time']]
    except RecursionError:
        tqdm.write(f'too many tempo changes in {f}')

    by_inst = defaultdict(list)
    for (track, channel, inst), events in parts.items():
        unique_pitch = set(events['pitch'])
        time_bounds = (min(events['time']), max(events['time']))
        # print(part, unique_pitch)
        by_inst[inst].append({
            'size':len(events['pitch']),
            'part':(track, channel),
            'pitches':unique_pitch,
            'bounds':time_bounds,
            'events':events
        })
    
    complete_parts = []
    # within each instrument, 
    for inst, parts in by_inst.items():
        assigned_parts = []
        # sort parts by number of events
        parts.sort(key=lambda e: e['size'])
        
        n_parts = 0
        
        # pop off parts and check for collisions
        while len(parts):
            no_collide = None
            part = parts.pop()
            # try for each assigned part:
            for ap in assigned_parts:
                # check for collision
                # (can speed this up by checking for pitch and bounding time collisions first)
                if not note_collision(part, ap):
                    # print('no collision')
                    no_collide = ap
                    break
                # else:
                    # print('collision')
            # if all collide, assign the part to a new anonymous instrument
            if no_collide is None:
                part['inst'] = n_parts*1000 + inst
                # print(f'assigned instrument {part["inst"]} for {part["part"]}')
                n_parts += 1
                assigned_parts.append(part)
            # otherwise merge with non-colliding part
            else:
                # print(f'merge {inst} {ap["part"]} < {part["part"]}')
                for k in ('pitch', 'time', 'vel'):
                    ap['events'][k].extend(part['events'][k])
        
        for p in assigned_parts:
            complete_parts.append(repair_events(p))

    events = sum(complete_parts, start=[])

    if len(events) < 64:
        return

    events.sort()

    time, vel, pitch, prog = zip(*events)
    torch.save(dict(
        time=torch.DoubleTensor(time), 
        pitch=torch.LongTensor(pitch), 
        velocity=torch.LongTensor(vel),
        program=torch.LongTensor(prog)
    ), g.with_suffix('.pkl'))


def main(data_path, dest_path, n_jobs=4, n_files=None):
    data_dir = Path(data_path)
    files = list(data_dir.glob('**/*.mid'))
    files_out = [
        (Path(dest_path) / f.relative_to(data_path))
        for f in files]
    parents = {g.parent for g in files_out}
    for parent in list(parents):
        parent.mkdir(parents=True, exist_ok=True)

    if n_files is not None:
        files = files[:n_files]

    with Pool(n_jobs) as pool:
        for _ in tqdm(pool.imap_unordered(process, zip(files, files_out), 32)):
            pass
       

if __name__=="__main__":
    fire.Fire(main)
    