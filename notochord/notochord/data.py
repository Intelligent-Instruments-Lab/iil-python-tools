from pathlib import Path
import random

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, data_dir, batch_len, 
        transpose=5, speed=0.1, glob='**/*.pkl', test_len=2048,
        onsets_only=False):
        """
        """
        super().__init__()
        dirs = data_dir.split(',')
        self.files = []
        for d in dirs:
            self.files.extend(list(Path(d).glob(glob)))
        self.batch_len = batch_len
        self.transpose = transpose
        self.speed = speed
        self.start_token = 128
        self.n_anon = 32 # this needs to match n_instruments in model.py
        self.prog_start_token = 0
        self.testing = False
        self.max_test_len = test_len
        self.onsets_only = onsets_only
        
    def __len__(self):
        return len(self.files)

    def is_melodic(self, program):
        orig_program = program%1000
        return (orig_program<=128) | (orig_program>256)

    def is_anon(self, program):
        return program > 256
    
    def _remap_anonymous_instruments(self, program: torch.Tensor) -> torch.Tensor:
        """
        Randomly map instruments to additional ‘anonymous’ melodic and drum identities
        with a probability of 10% per instrument, without replacement. 
        Also map any parts > 256 to appropriate anonymous ids.
        """
        is_melodic = self.is_melodic(program)
        is_anon = self.is_anon(program)
        named_melodic = list(program.masked_select(is_melodic & ~is_anon).unique())
        anon_melodic = list(program.masked_select(is_melodic & is_anon).unique())
        named_drum = list(program.masked_select(~is_melodic & ~is_anon).unique())
        anon_drum = list(program.masked_select(~is_melodic & is_anon).unique())

        anon_melodic_start = 257
        anon_drum_start = anon_melodic_start + self.n_anon
        perm_anon_melodic = torch.randperm(self.n_anon) + anon_melodic_start 
        perm_anon_drum = torch.randperm(self.n_anon) + anon_drum_start 

        for pr in named_melodic:
            if torch.rand((1,)) < 0.1:
                anon_melodic.append(pr)
        for pr in named_drum:
            if torch.rand((1,)) < 0.1:
                anon_drum.append(pr)

        new_program = program.clone()

        if len(anon_melodic)>self.n_anon:
            print(f'warning: {len(anon_melodic)} > {self.n_anon} anon melodic instruments')
        if len(anon_drum)>self.n_anon:
            print(f'warning: {len(anon_drum)} > {self.n_anon} anon drum instruments')

        i = 0
        for pr in anon_melodic:
            new_program[program==pr] = perm_anon_melodic[i%self.n_anon]
            i += 1
        i = 0
        for pr in anon_drum:
            new_program[program==pr] = perm_anon_drum[i%self.n_anon]
            i += 1

        # print(new_program.unique())

        return new_program

    def __getitem__(self, idx):
        f = self.files[idx]
        item = torch.load(f)
        program = item['program'] # 1-d LongTensor of MIDI programs
        # 0 is unused
        # (128-256 are drums)
        # 257+ are 'true anonymous' (no program change on track)
        # (drums with no PC are just mapped to 129)
        # N + 1000*K is the Kth additional part for instrument N
        pitch = item['pitch'] # 1-d LongTensor of MIDI pitches 0-127
        time = item['time'] # 1-d DoubleTensor of absolute times in seconds
        velocity = item['velocity'] # 1-d LongTensor of MIDI velocities 0-127

        assert len(pitch) == len(time)

        if self.onsets_only:
            b = velocity > 0
            program, pitch, time, velocity = (
                program[b], pitch[b], time[b], velocity[b])
            
        program, pitch, time, velocity = self.data_augmentation(
            program, pitch, time, velocity)

        # sort (using argsort on time and indexing the rest)
        # compute delta time
        time, idx = time.sort()
        time = torch.cat((time.new_zeros((1,)), time)).diff(1).float()
        velocity = velocity[idx]
        program = program[idx]
        pitch = pitch[idx]

        # pad with start tokens, zeros
        # always pad with batch_len so that end tokens don't appear in a biased
        # location
        pad = 0 if self.testing else self.batch_len-1#max(0, self.batch_len-len(pitch))
        program = torch.cat((
            program.new_full((1,), self.prog_start_token),
            program,
            program.new_zeros((pad,))))
        pitch = torch.cat((
            pitch.new_full((1,), self.start_token),
            pitch,
            pitch.new_zeros((pad,))))
        time = torch.cat((
            time.new_zeros((1,)),
            time,
            time.new_zeros((pad,))))
        velocity = torch.cat((
            velocity.new_zeros((1,)),
            velocity,
            velocity.new_zeros((pad,))))
        # end signal: nonzero for last event
        end = torch.zeros_like(program)
        end[-pad-1:] = 1
        # compute binary mask for the loss
        mask = torch.ones_like(program, dtype=torch.bool)
        if pad > 0:
            mask[-pad:] = False

        if self.testing:
            sl = slice(0, self.max_test_len)
        else:
            # random slice
            i = random.randint(0, len(pitch)-self.batch_len)
            sl = slice(i, i+self.batch_len)
        program = program[sl]
        pitch = pitch[sl]
        time = time[sl]
        velocity = velocity[sl]
        end = end[sl]
        mask = mask[sl]

        return {
            'mask':mask,
            'end':end,
            'instrument':program,
            'pitch':pitch,
            'time':time,
            'velocity':velocity
        }
    
    def velocity_dequantize(self, velocity):
        velocity = velocity.float()
        velocity = (
            velocity + 
            (torch.rand_like(velocity, dtype=torch.float)-0.5) * ((velocity>0) & (velocity<127)).float()
            ).clamp(0., 127.)
        return velocity

    def velocity_curve(self, velocity):
        # take care not to map any positive values closer to 0 than 1
        to_curve = (velocity >= 0.5)
        velocity[to_curve] -= 0.5
        velocity[to_curve] /= 126.5
        velocity[to_curve] = velocity[to_curve] ** (2**(torch.randn((1,))/3))
        velocity[to_curve] *= 126.5
        velocity[to_curve] += 0.5
        return velocity
    
    def data_augmentation(self, program, pitch, time, velocity):
        """override this in subclass for different data augmentation"""
        # random transpose avoiding out of range notes
        transpose_down = min(self.transpose, pitch.min().item())
        transpose_up = min(self.transpose, 127-pitch.max())
        transpose = (
            random.randint(-transpose_down, transpose_up)
            * self.is_melodic(program).long() # don't transpose drums
        )
        pitch = pitch + transpose

        # scramble anonymous and extra parts to 'anonymous melodic' and 'anonymous drum' parts
        program = self._remap_anonymous_instruments(program)

        time_margin = 1e-3

        # dequantize: add noise up to +/- margin
        # move note-ons later, note-offs earlier
        time = (time + 
            torch.rand_like(time) * ((velocity==0).double()*2-1) * time_margin
        )
        # random augment tempo
        time = time * (1 + random.random()*self.speed*2 - self.speed)

        velocity = self.velocity_dequantize(velocity)
        velocity = self.velocity_curve(velocity)

        return program, pitch, time, velocity
    
class TxalaDataset(MIDIDataset):
    def __init__(self, *a, 
        onsets_only=True, permute=False, offset=True, remap=False,
        **kw):
        super().__init__(*a, onsets_only=onsets_only, **kw)
        self.remap = remap
        self.offset = offset
        self.permute = permute

    def data_augmentation(self, program, pitch, time, velocity):
        # map program, pitch pairs to txala
        txala_progs = [290, 291, 292, 293]
        txala_pits = [41, 43, 45]
        if self.remap:
            pairs = set(zip(
                [t.item() for t in program], [t.item() for t in pitch]))
            # NOTE: this will mangle authentic appearances of 
            # the txala pairs in the original data...
            # i.e. anonymized drums playing the low toms
            # effect should be small
            # but could instead compute every mask before doing replacements
            for prog, pit in pairs:
                new_prog = random.choice(txala_progs)
                new_pit = random.choice(txala_pits)
                # print(prog, pit, new_prog, new_pit)
                b = (program==prog) & (pitch==pit)
                program[b] = new_prog
                pitch[b] = new_pit
        elif self.offset:
            # for karl's prep using 265-268
            program = program + 25

        if self.permute:
            # permute the pitches
            masks = [pitch==p for p in txala_pits]
            random.shuffle(txala_pits)
            for b,p in zip(masks, txala_pits):
                pitch[b] = p

            # permute hands+players without exchanging hands
            masks = [program==p for p in txala_progs]
            p1 = txala_progs[:2]
            p2 = txala_progs[2:]
            if random.getrandbits(1):
                p1 = list(reversed(p1))
            if random.getrandbits(1):
                p2 = list(reversed(p2))
            if random.getrandbits(1):
                txala_progs = p1 + p2
            else:
                txala_progs = p2 + p1
            for b,p in zip(masks, txala_progs):
                program[b] = p

        time_margin = 1e-3

        # dequantize: add noise up to +/- margin
        # move note-ons later, note-offs earlier
        time = (time + 
            torch.rand_like(time) * ((velocity==0).double()*2-1) * time_margin
        )
        # random augment tempo
        # for txalaparta: since these are expert performances,
        # make them just a little bit faster but substantially slower
        # to accomodate novices
        time = time * (1 + random.random()*self.speed*4 - self.speed)

        velocity = self.velocity_dequantize(velocity)
        # keep within the 80,120 velocity range
        velocity = (velocity - 79) * 128 / 40
        velocity = self.velocity_curve(velocity)
        velocity = (velocity * 40 / 128) + 79

        return program, pitch, time, velocity