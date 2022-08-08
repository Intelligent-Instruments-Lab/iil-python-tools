from pathlib import Path
import random

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, data_dir, batch_len, transpose=5, speed=0.1, glob='**/*.pkl', test_len=2048):
        #, clamp_time=(-,10)):
        """
        """
        super().__init__()
        self.files = list(Path(data_dir).glob(glob))
        self.batch_len = batch_len
        self.transpose = transpose
        self.speed = speed
        self.start_token = 128
        self.n_anon = 8
        self.prog_start_token = 0
        # self.clamp_time = clamp_time
        self.testing = False
        self.max_test_len = test_len
        
    def __len__(self):
        return len(self.files)
    
    def _random_map_anonymous_instruments(self, program: torch.Tensor) -> torch.Tensor:
        """
        Randomly map instruments to eight additional ‘anonymous’ melodic and drum identities
        with a probability of 10% per instrument, without replacement.

        The input program should contain melodic instruments from MIDI note numbers 0-127 and
        drum instruments from 128-255. Anonymous instruments are mapped to subsequent note numbers.
        """
        unique_melodic = program.masked_select(program<128).unique()
        unique_drum = program.masked_select(program>=128).unique()

        anon_melodic_start = 256
        anon_drum_start = anon_melodic_start + self.n_anon
        anon_melodic = torch.randperm(self.n_anon) + anon_melodic_start  # array of anon melodic programs
        anon_drum = torch.randperm(self.n_anon) + anon_drum_start  # array of anon drum programs

        i = 0
        for pr in unique_melodic:
            if torch.rand((1,)) < 0.1:
                program[program==pr] = anon_melodic[i]
                i += 1
                if i >= len(anon_melodic):  # no more anon instruments to write to
                    break
        i = 0
        for pr in unique_drum:
            if torch.rand((1,)) < 0.1:
                program[program==pr] = anon_drum[i]
                i += 1
                if i >= len(anon_drum):  # no more anon instruments to write to
                    break

        return program

    def __getitem__(self, idx):
        f = self.files[idx]
        item = torch.load(f)
        program = item['program'] # 1-d LongTensor of MIDI programs 0-255
        # (128-255 are drums)
        pitch = item['pitch'] # 1-d LongTensor of MIDI pitches 0-127
        time = item['time']
        velocity = item['velocity']

        assert len(pitch) == len(time)

        # random transpose avoiding out of range notes
        transpose_down = min(self.transpose, pitch.min().item())
        transpose_up = min(self.transpose, 127-pitch.max())
        transpose = (
            random.randint(-transpose_down, transpose_up)
            * (program<128) # don't transpose drums
        )
        pitch = pitch + transpose

        # randomly map instruments to 'anonymous melodic' and 'anonymous drum'
        program = self._random_map_anonymous_instruments(program)

        # shift from 0-index to general MIDI 1-index; reserve 0 for start token
        program += 1

        time_margin = 1e-3 # hardcoded since it should match prep script

        # dequantize: add noise up to +/- margin
        time = time + (torch.rand_like(time)*2-1)*time_margin
        # random augment tempo
        time = time * (1 + random.random()*self.speed*2 - self.speed)

        # dequantize velocity
        velocity = velocity.float()
        velocity = (
            velocity + 
            (torch.rand_like(time)-0.5) * ((velocity>0) & (velocity<127)).float()
            ).clamp(0., 127.)
        # random velocity curve
        # take care not to map any positive values closer to 0 than 1
        to_curve = (velocity >= 0.5)
        velocity[to_curve] -= 0.5
        velocity[to_curve] /= 126.5
        velocity[to_curve] = velocity[to_curve] ** (2**(torch.randn((1,))/3))
        velocity[to_curve] *= 126.5
        velocity[to_curve] += 0.5

        # sort (using argsort on time and indexing the rest)
        # compute delta time
        time, idx = time.sort()
        time = torch.cat((time.new_zeros((1,)), time)).diff(1)
        program = program[idx]
        pitch = pitch[idx]
        velocity = velocity[idx]

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