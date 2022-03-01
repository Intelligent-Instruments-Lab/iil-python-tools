from pathlib import Path
import random

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

class MIDIDataset(Dataset):
    def __init__(self, data_dir, batch_len, transpose=2, glob='**/*.pkl'):
        """
        """
        super().__init__()
        self.files = list(Path(data_dir).glob(glob))
        self.batch_len = batch_len
        self.transpose = transpose
        self.start_token = 128
        self.end_token = 129
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        f = self.files[idx]
        item = torch.load(f)
        pitch = item['pitch'] # 1-d LongTensor of MIDI pitches 0-127
        time = item['time']
        assert len(pitch) == len(time)

        # random transpose avoiding out of range notes
        transpose_down = min(self.transpose, pitch.min().item())
        transpose_up = min(self.transpose, 127-pitch.max())
        transpose = random.randint(-transpose_down, transpose_up)
        pitch = pitch + transpose

        # pad with start, end tokens
        pad = max(1, self.batch_len-len(pitch))
        pitch = torch.cat((
            pitch.new_full((1,), self.start_token),
            pitch,
            pitch.new_full((pad,), self.end_token)))
        time = torch.cat((
            time.new_zeros((1,)),
            time,
            time.new_zeros((pad,))))

        # random slice
        i = random.randint(0, len(pitch)-self.batch_len)
        pitch = pitch[i:i+self.batch_len]
        time = time[i:i+self.batch_len]

        return {
            'pitch':pitch,
            'time':time
        }