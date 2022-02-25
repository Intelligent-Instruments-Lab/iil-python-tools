from pathlib import Path
import random

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

class MIDIPitchDataset(Dataset):
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
        notes = item['pitch'] # 1-d LongTensor of MIDI pitches 0-127

        # random transpose avoiding out of range notes
        transpose_down = min(self.transpose, notes.min().item())
        transpose_up = min(self.transpose, 127-notes.max())
        transpose = random.randint(-transpose_down, transpose_up)
        notes = notes + transpose

        # pad with start, end tokens
        pad = max(1, self.batch_len-len(notes))
        notes = torch.cat((
            notes.new_full((1,),self.start_token),
            notes,
            notes.new_full((pad,),self.end_token)))

        # random slice
        i = random.randint(0, len(notes)-self.batch_len)
        notes = notes[i:i+self.batch_len]

        return notes