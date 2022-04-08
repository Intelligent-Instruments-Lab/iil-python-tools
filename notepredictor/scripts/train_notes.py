from pathlib import Path
import random
from collections import defaultdict
from collections.abc import Mapping
import itertools as it

from tqdm import tqdm
import fire

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from notepredictor import NotePredictor, MIDIDataset
from notepredictor.util import get_class_defaults

class Trainer:
    def __init__(self, 
        experiment, # experiment name
        model_dir,
        log_dir,
        data_dir,
        model = None, # dict of model constructor overrides
        # clamp_time = (0,10), # given to trainer because it needs to go to dataset+model
        batch_size = 128,
        batch_len = 64,
        lr = 3e-4,
        adam_betas = (0.9, 0.999),
        adam_eps = 1e-08, 
        weight_decay = 0.01,
        grad_clip = 1.0,
        seed = 0, # random seed
        n_jobs = 1, # for dataloaders
        device = 'cpu', # 'cuda:0'
        epoch_size = None, # in iterations, None for whole dataset
        ):
        """TODO: Trainer __init__ docstring"""
        kw = locals(); kw.pop('self')

        # store all hyperparams for checkpointing
        self.kw = kw

        # get model defaults from model class
        model_cls = NotePredictor
        if model is None: model = {}
        assert isinstance(model, dict), """
            model keywords are not a dict. check shell/fire syntax
            """
        kw['model'] = model = get_class_defaults(model_cls) | model
        model['num_pitches'] = 128
        model['num_instruments'] = 272
        # model['time_bounds'] = clamp_time

        # assign all arguments to self by default
        self.__dict__.update(kw)
        # mutate some arguments:
        self.model_dir = Path(model_dir) / self.experiment
        self.log_dir = Path(log_dir) / self.experiment
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)

        # filesystem
        for d in (self.model_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

        # random states
        self.seed_random()

        # logging
        self.writer = SummaryWriter(self.log_dir)

        # Trainer state
        self.iteration = 0
        self.exposure = 0
        self.epoch = 0

        # construct model from arguments 
        self.model = model_cls(**model).to(self.device)
        tqdm.write(repr(self.model))

        # dataset
        self.dataset = MIDIDataset(
            self.data_dir, self.batch_len)#, clamp_time=clamp_time)
        valid_len = int(len(self.dataset)*0.05)
        train_len = len(self.dataset) - valid_len
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
            self.dataset, [train_len, valid_len], 
            generator=torch.Generator().manual_seed(0))

        self.opt = torch.optim.AdamW(
            self.model.parameters(), 
            self.lr, self.adam_betas, self.adam_eps, self.weight_decay)

    @property
    def gpu(self):
        return self.device.type!='cpu'

    def seed_random(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def set_random_state(self, states):
        # note: GPU rng state not handled
        std_state, np_state, torch_state = states
        random.setstate(std_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)

    def save(self, fname):
        torch.save(dict(
            kw=self.kw,
            model_state=self.model.state_dict(),
            optimizer_state=self.opt.state_dict(),
            step=(self.exposure, self.iteration, self.epoch),
            random_state=(random.getstate(), np.random.get_state(), torch.get_rng_state())
        ), fname)

    def load_state(self, d):
        d = d if hasattr(d, '__getitem__') else torch.load(d)
        self.model.load_state_dict(d['model_state'])
        # note that this loads optimizer lr, beta etc
        # from stored, even if different values given to constructor
        self.opt.load_state_dict(d['optimizer_state'])
        self.exposure, self.iteration, self.epoch = d['step']
        self.set_random_state(d['random_state'])

    def log(self, tag, d):
        # self.writer.add_scalars(tag, d, self.exposure)
        for k,v in d.items():
            self.writer.add_scalar(f'{tag}/{k}', v, self.exposure)
    
    def process_grad(self):
        r = {}
        if self.grad_clip is not None:
            r['grad_l2'] = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
        return r

    def get_loss_components(self, result):
        return {
            'instrument_nll': -result['instrument_log_probs'].mean(),
            'pitch_nll': -result['pitch_log_probs'].mean(),
            'time_nll': -result['time_log_probs'].mean(),
            'velocity_nll': -result['velocity_log_probs'].mean(),
            'end_nll': -result['end_log_probs'].mean()
        }

    def train(self):
        """TODO: train docstring"""
        self.save(self.model_dir / f'{self.epoch:04d}.ckpt')

        train_loader = DataLoader(
            self.train_dataset, self.batch_size,
            shuffle=True, num_workers=self.n_jobs, pin_memory=self.gpu)
        valid_loader = DataLoader(
            self.valid_dataset, self.batch_size,
            shuffle=False, num_workers=self.n_jobs, pin_memory=self.gpu)

        ##### validation loop
        def validate():
            metrics = defaultdict(float)
            self.model.eval()
            for batch in tqdm(valid_loader, desc=f'validating epoch {self.epoch}'):
                end = batch['end'].to(self.device, non_blocking=True)
                inst = batch['instrument'].to(self.device, non_blocking=True)
                pitch = batch['pitch'].to(self.device, non_blocking=True)
                time = batch['time'].to(self.device, non_blocking=True)
                vel = batch['velocity'].to(self.device, non_blocking=True)
                with torch.no_grad():
                    result = self.model(
                        inst, pitch, time, vel, end, validation=True)
                    losses = {k:v.item() for k,v in self.get_loss_components(result).items()}
                    metrics['loss'] += sum(losses.values())
                    for k,v in losses.items():
                        metrics[k] += v
                    metrics['instrument_acc'] += (result['instrument_log_probs']
                        .exp().mean().item())
                    metrics['pitch_acc'] += (result['pitch_log_probs']
                        .exp().mean().item())
                    metrics['time_acc_30ms'] += (result['time_acc_30ms']
                        .mean().item())
                    metrics['velocity_acc'] += (result['velocity_log_probs']
                        .exp().mean().item())
            self.log('valid', {k:v/len(valid_loader) for k,v in metrics.items()})

        epoch_size = self.epoch_size or len(train_loader)

        # validate at initialization
        validate()

        while True:
            self.epoch += 1

            ##### training loop
            self.model.train()
            for batch in tqdm(it.islice(train_loader, epoch_size), 
                    desc=f'training epoch {self.epoch}', total=epoch_size):

                end = batch['end'].to(self.device, non_blocking=True)
                inst = batch['instrument'].to(self.device, non_blocking=True)
                pitch = batch['pitch'].to(self.device, non_blocking=True)
                time = batch['time'].to(self.device, non_blocking=True)
                vel = batch['velocity'].to(self.device, non_blocking=True)

                self.iteration += 1
                self.exposure += self.batch_size

                logs = {}

                self.opt.zero_grad()
                result = self.model(inst, pitch, time, vel, end)
                losses = self.get_loss_components(result)
                loss = sum(losses.values())
                loss.backward()
                logs |= self.process_grad()
                self.opt.step()

                logs |= {k:v.item() for k,v in losses.items()}
                logs |= {k:v.item() for k,v in result.items() if v.numel()==1}
                logs |= {'loss':loss.item()}
                self.log('train', logs)

            validate()

            self.save(self.model_dir / f'{self.epoch:04d}.ckpt')

def deep_update(a, b):
    """
    in-place update a with contents of b, recursively for nested Mapping objects.
    """
    for k in b:
        if isinstance(a[k], Mapping) and isinstance(b[k], Mapping):
            deep_update(a[k], b[k])
        else:
            a[k] = b[k]

class Resumable:
    def __init__(self, checkpoint=None, **kw):
        if checkpoint is not None:
            d = torch.load(checkpoint)
            # merges sub dicts, e.g. model hyperparameters
            deep_update(d['kw'], kw)
            self._trainer = Trainer(**d['kw'])
            self._trainer.load_state(d)
        else:
            self._trainer = Trainer(**kw)

    def train(self):
        self._trainer.train()

Resumable.__doc__ = Trainer.__init__.__doc__
Resumable.train.__doc__ = Trainer.train.__doc__

if __name__=='__main__':
    # TODO: improve fire-generated help message
    fire.Fire(Resumable)