import os
# fix torch device order to be same as nvidia-smi order
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

from pathlib import Path
import random
from collections import defaultdict
import itertools as it

import pdb

from tqdm import tqdm
import fire

import numpy as np
import scipy.stats

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from notochord import Notochord, MIDIDataset, TxalaDataset
from notochord.util import deep_update, get_class_defaults, gen_masks

class Trainer:
    def __init__(self, 
        experiment, # experiment name
        model_dir,
        log_dir,
        data_dir,
        results_dir,
        model = None, # dict of model constructor overrides
        batch_size = 128,
        batch_len = 64,
        batch_len_schedule = None,
        batch_len_max = 512,
        lr = 3e-4,
        adam_betas = (0.9, 0.999),
        adam_eps = 1e-08, 
        weight_decay = 0.01,
        grad_clip = 1.0,
        seed = 0, # random seed
        n_jobs = 1, # for dataloaders
        device = 'cpu', # 'cuda:0'
        epoch_size = None, # in iterations, None for whole dataset
        txala = False,
        txala_remap = False,
        txala_permute = False,
        min_valid = 8,
        min_test = 8
        ):
        """TODO: Trainer __init__ docstring"""
        kw = locals(); kw.pop('self')

        # store all hyperparams for checkpointing
        self.kw = kw

        # get model defaults from model class
        model_cls = Notochord
        if model is None: model = {}
        assert isinstance(model, dict), """
            model keywords are not a dict. check shell/fire syntax
            """
        kw['model'] = model = get_class_defaults(model_cls) | model
        model['num_pitches'] = 128
        model['num_instruments'] = 320
        # model['time_bounds'] = clamp_time

        # assign all arguments to self by default
        self.__dict__.update(kw)
        # mutate some arguments:
        self.model_dir = Path(model_dir) / self.experiment
        self.log_dir = Path(log_dir) / self.experiment
        self.results_dir = Path(results_dir) / self.experiment
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)

        # filesystem
        for d in (self.model_dir, self.log_dir, self.results_dir):
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
        if txala:
            self.dataset = TxalaDataset(data_dir, self.batch_len, 
                remap=txala_remap, permute=txala_permute)
        else:
            self.dataset = MIDIDataset(data_dir, self.batch_len)
            
        valid_len = max(min_valid, int(len(self.dataset)*0.03))
        test_len = max(min_test, int(len(self.dataset)*0.02))
        train_len = len(self.dataset) - valid_len - test_len
        self.train_dataset, self.valid_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_len, valid_len, test_len], 
            generator=torch.Generator().manual_seed(0))

        # params = {k:v for k,v in self.model.named_parameters()}
        # ks = ['projections.3.net.1.weight', 'projections.2.net.1.weight']
        # slow_params = {k:params.pop(k) for k in ks}
        # self.opt = torch.optim.AdamW([
        #     {'params':params.values()},
        #     {'params':slow_params.values(), 'lr':self.lr*1e-1}], 
        #     self.lr, self.adam_betas, self.adam_eps, self.weight_decay)
        self.opt = torch.optim.AdamW(self.model.parameters(),
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

    def load_state(self, d, resume):
        d = d if hasattr(d, '__getitem__') else torch.load(d)
        self.model.load_state_dict(d['model_state'], strict=resume)
        if resume:
            print('loading optimizer state, RNG state, step counts')
            print("""
            warning: optimizer lr, beta etc are restored with optimizer state,
            even if different values given on the command line, when resume=True
            """)
            self.opt.load_state_dict(d['optimizer_state'])
            self.exposure, self.iteration, self.epoch = d['step']
            self.set_random_state(d['random_state'])
        else:
            print('fresh run transferring only model weights')

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

    def get_loss_components(self, result, mask):
        def reduce(k):
            return result[k].masked_select(mask).mean()
        return {
            'instrument_nll': -reduce('instrument_log_probs'),
            'pitch_nll': -reduce('pitch_log_probs'),
            'time_nll': -reduce('time_log_probs'),
            'velocity_nll': -reduce('velocity_log_probs'),
            'end_nll': -reduce('end_log_probs'),
        }

    def _validate(self, valid_loader, ar_mask=None, testing=False):
        """"""
        pops = defaultdict(list)
        self.model.eval()
        if testing:
            self.dataset.testing = True
        for batch in tqdm(valid_loader, desc=f'validating epoch {self.epoch}'):
            # print(batch['mask'].shape, batch['mask'].sum())
            mask = batch['mask'].to(self.device, non_blocking=True)[...,1:]
            end = batch['end'].to(self.device, non_blocking=True)
            inst = batch['instrument'].to(self.device, non_blocking=True)
            pitch = batch['pitch'].to(self.device, non_blocking=True)
            time = batch['time'].to(self.device, non_blocking=True)
            vel = batch['velocity'].to(self.device, non_blocking=True)
            with torch.no_grad():
                result = self.model(
                    inst, pitch, time, vel, end, 
                    validation=True, ar_mask=ar_mask)
                losses = {k:v.item() for k,v in self.get_loss_components(
                    result, mask).items()}
                for k,v in losses.items():
                    pops[k].append(v)
                pops['loss'].append(sum(losses.values()))
                pops['instrument_acc'].append(result['instrument_log_probs']
                    .masked_select(mask).exp().mean().item())
                pops['pitch_acc'].append(result['pitch_log_probs']
                    .masked_select(mask).exp().mean().item())
                pops['time_acc_30ms'].append(result['time_acc_30ms']
                    .masked_select(mask).mean().item())
                pops['velocity_acc'].append(result['velocity_log_probs']
                    .masked_select(mask).exp().mean().item())
        return {
            'logs':{k:np.mean(v) for k,v in pops.items()},
            # 'bootstraps':{
            #     k:scipy.stats.bootstrap((v,), np.mean).confidence_interval 
            #     for k,v in pops.items()},
            'pops':pops
        }


    def test(self):
        """Entry point to testing"""
        # TODO: should make a test split before doing serious
        # model comparison.
        # ds = torch.utils.data.Subset(self.valid_dataset, [0,1,2])
        ds = self.test_dataset
        loader = DataLoader(
            ds, 1,#self.batch_size,
            shuffle=False, num_workers=self.n_jobs if self.gpu else 0, pin_memory=self.gpu)

        results = []
        for perm, mask in gen_masks(self.model.note_dim):
            # TODO: bootstrap CI. need to return all likelihoods, not mean, from _validate
            r = self._validate(
                loader, ar_mask=mask.to(self.device, non_blocking=True),
                testing=True)
            # print(r['bootstraps'])
            perm = [['instrument', 'pitch', 'time', 'velocity'][i] for i in perm]
            results.append((perm, r['pops']))
        torch.save(results, self.results_dir / f'result-{self.epoch:04d}.pt')

    def train(self):
        """Entry point to model training"""
        self.save(self.model_dir / f'{self.epoch:04d}.ckpt')

        train_loader = DataLoader(
            self.train_dataset, self.batch_size,
            shuffle=True, num_workers=self.n_jobs, pin_memory=self.gpu)

        valid_loader = DataLoader(
            self.valid_dataset, self.batch_size,#//4,
            shuffle=False, num_workers=self.n_jobs, pin_memory=self.gpu,
            sampler=RandomSampler(
                self.valid_dataset, 
                num_samples=self.batch_size, replacement=True))

        ##### validation loop
        def run_validation():
            self.dataset.batch_len = self.dataset.max_test_len
            logs = self._validate(valid_loader, testing=False)['logs']
            self.log('valid', logs)

        epoch_size = self.epoch_size or len(train_loader)

        # validate at initialization
        run_validation()

        while True:
            self.epoch += 1

            ##### training loop
            self.model.train()
            self.dataset.testing = False
            self.dataset.batch_len = self.batch_len
            for batch in tqdm(
                # itertools incantation to support epoch_size larger than train set
                it.islice(
                    it.chain.from_iterable(it.repeat(train_loader)), epoch_size), 
                desc=f'training epoch {self.epoch}', total=epoch_size
                ):
                mask = batch['mask'].to(self.device, non_blocking=True)
                end = batch['end'].to(self.device, non_blocking=True)
                inst = batch['instrument'].to(self.device, non_blocking=True)
                pitch = batch['pitch'].to(self.device, non_blocking=True)
                time = batch['time'].to(self.device, non_blocking=True)
                vel = batch['velocity'].to(self.device, non_blocking=True)

                self.iteration += 1
                # TODO: use mask instead of batch dims
                self.exposure += self.batch_size * self.batch_len
                logs = {}

                ### forward+backward+optimizer step ###
                self.opt.zero_grad()
                result = self.model(inst, pitch, time, vel, end)
                losses = self.get_loss_components(result, mask[...,1:])
                loss = sum(losses.values())
                loss.backward()
                logs |= self.process_grad()
                self.opt.step()
                ########

                # log loss components
                logs |= {k:v.item() for k,v in losses.items()}
                # log total loss
                logs |= {'loss':loss.item()}
                # log any other returned scalars
                logs |= {k:v.item() for k,v in result.items() if v.numel()==1}
                self.log('train', logs)

            run_validation()

            if self.batch_len_schedule is not None:
                self.batch_len = min(
                    self.batch_len_max, self.batch_len+self.batch_len_schedule)
                self.dataset.batch_len = self.batch_len

            self.save(self.model_dir / f'{self.epoch:04d}.ckpt')

class Resumable:
    def __init__(self, checkpoint=None, resume=True, **kw):
        """
        Args:
            checkpoint: path to training checkpoint file
            resume: if True, retore optimizer states etc
                otherwise, restore only model weights (for transfer learning)
        """
        if checkpoint is not None:
            d = torch.load(checkpoint, map_location=torch.device('cpu'))
            print(f'loaded checkpoint {checkpoint}')
            if d['kw'].get('batch_len_schedule') is not None: print("""
            warning: checkpoints don't track `batch_len`. 
            be sure to manually set batch_len if resuming a run 
            using `batch_len_schedule`
            """)
            # merges sub dicts, e.g. model hyperparameters
            deep_update(d['kw'], kw)
            self._trainer = Trainer(**d['kw'])
            self._trainer.load_state(d, resume=resume)
        else:
            self._trainer = Trainer(**kw)

    def train(self):
        self._trainer.train()

    def test(self):
        self._trainer.test()

Resumable.__doc__ = Trainer.__init__.__doc__
Resumable.train.__doc__ = Trainer.train.__doc__

if __name__=='__main__':
    # TODO: improve fire-generated help message
    fire.Fire(Resumable)