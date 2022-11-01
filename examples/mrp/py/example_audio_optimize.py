from iipyper import Audio, OSC, run, repeat, cleanup
import math
import copy
from time import process_time_ns
from collections import defaultdict
import random

import numpy as np
import sounddevice as sd
from scipy.signal import butter, kaiserord, lfilter, firwin, freqz

from mrp import MRP

#  TODO:
# changing notes to respect 90s duty limitation
    
class Lag:
    def __init__(self, coef, val=None):
        self.coef = coef
        self.val = val

    def __call__(self, val):
        if self.val is None:
            self.val = val
        else:
            self.val = self.val*self.coef + val*(1-self.coef)
        return self.val

class Filter:
    def __init__(self, 
        cutoff_hz, width_hz=5.0, sample_rate=48000):
        """
        Args:
            cutoff_hz: cutoff frequency in Hz
            width_hz: width in Hz
        """
        # nyq_rate = sample_rate / 2.0
        # width = width_hz / nyq_rate
        # # Compute the order and Kaiser parameter for the FIR filter.
        # N, beta = kaiserord(ripple_db, width)
        # # Use firwin with a Kaiser window to create a lowpass FIR filter.
        # self.taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

        nyq = 0.5 * sample_rate
        low = (cutoff_hz - width_hz) / nyq
        high = (cutoff_hz + width_hz ) / nyq
        order = 3
        self.b, self.a = butter(order, [low, high], btype='band')

    def __call__(self, x):
        return lfilter(self.b, self.a, x)
        # return lfilter(self.taps, 1.0, x)

# print(filter(np.random.randn(4096), 30))
# exit(0)


def main(host="127.0.0.1", receive_port=7563, send_port=7770,
        block_size=4096, device='Macbook Pro Mic', sample_rate=48000,
        channels=1,    
        ):

    osc = OSC(host, receive_port, verbose=False)
    osc.create_client("mrp", host="127.0.0.1", port=send_port)
    mrp = MRP(osc, verbose=False)

    min_n = mrp.settings['range']['start']
    max_n = mrp.settings['range']['end']

    mutate_amt = defaultdict(lambda:0.03)
    mutate_amt['pitch'] = 0.09

    def mutate(mrp_data):
        new_data = {}
        for n,q in mrp_data.items():
            new_data[n] = q_new = copy.deepcopy(q)

            for i in range(16):
                q_new['harmonics_raw'][i] = min(1,max(0, q['harmonics_raw'][i] 
                    + (random.random()-0.5)*mutate_amt[f'harmonics_{i}']))
            # for k in ('intensity', 'brightness', 'harmonic'):
            #     q_new[k] = min(1,max(0, q[k] 
            #         + (random.random()-0.5)*mutate_amt[k]))

            q_new['pitch'] = min(1,max(-1, q['pitch'] 
                + (random.random()-0.5)*mutate_amt['pitch']))
            if q_new['pitch'] < -0.6 and n-1 not in mrp_data and n-1 >= min_n:
                new_data[n-1] = q_new
                q_new['pitch'] += 1
                del new_data[n]
            if q_new['pitch'] > 0.6 and n+1 not in mrp_data and n+1 <= max_n:
                new_data[n+1] = q_new
                q_new['pitch'] -= 1
                del new_data[n]

        return new_data

    # window = np.hanning(block_size)
    roughness_filter = Filter(30, width_hz=5.0, sample_rate=sample_rate)
    def get_feat(audio):
        # return np.std(audio)
        # audio = audio * window
        # mag = np.abs(audio)
        mag = np.maximum(audio, 0)
        roughness = roughness_filter(mag-mag.mean())[block_size//4:]
        power_sq = np.mean(audio**2)
        rough_power_sq = np.mean(roughness**2)
        # print(np.max(np.abs(roughness)), np.max(np.abs(audio)))
        # print(rough_power_sq, power_sq)
        # print(power_sq, rough_power_sq)
        return (rough_power_sq / (power_sq + 1e-7)) ** 0.5


    # notes = np.random.choice(np.arange(21, 109), 16, replace=False)
    notes = [21+12*i for i in range(6)]+[21+5+12*i for i in range(6)]
    note_duty = defaultdict(float)
    mrp_data = [{
        int(n):{
            'intensity':  0.5,#random.random(),
            # 'brightness': 0,#random.random(),
            # 'harmonic':   0,#random.random(),
            'pitch':      0,#random.random(),
            'harmonics_raw': [0]*16
        } for n in notes
    }]

    def set_notes():
        mrp.all_notes_off()
        for n in mrp_data[-1]:
            mrp.note_on(n)

    set_notes()
    for n, q in mrp_data[-1].items():
        mrp.qualities_update(n, q)


    frame_count = 0
    prev_feat = 0

    process_every = 1

    lag = Lag(0.5)
    audio_features = []

    t = None

    def callback(indata, frames, time, status):
        if status:
            print(f'sounddevice error {status=}')

        nonlocal t, frame_count
        new_t = process_time_ns()
        delta_t = (new_t - t) if t is not None else 0
        t = new_t
        # print(f'{frames=}')
        # print(f'{time=}')
        audio = np.mean(indata, -1)
        audio_feat = lag(get_feat(audio))
        # print(audio_feat)
        # print(audio_feat)
        # print('█'*int(80*min(1, audio_feat**0.5)))

        frame_count+=1

        if (frame_count % process_every)==0:

            # if len(audio_features) < 2 or np.random.choice(2):
            if len(audio_features) < 2 or audio_feat > audio_features[-1]:
                # accept change
                msg = 'accept   '
                branch_from = -1
            else:
                # rollback change
                msg = 'rollback '
                branch_from = -2

            audio_features.append(audio_feat)
            mrp_data.append(mutate(mrp_data[branch_from]))

            notes_on = set(mrp.note_on_numbers())
            for n in notes_on:
                note_duty[n] += delta_t
                if n not in mrp_data[-1]:
                    print(f'note off {n}')
                    mrp.note_off(n)
            for n in note_duty:
                if n not in notes_on:
                    note_duty[n] = max(0, note_duty[n]-delta_t)
            for n, q in mrp_data[-1].items():
                if n not in notes_on:
                    print(f'note on {n}')
                    mrp.note_on(n)
                mrp.qualities_update(n, q)

            print(msg+ '█'*int(60*min(1, audio_feat**0.5)) + f' {audio_feat}')
            print('duty:', sorted([
                (d*1e-9,n) for n,d in note_duty.items() if d > 90e9]))

    @cleanup
    def _():
        # pass
        mrp.all_notes_off()
    
    Audio(callback=callback,
        channels=channels, device=device, 
        blocksize=block_size, samplerate=sample_rate)
    # return sd.InputStream(callback=callback,
            # channels=1, device=device, 
            # blocksize=block_size, samplerate=sample_rate)
        # while True:
        # pass

if __name__ == "__main__":
    print(sd.query_devices())
    run(main)
