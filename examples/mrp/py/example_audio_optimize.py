from iipyper import Audio, OSC, run, repeat, cleanup
import math
import copy
from time import process_time_ns
from collections import defaultdict
import random

import numpy as np
import scipy
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

    duty_limit = 90e9
    turning_point = 200e9

    def wrap(n):
        return (n - min_n) % (max_n - min_n + 1) + min_n

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
            if (q_new['pitch'] < -0.6 
                and n-1 not in mrp_data 
                and n-1 >= min_n 
                and note_duty[n-1] < duty_limit
                ):
                new_data[n-1] = q_new
                q_new['pitch'] += 1
                del new_data[n]
            if (q_new['pitch'] > 0.6 
                and n+1 not in mrp_data 
                and n+1 <= max_n 
                and note_duty[n+1] < duty_limit
                ):
                new_data[n+1] = q_new
                q_new['pitch'] -= 1
                del new_data[n]

        # for any notes over duty limit,
        for n, duty in note_duty.items():
            # switch to another note 
            new_n = n
            if n in new_data and duty > duty_limit:
                # print(n, new_data)
                # which is closest in the direction 'pitch' is pointing
                d = int(np.sign(new_data[n]['pitch']))
                for i in range(max_n-min_n):
                    new_n = wrap(new_n + d)
                    # and is below the limit 
                    # and not already sounding
                    if new_n not in new_data and note_duty[new_n] < duty_limit/2:
                        new_data[new_n] = new_data.pop(n)
                        print(f'switched {n} to {new_n}')
                        break
                    if i==max_n-min_n-1:
                        print('WARNING: all notes exhausted')
                        print(note_duty)
            # with wraparound
            # print a warning if all notes are exhausted (shouldn't happen)

        return new_data

    # roughness_filter = Filter(30, width_hz=5.0, sample_rate=sample_rate)
    # def get_feat(audio):
    #     mag = np.maximum(audio, 0)
    #     roughness = roughness_filter(mag-mag.mean())[block_size//4:]
    #     power_sq = np.mean(audio**2)
    #     rough_power_sq = np.mean(roughness**2)
    #     return (rough_power_sq / (power_sq + 1e-7)) ** 0.5
    window = np.hanning(block_size)
    def get_feat(audio):
        spectrum = np.fft.rfft(audio*window)
        mag = np.abs(spectrum)
        ent = scipy.stats.entropy(mag)
        # print(ent)
        return ent / np.log2(block_size)

    # notes = np.random.choice(np.arange(21, 109), 16, replace=False)
    notes = [21+12*i for i in range(6)]+[21+5+12*i for i in range(6)]
    note_duty = {n:0 for n in range(min_n, max_n+1)}
    mrp_data = [{
        int(n):{
            # 'intensity':  0.5,#random.random(),
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
            mrp.quality_update(n, 'intensity', 1.0)

    set_notes()
    for n, q in mrp_data[-1].items():
        mrp.qualities_update(n, q)


    frame_count = 0

    # process_every = 1

    lag = Lag(0.5)
    audio_features = []

    t = None

    def callback(indata, frames, time, status):
        if status:
            print(f'sounddevice error {status=}')
        nonlocal t, frame_count
        new_t = process_time_ns()
        delta_t = (new_t - t) if t is not None else 0
        # if delta_t > block_size / sample_rate * 1.9e9:
        #     print('XXXX')
        t = new_t
        # print(f'{frames=}')
        # print(f'{time=}')
        audio = np.mean(indata, -1)
        audio_feat = lag(get_feat(audio))
        # audio_feat = get_feat(audio)
        # print(audio_feat)
        # print('█'*int(80*min(1, audio_feat**0.5)))

        t_audio = process_time_ns()

        frame_count+=1

        # if (frame_count % process_every)==0:

        maximize = (t < turning_point)
        # print(maximize)

        # if len(audio_features) < 2 or np.random.choice(2):
        if len(audio_features) < 2:
            accept = True
        else:
            accept = audio_feat > audio_features[-1]
            if not maximize: 
                accept = not accept
        if accept:                
            # accept change
            msg = 'accept   '
            branch_from = -1
        else:
            # rollback change
            msg = 'rollback '
            branch_from = -2

        audio_features.append(audio_feat)
        mrp_data.append(mutate(mrp_data[branch_from]))

        t_mutate = process_time_ns()

        notes_on = set(mrp.note_on_numbers())
        print(notes_on)
        for n in notes_on:
            # increment duty
            note_duty[n] += delta_t
            # turn off any notes which are no longer playing
            if n not in mrp_data[-1]:
                # print(f'note off {n}')
                mrp.note_off(n)
        # decrement duty for notes which are not currently playing
        for n in note_duty:
            if n not in notes_on:
                note_duty[n] = max(0, note_duty[n]-delta_t)
        # start / update notes
        for n, q in mrp_data[-1].items():
            if n not in notes_on:
                # print(f'note on {n}')
                mrp.note_on(n)
            mrp.qualities_update(n, q)

        t_mrp = process_time_ns()

        # t_str = f'{int((t_audio-t)*1e-6)} ms\t{int((t_mutate-t_audio)*1e-3)} us\t{int((t_mrp-t_mutate)*1e-6)} ms\t'
        t_str = f'{int((t_audio-t)*1e-6)} ms\t{int((t_mrp-t_mutate)*1e-6)} ms\t'

        print(t_str + msg + '█'*int(60*min(1, audio_feat**0.5)) + f' {audio_feat}')
        danger_notes = sorted([
            (d*1e-9,n) for n,d in note_duty.items() if d > 90e9])
        if len(danger_notes):
            print('duty:', danger_notes)

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
