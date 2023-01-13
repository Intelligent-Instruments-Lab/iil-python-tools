import numpy as np
from iipyper import OSC, run, repeat

def main(host="192.168.7.1", port=5000):
    osc = OSC(host, port, verbose=True)
    osc.create_client("bela", host="192.168.7.2", port=57120)

    @osc.args("/*")
    def _(address, *args):
        print(f"{address} {args}")

    @repeat(1)
    def _():
        arr = [
            3, #// 7 , # shift_onsets
            -0.03, # shift_hz
            1, # hz_mix
            1, # onsets_mix
            0.02, # 0.1 , # onsets_floor
            700, # low_pass
            10, # high_pass
            np.random.rand(),#0.2, # drive
            0.02, # 0.01 , # thresh
            1  # comp_exponent
        ]
        # print(arr)
        # osc("bela", "/langspil", *arr)

if __name__=='__main__':
    run(main)
