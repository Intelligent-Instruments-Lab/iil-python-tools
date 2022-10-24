"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from iipyper import OSC, run
import mrp

def main(host="127.0.0.1", receive_port=8888, send_port=7777):
    osc = OSC(host, receive_port, send_port)

    mrp = None
 
    @osc.args(return_port=7777)
    def feed(address, value):
        """
        docstring
        """
        print(f"{address} {value}")
        # mrp.func(value)

    @osc.args(return_port=7777)
    def reset(address, kind=None):
        """
        reset the mrp
        """
        print(f"{address} {kind}")

        nonlocal mrp
        mrp = MRP()

    reset(None)

if __name__=='__main__':
    run(main)
