"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
This example corresponds to the help file for CNMAT's OpenSoundControl Max object.
"""

from iipyper import OSC, run, repeat

def main(port=7005):
    
    osc = OSC(port=port)
    osc.create_client("max", port=5432)
 
    count = 0

    @osc.args("/*")
    def _(address, *args):
        """
        Handle OSC messages from Max
        """
        print(f"{address} {args}")

    @repeat(1)
    def _():
        nonlocal count
        osc("max", "/test", count)
        count = count + 1

if __name__=='__main__':
    run(main)
