"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
This example corresponds to the help file for the `updsend` and `updreceive` Max objects.
In the Max help file, make sure the `updreceive` object's port is set to 7401.
"""

from iipyper import OSC, run, repeat

def main(port=7400):
    
    osc = OSC(port=port)
    osc.create_client("max", port=7401)
 
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
