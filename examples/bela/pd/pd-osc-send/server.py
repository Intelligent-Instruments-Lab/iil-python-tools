"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
Run this example alongside the Bela Pure Data osc-send example at:
https://github.com/BelaPlatform/Bela/tree/master/examples/PureData/osc-send
"""

from iipyper import OSC, run, repeat

def main(host="192.168.7.1", port=7562):
    
    osc = OSC(host, port)
 
    @osc.args("/slider")
    def _(address, *args):
        """
        Handle OSC messages from Bela
        """
        print(f"{address} {args}")

        if address=="/slider":
            print("slider", args)

        elif address=="/counter":
            print("slider", args)
            
        else:
            print(f"Unrecognised OSC {address} with {args}")

if __name__=='__main__':
    run(main)
