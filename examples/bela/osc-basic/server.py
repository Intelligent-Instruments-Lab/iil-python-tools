"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from iipyper import OSC, run, repeat

def main(host="192.168.7.1", port=9999):
    
    osc = OSC(host, port)
    osc.create_client("bela", host="192.168.7.2", port=8888)

    connected = False
    count = 0
 
    @osc.kwargs("/*")
    def _(address, **kw):
        """
        Handle OSC messages from Bela
        """
        print(f"{address} {kw}")

        if address=="/osc-setup":
            nonlocal connected
            connected = True
            print("Bela connected!")
            osc("bela", "/osc-setup-reply")

        elif address=="/osc-acknowledge":
            print(f"Bela acknowledged osc-test: {kw}")
            
        else:
            print(f"Unrecognised OSC {address} with {kw}")

    @repeat(1)
    def _():
        nonlocal connected
        nonlocal count
        if connected==True:
            osc("bela", "/osc-test", count)
            count=count+1
        else:
            print("Waiting for Bela to connect...")

if __name__=='__main__':
    run(main)
