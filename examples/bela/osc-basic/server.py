"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from iipyper import OSC, run

def main(host="127.0.0.1", port=9999, checkpoint=None):
    osc = OSC(host, port)
    osc.create_client('bela', port=8888)
    connected = False
    count = 0
 
    @osc.kwargs('/bela/*')
    def _(address, **kw):
        """
        Handle OSC messages from Bela
        """
        print(f"{address} {kw}")

        address = address.split("/")
        cmd = address[2]

        if cmd=="osc-setup":
            connected = True
            osc("bela", "osc-setup-reply", "")

        elif cmd=="osc-acknowledge":
            print(f"Bela: test acknowledged: {kw}")
            
        else:
            print(f"Bela: Unrecognised OSC {address} with {kw}")

    if connected=True:
        osc("bela", "osc-test", [count++, 3.14])

if __name__=='__main__':
    run(main)
