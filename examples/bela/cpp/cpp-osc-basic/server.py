"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
To run this example, open the Bela OSC example at:
https://github.com/BelaPlatform/Bela/blob/master/examples/Communication/OSC/render.cpp

And change the IP address to the following:
```
const char* remoteIp = "192.168.7.1";
```
"""

from iipyper import OSC, run, repeat

def main(host="192.168.7.1", port=7563):
    
    osc = OSC(host, port)
    osc.create_client("bela", host="192.168.7.2", port=7562)

    connected = False
    count = 0
 
    @osc.args("/*")
    def _(address, *args):
        """
        Handle OSC messages from Bela
        """
        print(f"{address} {args}")

        if address=="/osc-setup":
            nonlocal connected
            connected = True
            print("Bela connected!")
            osc("bela", "/osc-setup-reply")

        elif address=="/osc-acknowledge":
            print(f"Bela acknowledged osc-test: {args}")
            
        else:
            print(f"Unrecognised OSC {address} with {args}")

    @repeat(1)
    def _():
        nonlocal connected
        nonlocal count
        if connected==True:
            osc("bela", "/osc-test", count, 3.14)
            count=count+1
        else:
            print("Waiting for Bela to connect...")

if __name__=='__main__':
    run(main)
