"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
Create a file called `process.dsp` and add the following code

```
import("stdfaust.lib");
process = no.noise*hslider("level",0,0,1,0.01);
```

Open this in FaustLive or compile it via faust2X
"""

import random

from iipyper import OSC, run, repeat

def main(port=5511, verbose=False):
    
    osc = OSC(port=port)
    osc.create_client("faust", port=5510)

    connected = False

    @osc.args("/*")
    def _(address, *args):
        """
        Handle OSC messages from Faust
        """

        if address=="/FaustDSP":
            # `osc("faust", "/*", "get")` response
            print(f"Faust interface: {args}")
        
        else:
            address = address.split("/")
            cmd = address[2]
            if cmd=="level":
                print(f"Level updated to: {args}")

    @repeat(1)
    def _():
        nonlocal connected
        if connected==False:
            connected=True
            osc("faust", "/*", "get") # discover OSC interface
            osc("faust", "/xmit", 1) # turn transmission on
        osc("faust", "/FaustDSP/level", random.random())

if __name__=='__main__':
    run(main)
