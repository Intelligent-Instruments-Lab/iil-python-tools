"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

"""
This example demonstrates send & receive of JSON using OSC strings
(as an alternate to OSC bundles for structured data)
"""

from iipyper import OSC, run, repeat
import json

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

        if address=="/jsondict":
            jsondict = json.loads(args[0])
            print('jsondict', jsondict)

    @repeat(1)
    def _():
        nonlocal count
        msg = {
            "count": count,
            "list": [0,1,2,"hi"]
        }
        osc("max", "/test", json.dumps(msg))
        count = count + 1

if __name__=='__main__':
    run(main)
