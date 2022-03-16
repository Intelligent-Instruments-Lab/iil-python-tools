"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from notepredictor import NotePredictor

from iipyper import OSC, run

def main(host="127.0.0.1", port=9999, checkpoint=None):
    osc = OSC(host, port)

    if checkpoint is not None:
        predictor = NotePredictor.from_checkpoint(checkpoint)
        predictor.eval()
    else:
        predictor = None
 
    @osc.kwargs('/predictor/*')
    def _(address, **kw):
        """
        Handle OSC messages to Predictor
        """
        print(f"{address} {kw}")

        address = address.split("/")
        cmd = address[2]

        if cmd=="load":
            # `nonlocal` is needed to assign to closed-over name
            nonlocal predictor
            predictor = NotePredictor.from_checkpoint(**kw)
            predictor.eval()

        elif cmd=="predict":
            if predictor is None:
                print('no model loaded')
            else:
                r = predictor.predict(**kw)
                return '/prediction', r['pitch'], r['time']

        elif cmd=="reset":
            if predictor is None:
                print('no model loaded')
            else:
                predictor.reset(**kw)
            
        else:
            print(f"PitchPredictor: Unrecognised OSC {address} with {kw}")

if __name__=='__main__':
    run(main)
