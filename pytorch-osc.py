"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

import argparse
from lib import OSCServer as OSC
from lib import PitchPredictor as Predictor

predictor = Predictor()

def predictor_msg_handler(address, args):
  """
  Handle OSC messages to Predictor
  Passed to OSCServer as custom process method
  """
  # TODO: verify split string
  add = address.split("/")
  if(add[0] is not "predictor"):
    print('[predictor_msg_handler()] OSC message {} received at unrecognised address {}.'.format(msg, address))
  
  if(add[1] == "forward"):
    # TODO: check args for (Int Arr) Notes
    predictor.forward(args)
  else if(add[1] == "predict")
    # TODO: check args for (Int) Note
    predictor.predict(args)
  else if(add[1] == "reset"):
    predictor.reset()
  # else if(add[1] == "options"):
    # TODO: handle predictor options (and reset?)

def main(args):
  osc = OSC(args.ip, args.tx, args.rx, process=predictor_msg_handler)
  osc.start()
  # TODO: handle event loop

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--ip', default=OSC.ip, help='OSC receive IP, quoted')
  parser.add_argument('--tx', default=OSC.tx, type=int, help='OSC receive port')
  parser.add_argument('--rx', default=OSC.rx, type=int, help='OSC send port')
  parser.add_argument('--endpt', default=OSC.endpt, help='OSC address, quoted')
  args = parser.parse_args()
  main(args)
