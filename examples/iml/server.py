"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from iml import IML
from iipyper import OSC, run
import numpy as np

def main(host="127.0.0.1", receive_port=9999, send_port=None, checkpoint=None):
    osc = OSC(host, receive_port)

    iml = IML(2)

    osc(iml)

if __name__=='__main__':
    run(main)
