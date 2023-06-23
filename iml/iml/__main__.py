import sys

from iipyper import run

from iml.app import *

def help():
    print("""
    available subcommands:
        server: run the IML OSC server
    """)

if __name__=='__main__':
    # print(sys.argv)
    try:
        if sys.argv[1] == 'server':
            sys.argv = sys.argv[1:]
            run(server)
        else:
            help()
    except IndexError:
        help()