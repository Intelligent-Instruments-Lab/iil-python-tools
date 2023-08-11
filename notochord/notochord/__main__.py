import sys

from iipyper import run

from notochord.app import *

def help():
    print("""
    available subcommands:
        server: run the Notochord OSC server
        homunculus: run the Notochord homunculus TUI
        harmonizer: run the Notochord harmonizer TUI
        improviser: run the Notochord improviser TUI
    """)

if __name__=='__main__':
    # print(sys.argv)
    try:
        if sys.argv[1] == 'server':
            sys.argv = sys.argv[1:]
            run(server)
        if sys.argv[1] == 'homunculus':
            sys.argv = sys.argv[1:]
            run(homunculus)
        if sys.argv[1] == 'harmonizer':
            sys.argv = sys.argv[1:]
            run(harmonizer)
        if sys.argv[1] == 'improviser':
            sys.argv = sys.argv[1:]
            run(improviser)
        else:
            help()
    except IndexError:
        help()