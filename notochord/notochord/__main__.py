import sys

from iipyper import run

from notochord.server import main as server
from notochord.harmonizer import main as harmonizer
from notochord.improviser import main as improviser

def help():
    print("""
    available subcommands:
        server: run the Notochord OSC server
    """)

if __name__=='__main__':
    # print(sys.argv)
    try:
        if sys.argv[1] == 'server':
            sys.argv = sys.argv[1:]
            run(server)
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