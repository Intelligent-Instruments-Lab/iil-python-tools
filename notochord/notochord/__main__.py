import sys

from iipyper import run

def help():
    print("""
    available subcommands:
        server: run the Notochord OSC server
    """)

if __name__=='__main__':
    # print(sys.argv)
    try:
        if sys.argv[1] == 'server':
            from .server import main
            sys.argv = sys.argv[1:]
            run(main)
        else:
            help()
    except IndexError:
        help()