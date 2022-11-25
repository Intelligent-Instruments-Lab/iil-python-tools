import sys
from pysensel.tisensel import main as tisensel

def help():
    # TODO: Add test/sim for device-less dev
    print("""
    available subcommands:
        device: test with device connected
        simulate: simulation of sensel inputs
    """)

if __name__=='__main__':
    print(sys.argv)
    try:
        match sys.argv[1]:
            case 'device':
                tisensel()
            case 'help':
                help()
            case _:
                help()
    except IndexError:
        help()
