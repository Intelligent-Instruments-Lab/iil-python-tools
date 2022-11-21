import sys
import taichi as ti

from tulvera.vera._physarum import main as physarum
from tulvera.vera._boids import main as boids

def help():
    print("""
    available subcommands:
        boids: run the Boids example
        physarum: run the Physarum example
    """)

if __name__=='__main__':
    print(sys.argv)
    try:
        match sys.argv[1]:
            case 'boids':
                boids()
            case 'physarum':
                physarum()
            case 'help':
                help()
            case _:
                help()
    except IndexError:
        help()
