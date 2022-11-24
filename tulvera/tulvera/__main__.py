import sys
import taichi as ti

from tulvera.vera._boids import main as boids
from tulvera.vera._physarum import main as physarum
from tulvera.tulvera import main as world

def help():
    print("""
    available subcommands:
        help: list available subcommands
        boids: run Boids example
        physarum: run Physarum example
    """)

if __name__=='__main__':
    print(sys.argv)
    try:
        match sys.argv[1]:
            case 'boids':
                boids()
            case 'physarum':
                physarum()
            case 'world':
                world()
            case 'help':
                help()
            case _:
                help()
    except IndexError:
        help()
