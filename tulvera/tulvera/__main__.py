import sys
import taichi as ti

from tulvera.vera._boids import main as boids
from tulvera.vera._boids_p import main as boids_p
from tulvera.vera._physarum import main as physarum
from tulvera.vera._physarum_p import main as physarum_p

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
            case 'boids_p':
                boids_p()
            case 'physarum':
                physarum()
            case 'physarum_p':
                physarum_p()
            case 'help':
                help()
            case _:
                help()
    except IndexError:
        help()
