import sys
import taichi as ti

from tolvera.tolvera import main as world

from tolvera.vera._boids import main as boids
from tolvera.vera._physarum import main as physarum
from tolvera.vera._lenia import main as lenia
from tolvera.vera._reaction_diffusion import main as rea_diff

def help():
    print("""
    available subcommands:
        help:     list available subcommands
        boids:    run Boids example
        physarum: run Physarum example
        lenia:    run Lenia example
        physarum: run Physarum example
        reaction_diffusion: run ReactionDiffusion example
    """)

if __name__=='__main__':
    print(sys.argv)
    try:
        match sys.argv[1]:
            case 'boids':
                boids()
            case 'physarum':
                physarum()
            case 'reaction_diffusion':
                rea_diff()
            case 'lenia':
                lenia()
            case 'world':
                world()
            case 'help':
                help()
            case _:
                help()
    except IndexError:
        help()
