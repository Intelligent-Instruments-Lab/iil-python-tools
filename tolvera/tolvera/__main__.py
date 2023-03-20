import sys
import taichi as ti

# TODO: CL arg for arch

from tolvera.tolvera import main as world

from tolvera.vera.simple import main as simple
from tolvera.vera.boids import main as boids
from tolvera.vera.physarum import main as physarum

# from tolvera.vera._boids import main as boids
# from tolvera.vera._boids_multi import main as boids_multi
# from tolvera.vera._physarum import main as physarum
# from tolvera.vera._physarum_multi import main as physarum_multi
from tolvera.vera._lenia import main as lenia
from tolvera.vera._reaction_diffusion import main as rea_diff
from tolvera.vera._mrp import main as mrp
from tolvera.vera._obstacles import main as obstacles

def help():
    print("""
    available subcommands:
        help:     list available subcommands
        simple:   run Particles & Pixels example
        boids:    run Boids example
        physarum: run Physarum example
        lenia:    run Lenia example
        physarum: run Physarum example
        reaction_diffusion: run ReactionDiffusion example
        obstacles: run Obstacles exmaple
    """)

if __name__=='__main__':
    print(sys.argv)
    try:
        match sys.argv[1]:
            case 'simple':
                simple()
            case 'boids':
                boids()
            # case 'boids_multi':
            #     boids_multi()
            case 'physarum':
                physarum()
            # case 'physarum_multi':
            #     physarum_multi()
            case 'reaction_diffusion':
                rea_diff()
            case 'lenia':
                lenia()
            case 'world':
                world()
            case 'help':
                help()
            case 'mrp':
                mrp()
            case 'obstacles':
                mrp()
            case _:
                help()
    except IndexError:
        help()
