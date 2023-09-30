'''
TODO: fire CL args each main() func
TODO: default app with OSC etc.
'''

import sys
import taichi as ti

from iipyper import run

from tolvera.vera.simple import main as simple
from tolvera.vera.boids import main as boids
from tolvera.vera.physarum import main as physarum
from tolvera.vera.particle_life import main as particle_life
from tolvera.cv import main as CV

# from tolvera.vera._lenia import main as lenia
# from tolvera.vera._reaction_diffusion import main as rea_diff
# from tolvera.vera._mrp import main as mrp
# from tolvera.vera._obstacles import main as obstacles

def help():
    print("""
    available subcommands:
        help:     list available subcommands
        simple:   run Particles & Pixels example
        boids:    run Boids example
        physarum: run Physarum example
        lenia:    run Lenia example
        particle_life: run ParticleLife example
        reaction_diffusion: run ReactionDiffusion example
        obstacles: run Obstacles exmaple
    """)

def main():
    print(sys.argv)
    try:
        match sys.argv[1]:
            case 'simple':
                simple()
            case 'boids':
                boids()
            case 'physarum':
                physarum()
            case 'particle_life':
                particle_life()
            case 'cv':
                CV()
            # case 'reaction_diffusion':
            #     rea_diff()
            # case 'lenia':
            #     lenia()
            # case 'world':
            #     world()
            # case 'help':
            #     help()
            # case 'mrp':
            #     mrp()
            # case 'obstacles':
            #     mrp()
            case _:
                help()
    except IndexError:
        help()


if __name__=='__main__':
    run(main)
