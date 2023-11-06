from tolvera.utils import init, render
from tolvera.particles import Particles
from tolvera.pixels import Pixels
from iipyper import run

def main(**kwargs):
    o = init(**kwargs)
    particles = Particles(o.x,o.y,o.particles,o.species)
    pixels = Pixels(o.x,o.y)

    def _():
        pixels.clear()
        particles(pixels)

    render(_, pixels)

if __name__ == '__main__':
    run(main)
