from tolvera.utils import *
from tolvera.particles import Particles
from tolvera.pixels import Pixels
from iipyper import run

def main():
    init()
    particles = Particles(x,y,n,species)
    pixels = Pixels(x,y)

    def _():
        pixels.clear()
        particles(pixels)

    render(_, pixels)

if __name__ == '__main__':
    run(main)
