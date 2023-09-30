from iipyper import run
import tolvera as tol

def main(x=1920,y=1080,n=1024,species=3,attractors=3):
  tol.init(x=x,y=y,n=n,species=species)
  particles = tol.Particles(x, y, n, species)
  pixels = tol.Pixels(x, y, evaporate=0.95)
  boids = tol.vera.Boids(x, y, species)
  attractors = tol.vera.Attractors(x, y, n=attractors)

  def _():
    pixels.clear()
    particles.seeks(attractors)
    boids(particles)
    particles(pixels)

  tol.utils.render(_, pixels)

if __name__ == '__main__':
  run(main)
