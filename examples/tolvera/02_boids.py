from iipyper import run
import tolvera as tol

def main(x=1920,y=1080,n=1024,species=4):
  tol.init(x=x,y=y,n=n,species=species)
  particles = tol.Particles(x, y, n, species)
  pixels = tol.Pixels(x, y, evaporate=0.95)
  boids = tol.vera.Boids(x, y, species)

  def _():
    pixels.diffuse()
    boids(particles)
    particles(pixels)

  tol.utils.render(_, pixels)

if __name__ == '__main__':
  run(main)
