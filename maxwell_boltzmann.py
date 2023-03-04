import math
import numpy

def pdf(speed):
    k = 1.7 # Boltzmann Constant
    T = 1 # Temperature
    m = 1 # Particle mass

    a = math.sqrt(k * T / m)
    return math.sqrt(2 / math.pi) * speed**2 * math.e**(-speed**2 / (2*a**2)) / a**3


def test_valid_pdf():
  # Make sure integrating over pdf sums to 1
  xs = []
  ys = []
  num_sections = 10
  for x in range(num_sections + 1):
      xs.append(10 * x/num_sections)
      ys.append(pdf(10 * x / num_sections))
  print(xs, ys)
  print(numpy.trapz(ys, xs))

test_valid_pdf()
