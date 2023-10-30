import numpy as np

def mandelbrot(z0, c, iterations, sequence=False):
    k = 0
    z = [z0]
    while k < iterations:
        z.append(z[k] * z[k] + c)
        k += 1

    if sequence:
        return z
    else:
        return z[-1]

def modulo(z):
    return np.sqrt((z.real)*(z.real) + (z.imag)*(z.imag))






#print(mandelbrot(z0=1+1j, c=0, iterations=15))
