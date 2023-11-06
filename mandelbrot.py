import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

# ? change name cause it's not necessarily mandelbrot result?
@jit
def mandelbrot_func(z0, c, iterations, iteration_bins, sequence=False):
    k = 0
    z = [z0]
    while k < iterations:
        z.append(z[k] * z[k] + c)
        k += 1

    z_bins = []
    if sequence:
        z_step = int(len(z)/iteration_bins)
        for i in range(iteration_bins):
            z_bins.append(z[i * z_step])

        return z_bins
    else:
        return z[-1]

@jit
def modulo(z):
    return np.sqrt((z.real)*(z.real) + (z.imag)*(z.imag))

@jit
def generate_mandelbrot(z0, c_array, iterations, iteration_bins, z_threshold, heights = False):
    mandelbrot_set = []
    heightmap = []

    if heights:
        for c in c_array:
            f = mandelbrot_func(z0, c, iterations, iteration_bins, sequence=False)
            if modulo(f) < z_threshold:
                mandelbrot_set.append(c)
                heightmap.append(f)
        
        return mandelbrot_set, heightmap
    
    else:
        for c in c_array:
            f = mandelbrot_func(z0, c, iterations, iteration_bins, sequence=False)
            if modulo(f) < z_threshold:
                #print(modulo(f))
                mandelbrot_set.append(c)

        return mandelbrot_set
        
def main():
    c_shape = (1, int(1E6))
    left_bound, right_bound = -2, 1
    bottom_bound, top_bound = -1, 1
    c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]

    mand_set, colormap_complex = generate_mandelbrot(0, c_arr, 100, 10, 4, heights=True)

    x = [elem.real for elem in mand_set]
    y = [elem.imag for elem in mand_set]
    colormap = [modulo(elem) for elem in colormap_complex]

    print(len(c_arr))
    print(len(mand_set))

    S_rect = abs(right_bound - left_bound) * abs(top_bound - bottom_bound)
    print('area = ', len(mand_set) / len(c_arr) * S_rect)


    cmap = plt.get_cmap('viridis')
    colors = cmap(colormap)
    plt.scatter(x, y, c=colors, s=0.1)
    plt.show()




if __name__ == '__main__':
    main()