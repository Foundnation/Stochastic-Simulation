import numpy as np
import matplotlib.pyplot as plt

# ? change name cause it's not necessarily mandelbrot result?
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

def generate_mandelbrot(z0, c_array, iterations, c_threshold, heights = False):
    mandelbrot_set = []
    heigtmap = []

    if heights:
        for c in c_array:
            f = mandelbrot(z0, c, iterations)
            if modulo(f) < c_threshold:
                mandelbrot_set.append(c)
                heigtmap.append(f)
        
        return mandelbrot_set, heigtmap
    
    else:
        for c in c_array:
            f = mandelbrot(z0, c, iterations)
            if modulo(f) < c_threshold:
                #print(modulo(f))
                mandelbrot_set.append(c)

        return mandelbrot_set
        
        

c_shape = (1, int(1E6))
c_arr = (np.random.uniform(-2, 1, c_shape) + 1.j * np.random.uniform(-1, 1, c_shape))[0]

mand_set, colormap_complex = generate_mandelbrot(0, c_arr, 100, 4, heights=True)

x = [elem.real for elem in mand_set]
y = [elem.imag for elem in mand_set]
colormap = [modulo(elem) for elem in colormap_complex]

print(len(c_arr))
print(len(mand_set))
print(len(mand_set) / len(c_arr))


cmap = plt.get_cmap('viridis')
colors = cmap(colormap)
plt.scatter(x, y, c=colors, s=0.1)
plt.show()