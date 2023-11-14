import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

#@jit
def mandelbrot_func(z0, c, iterations, num_intermediate_steps, sequence=False):
    k = 0
    z = [z0]
    while k < iterations:
        z.append(z[k] * z[k] + c)
        k += 1

    z_bins = []
    if sequence:
        z_bins_length = len(z) // num_intermediate_steps
        #print('z_bins_length', z_bins_length)
        for i in range(num_intermediate_steps):
            #print(i)
            #print('i * z_bins_length', i * z_bins_length)
            #print('z[i * z_bins_length]', z[i * z_bins_length])
            z_bins.append(z[i * z_bins_length] - 1)
        return z_bins
    else:
        return z[-1]

#@jit
def modulo(z):
    return np.sqrt((z.real)*(z.real) + (z.imag)*(z.imag))

#@jit
def generate_mandelbrot(z0, c_array, iterations, num_intermediate_steps, z_threshold, sequence = False, heights = False):
    mandelbrot_set = []
    heightmap = []

    if heights == True and sequence == False:
        for c in c_array:
            f = mandelbrot_func(z0, c, iterations, num_intermediate_steps, sequence)
            if modulo(f) < z_threshold:
                mandelbrot_set.append(c)
                heightmap.append(f)
        
        return mandelbrot_set, heightmap
    
    elif heights == False and sequence == False:
        for c in c_array:
            f = mandelbrot_func(z0, c, iterations, num_intermediate_steps, sequence)
            if modulo(f) < z_threshold:
                mandelbrot_set.append(c)
        return mandelbrot_set

    # WORK IN PROGRESS
    elif heights == False and sequence == True :
        mandelbrot_set_array = []
        len_f_array = num_intermediate_steps
        for i in range(len_f_array):
            mandelbrot_set_array.append([])

        for c in c_array:
            f_array = mandelbrot_func(z0, c, iterations, num_intermediate_steps, sequence)


            #print('c = ', c)
            for i in range(len_f_array):
                #print('modulo(f) = ', modulo(f_array[i]))
                #print('z_tr = ', z_threshold)
                if modulo(f_array[i]) < z_threshold:
                    mandelbrot_set_array[i].append(c)
            #print(mandelbrot_set_array)

        return mandelbrot_set_array

def calculate_area(left_bound, right_bound, bottom_bound, top_bound, mand_set_length, sample_length):
    S_rect = abs(right_bound - left_bound) * abs(top_bound - bottom_bound)
    return mand_set_length /sample_length * S_rect

def complex_random_array(length, left_bound = -2, right_bound = 1, bottom_bound = -1, top_bound = 1):
    c_shape = (1, length)
    left_bound, right_bound = -2, 1
    bottom_bound, top_bound = -1, 1
    c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]

    return c_arr





""" *** Test Section *** """

def test_convergence(sample_size, num_samples, iterations, iteration_step):
    area_array = np.zeros((num_samples))
    for i in range(num_samples):
        c_shape = (1, sample_size)
        left_bound, right_bound = -2, 1
        bottom_bound, top_bound = -1, 1
        c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]

        mand_set = generate_mandelbrot(0, c_arr, iterations, iteration_step, 2, sequence=False, heights=False)
        area = calculate_area(left_bound, right_bound, bottom_bound, top_bound, len(mand_set), len(c_arr))
        area_array[i] = area

    # plt.plot([i for i in range(num_samples)], area_array)
    # plt.show()

    return sample_size, np.mean(area_array), np.std(area_array)


def iterations_vs_no_samples(sample_size, num_samples_list, iterations_list):
    area_array = np.zeros((len(num_samples_list), len(iterations_list)))
    area_std_array = np.zeros((len(num_samples_list), len(iterations_list)))

    for i, num_samples in enumerate(num_samples_list):
        for j, num_iterations in enumerate(iterations_list):
            area_temp = np.zeros((num_samples))
            for k in range(num_samples):
                c_shape = (1, sample_size)
                left_bound, right_bound = -2, 1
                bottom_bound, top_bound = -1, 1
                c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]
                mand_set = generate_mandelbrot(0, c_arr, num_iterations, 1, 2, sequence=False, heights=False)
                area_temp[k] = calculate_area(left_bound, right_bound, bottom_bound, top_bound, len(mand_set), len(c_arr))
            area_mean = np.mean(area_temp)
            area_std = np.std(area_temp)
            area_array[i][j] = area_mean
            area_std_array[i][j] = area_std
    return area_array, area_std_array
    

def plot_mandelbrot(num_points, iterations):
    c_shape = (1, num_points)
    left_bound, right_bound = -2, 1
    bottom_bound, top_bound = -1, 1
    c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]

    mand_set, colormap_complex = generate_mandelbrot(0, c_arr, iterations, 10, 2, sequence=False, heights=True)

    x = [elem.real for elem in mand_set]
    y = [elem.imag for elem in mand_set]
    colormap = [modulo(elem) for elem in colormap_complex]

    print(len(c_arr))
    print(len(mand_set))

    S_rect = abs(right_bound - left_bound) * abs(top_bound - bottom_bound)
    print('area = ', len(mand_set) / len(c_arr) * S_rect)


    cmap = plt.get_cmap('viridis')
    colors = cmap(colormap)
    # plt.title(r'Mandelbrot set generated by uniform random sampling \n with $N = 10^{%i}$ points'%(int(np.log10(num_points))))
    plt.title(r'Mandelbrot set generated by uniform random sampling' + '\n' + r'with $N = 10^{%i}$ points, and %i iterations'%(int(np.log10(num_points)), iterations), multialignment='center')

    plt.scatter(x, y, c=colors, s=0.1)
    plt.xlabel(r'$Re[c]$')
    plt.ylabel(r'$Im[c]$')
    plt.savefig(f'mandelbrot_N=10^({int(np.log10(num_points))}).png', dpi=200)
    plt.show()

def intermediate_iterations_test():
    """
    Doesn't work for different iterations and intermediate_steps values, but for some
    seems to be correct
    """

    sample_size = int(1E5)
    c_arr = complex_random_array(length = sample_size)

    iterations = 300
    intermediate_steps = 30
    result = generate_mandelbrot(0, c_arr, iterations, intermediate_steps, 2, True, False)

    lengths = [len(x) for x in result]
    #x = [i * intermediate_steps for i in range(iterations // intermediate_steps)]
    plt.plot(lengths[1:])
    plt.xlabel('iteration / 10')
    plt.ylabel('# of points in generated set')

    areas = [calculate_area(-2, 1, -1, 1, len(x), sample_size) for x in result]

    plt.figure()
    plt.xlabel('iteration / 10')
    plt.ylabel('area')
    plt.plot(areas[1:])

    plt.figure()
    plt.xlabel('iteration / 10')
    plt.ylabel('|dA|')
    plt.plot(abs(np.diff(areas)))

    plt.show()

def area_vs_sample_size(sample_sizes, repititions, iterations, iteration_step):
    area_list = []
    area_std_list = []
    for sample_size in sample_sizes:
        area_array = np.zeros(repititions)
        for i in range(repititions):
            c_shape = (1, int(sample_size))
            left_bound, right_bound = -2, 1
            bottom_bound, top_bound = -1, 1
            c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]

            mand_set = generate_mandelbrot(0, c_arr, iterations, iteration_step, 2, sequence=False, heights=False)
            area = calculate_area(left_bound, right_bound, bottom_bound, top_bound, len(mand_set), len(c_arr))
            area_array[i] = area

        area_list.append(np.mean(area_array))
        area_std_list.append(np.std(area_array))
    
    plt.plot(sample_sizes, area_list, 'ko')
    plt.plot(sample_sizes, area_list, 'k-')
    # plt.errorbar(sample_sizes, area_list, yerr = area_std_list)
    plt.fill_between(sample_sizes, np.array(area_list) - np.array(area_std_list), np.array(area_list) + np.array(area_std_list), color='k', alpha=0.2)
    


    plt.xlabel('S')
    plt.ylabel('A')
    plt.show()

def main():
    # intermediate_iterations_test()
    # num_samples_list = [i for i in range(10, 101, 20)]
    # iterations_list = [i for i in range(10, 101, 20)]

    # area_array, area_std_array = iterations_vs_no_samples(int(1E4), num_samples_list, iterations_list)

    # # Calculate step sizes
    # sample_step = num_samples_list[1] - num_samples_list[0]
    # iteration_step = iterations_list[1] - iterations_list[0]

    # plt.imshow(area_array, origin='lower', cmap='Greys', extent=[min(iterations_list)-iteration_step/2, max(iterations_list)+iteration_step/2, min(num_samples_list)-sample_step/2, max(num_samples_list)+sample_step/2])
    # plt.colorbar()
    # plt.xlabel('Iterations')
    # plt.ylabel('Num Samples')
    # plt.title('Area Array')
    # plt.show()

    # # Calculate step sizes
    # sample_step = num_samples_list[1] - num_samples_list[0]
    # iteration_step = iterations_list[1] - iterations_list[0]

    # plt.imshow(area_std_array, origin='lower', cmap='Greys', extent=[min(iterations_list)-iteration_step/2, max(iterations_list)+iteration_step/2, min(num_samples_list)-sample_step/2, max(num_samples_list)+sample_step/2])
    # plt.colorbar()
    # plt.xlabel('Iterations')
    # plt.ylabel('Num Samples')
    # plt.title(r'$\sigma(A)$')
    # plt.show()

    # plot_mandelbrot(int(1E6), 500)
    sample_sizes = np.linspace(int(1E3), int(1E5), num=10)
    area_vs_sample_size(sample_sizes, 30, 100, 2)





if __name__ == '__main__':
    main()




