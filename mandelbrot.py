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
    

def plot_mandelbrot():
    c_shape = (1, int(1E5))
    left_bound, right_bound = -2, 1
    bottom_bound, top_bound = -1, 1
    c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]

    mand_set, colormap_complex = generate_mandelbrot(0, c_arr, 100, 10, 4, sequence=False, heights=True)

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

def intermediate_iterations_test():
    """
    Doesn't work for different iterations and intermediate_steps values, but for 100, 10 
    seems to be correct
    """

    c_arr = complex_random_array(length = int(1E5))

    iterations = 100
    intermediate_steps = 10
    result = generate_mandelbrot(0, c_arr, iterations, intermediate_steps, 2, True, False)

    lengths = [len(x) for x in result]
    #x = [i * intermediate_steps for i in range(iterations // intermediate_steps)]
    plt.plot(lengths[1:])
    plt.xlabel('iteration')
    plt.ylabel('# of points in generated set')
    plt.show()

    print(lengths)

def main():
    # intermediate_iterations_test()
    num_samples_list = [i for i in range(10, 101, 20)]
    iterations_list = [i for i in range(10, 101, 20)]

    area_array, area_std_array = iterations_vs_no_samples(int(1E4), num_samples_list, iterations_list)

    # Calculate step sizes
    sample_step = num_samples_list[1] - num_samples_list[0]
    iteration_step = iterations_list[1] - iterations_list[0]

    plt.imshow(area_array, origin='lower', cmap='Greys', extent=[min(iterations_list)-iteration_step/2, max(iterations_list)+iteration_step/2, min(num_samples_list)-sample_step/2, max(num_samples_list)+sample_step/2])
    plt.colorbar()
    plt.xlabel('Iterations')
    plt.ylabel('Num Samples')
    plt.title('Area Array')
    plt.show()

    # Calculate step sizes
    sample_step = num_samples_list[1] - num_samples_list[0]
    iteration_step = iterations_list[1] - iterations_list[0]

    plt.imshow(area_std_array, origin='lower', cmap='Greys', extent=[min(iterations_list)-iteration_step/2, max(iterations_list)+iteration_step/2, min(num_samples_list)-sample_step/2, max(num_samples_list)+sample_step/2])
    plt.colorbar()
    plt.xlabel('Iterations')
    plt.ylabel('Num Samples')
    plt.title(r'$\sigma(A)$')
    plt.show()





if __name__ == '__main__':
    main()


# data2 = test_convergence(1000, 100, 100, 10)
# data3 = test_convergence(2000, 100, 100, 10)
# data4 = test_convergence(5000, 100, 100, 10)
# data5 = test_convergence(10000, 100, 100, 10)
# data6 = test_convergence(15000, 100, 100, 10)
# data7 = test_convergence(20000, 100, 100, 10)
# data8 = test_convergence(50000, 100, 100, 10)
# data9 = test_convergence(100000, 100, 100, 10)



# # convergence_data = [data2, data3, data4]

# convergence_data = [data2, data3, data4, data5, data6, data7, data8, data9]

# for data in convergence_data:
#     sample_size = data[0]
#     area_mean = data[1]
#     area_std = data[2]
#     plt.plot(sample_size, area_mean, 'bo', label=f'N = {sample_size}')
#     plt.errorbar(sample_size, area_mean, yerr = area_std)

# plt.xlabel('N')
# plt.ylabel('A')
# plt.show()

