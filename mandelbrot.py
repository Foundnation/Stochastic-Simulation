import numpy as np
import matplotlib.pyplot as plt
from pyDOE2 import lhs
import scipy.stats as stats
from ortho_sampling import init_genrand, random_orthogonal_sampling

def mandelbrot_func(z0, c, iterations, num_intermediate_steps, sequence=False):
    k = 0
    z = [z0]
    while k < iterations:
        z.append(z[k] * z[k] + c)
        k += 1

    z_bins = []
    if sequence:
        z_bins_length = len(z) // num_intermediate_steps
        for i in range(num_intermediate_steps):
            z_bins.append(z[i * z_bins_length] - 1)
        return z_bins
    else:
        return z[-1]

def modulo(z):
    return np.sqrt((z.real)*(z.real) + (z.imag)*(z.imag))

def generate_mandelbrot(z0, c_array, iterations, z_threshold, num_intermediate_steps = 0, sequence = False, heights = False):
    mandelbrot_set = []
    heightmap = []

    if sequence == False and heights == True:
        for c in c_array:
            f = mandelbrot_func(z0, c, iterations, num_intermediate_steps, sequence)
            if modulo(f) < z_threshold:
                mandelbrot_set.append(c)
                heightmap.append(f)
        
        return mandelbrot_set, heightmap
    
    elif  sequence == False and heights == False:
        for c in c_array:
            f = mandelbrot_func(z0, c, iterations, num_intermediate_steps, sequence)
            if modulo(f) < z_threshold:
                mandelbrot_set.append(c)
        return mandelbrot_set

    elif sequence == True and heights == False:
        mandelbrot_set_array = []
        len_f_array = num_intermediate_steps
        for i in range(len_f_array):
            mandelbrot_set_array.append([])

        for c in c_array:
            f_array = mandelbrot_func(z0, c, iterations, num_intermediate_steps, sequence)

            for i in range(len_f_array):
                if modulo(f_array[i]) < z_threshold:
                    mandelbrot_set_array[i].append(c)

        return mandelbrot_set_array

def calculate_area(mand_set_length, sample_length, circle_sample_length = 0, circle_radius = 0.3, 
                   left_bound = -2, right_bound = 1, bottom_bound = -1, top_bound = 1):
    
    S_rect = abs(right_bound - left_bound) * abs(top_bound - bottom_bound)
    if circle_sample_length <= 0:
        return mand_set_length / sample_length * S_rect
    
    elif circle_sample_length > 0:
        S_circ = np.pi * circle_radius * circle_radius
        return (mand_set_length + circle_sample_length) / (sample_length + circle_sample_length) * (S_rect)


def cutout_sampling(length, left_bound=-2, right_bound=1, bottom_bound=-1, top_bound=1, circle_center=(-0.25, 0), circle_radius=0.3):
    uniform_sample = np.random.uniform(low=[left_bound, bottom_bound], high=[right_bound, top_bound], size=(length, 2))

    c_arr = np.vectorize(complex)(uniform_sample[:, 0], uniform_sample[:, 1])

    circle_x, circle_y = circle_center
    circle_radius_sq = circle_radius * circle_radius

    # number of points initially generated inside the circle
    initial_inside_circle = np.sum((uniform_sample[:, 0] - circle_x) ** 2 + (uniform_sample[:, 1] - circle_y) ** 2 <= circle_radius_sq)

    # reject points inside the circle
    inside_circle = (uniform_sample[:, 0] - circle_x) ** 2 + (uniform_sample[:, 1] - circle_y) ** 2 <= circle_radius_sq

    # keep regenerating points inside the rectangle until the required number outside the circle is achieved
    while np.sum(inside_circle) > 0:
        uniform_sample[inside_circle] = np.random.uniform(low=[left_bound, bottom_bound], high=[right_bound, top_bound], size=(np.sum(inside_circle), 2))

        c_arr[inside_circle] = np.vectorize(complex)(uniform_sample[inside_circle, 0], uniform_sample[inside_circle, 1])

        inside_circle = (uniform_sample[:, 0] - circle_x) ** 2 + (uniform_sample[:, 1] - circle_y) ** 2 <= circle_radius_sq

    return c_arr, initial_inside_circle



def complex_random_array(length, method = 'uniform', left_bound = -2, right_bound = 1, bottom_bound = -1, top_bound = 1):
    if method == 'uniform':
        c_shape = (1, length)
        c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]
        return c_arr
    
    elif method == 'lhs':
        # generate a Latin Hypercube Sample of real numbers in the interval [0, 1)
        lhs_sample = lhs(2, length)

        dimension_ranges = [(left_bound, right_bound), (bottom_bound, top_bound)] 
        scaled_lhs_samples = np.zeros((length, 2))
        for i in range(2):
            scaled_lhs_samples[:, i] = dimension_ranges[i][0] + lhs_sample[:, i] * (dimension_ranges[i][1] - dimension_ranges[i][0])

        c_arr = np.vectorize(complex)(scaled_lhs_samples[:, 0], scaled_lhs_samples[:, 1])
        return c_arr
    
    elif method == 'orthogonal' or method == 'ortho':
        ortho_sample = random_orthogonal_sampling(int(np.sqrt(length)), 1)
        return ortho_sample
    
    elif method == 'custom' or method == 'reduced_domain' or method == 'cutout':
        custom_sample, num_points_inside_circle = cutout_sampling(length)
        return custom_sample, num_points_inside_circle

def estimate_confidence_interval(data, confidence_level):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    num_samples = len(data)

    degrees_of_freedom = num_samples - 1

    # critical value from the t-distribution
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

    standard_error = sample_std / np.sqrt(num_samples)
    margin_of_error = t_critical * standard_error

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return [lower_bound, upper_bound]
    
def estimate_area(sample_size, num_runs, iterations, iteration_step, method='uniform', output_area_array = False):
    area_array = np.zeros(num_runs)
    if method == 'cutout' or method == 'custom' or method == 'reduced_domain':
        for i in range(num_runs):
            c_arr, num_points_inside_circle = complex_random_array(sample_size, method=method)

            mand_set = generate_mandelbrot(z0=0, c_array=c_arr, iterations=iterations, z_threshold=2,
                                            num_intermediate_steps=iteration_step, sequence=False, heights=False)
            
            area = calculate_area(len(mand_set), len(c_arr), circle_sample_length = num_points_inside_circle)
            area_array[i] = area
    else:
        for i in range(num_runs):
            c_arr = complex_random_array(sample_size, method=method)

            mand_set = generate_mandelbrot(z0=0, c_array=c_arr, iterations=iterations, z_threshold=2,
                                            num_intermediate_steps=iteration_step, sequence=False, heights=False)
            area = calculate_area(len(mand_set), len(c_arr))
            area_array[i] = area

    confidence_interval = estimate_confidence_interval(area_array, 0.95)

    if output_area_array:
        return np.mean(area_array), np.std(area_array), confidence_interval, area_array
    else:
        return np.mean(area_array), np.std(area_array), confidence_interval



""" *** Test Section *** """

def plot_mandelbrot(num_points, iterations):
    c_shape = (1, num_points)
    left_bound, right_bound = -2, 1
    bottom_bound, top_bound = -1, 1
    c_arr = (np.random.uniform(left_bound, right_bound, c_shape) + 1.j * np.random.uniform(bottom_bound, top_bound, c_shape))[0]

    mand_set, colormap_complex = generate_mandelbrot(0, c_arr, iterations, 2, sequence=False, heights=True)

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

def intermediate_iterations(sample_size, iterations, intermediate_steps, repetitions, method = 'uniform'):
    """
    Doesn't work for different iterations and intermediate_steps values, but for some
    seems to be correct
    """
    num_points_inside_circle = 0

    if method == 'cutout':
        c_arr, num_points_inside_circle = complex_random_array(length = sample_size, method=method)
    else:
        c_arr = complex_random_array(length = sample_size, method=method)

    mand_sets_array = []
    for i in range(repetitions):
        temp = generate_mandelbrot(0, c_arr, iterations, 2, num_intermediate_steps=intermediate_steps, sequence=True)
        mand_sets_array.append(temp)
    
    # calculate average number of points in Mandelbrot set
    # for each specified number of iterations
    lengths = [[] for i in range(len(mand_sets_array))]
    for i, mand_sets in enumerate(mand_sets_array):
        for x in mand_sets:
            lengths[i].append(len(x))

    averages = [0] * len(lengths[0])

    for run in lengths:
        for idx, val in enumerate(run):
            averages[idx] += val

    average_lengths = [avg / repetitions for avg in averages]

    # calculate average estimation of area of Mandelbrot set
    # for each specified number of iterations
    areas = [[] for i in range(len(mand_sets_array))]
    for i, mand_sets in enumerate(mand_sets_array):
        for x in mand_sets:
            areas[i].append(calculate_area(len(x), sample_size, circle_sample_length=num_points_inside_circle))

    average_temp = [0] * len(lengths[0])

    for run in areas:
        for idx, val in enumerate(run):
            average_temp[idx] += val

    average_areas = [avg / repetitions for avg in average_temp]
    
    return average_areas, average_lengths


def calculate_error_over_iterations(best_estimation, average_areas):
    errors = [x - best_estimation for x in average_areas]
    return errors


def area_vs_sample_size(sample_sizes, repititions, iterations, iteration_step, method):
    area_list = []
    area_std_list = []
    for sample_size in sample_sizes:
        area_array = np.zeros(repititions)
        for i in range(repititions):
            if method == 'cutout':
                c_arr, num_points_inside_circle = complex_random_array(sample_size, method=method)
                mand_set = generate_mandelbrot(0, c_arr, iterations, iteration_step, 2, sequence=False, heights=False)
                area = calculate_area(len(mand_set), len(c_arr), circle_sample_length=num_points_inside_circle)
            else:
                c_arr = complex_random_array(sample_size, method=method)
                mand_set = generate_mandelbrot(0, c_arr, iterations, iteration_step, 2, sequence=False, heights=False)
                area = calculate_area(len(mand_set), len(c_arr))

            area_array[i] = area

        area_list.append(np.mean(area_array))
        area_std_list.append(np.std(area_array))

    return area_list, area_std_list

def plot_area_vs_sample_size(sample_sizes, repetitions, iterations, iteration_step, methods):

    areas_with_methods = []
    areas_std_with_methods = []
    for method in methods:
        temp_area, temp_std = area_vs_sample_size(sample_sizes, repetitions, iterations, iteration_step, method)
        areas_with_methods.append(temp_area)
        areas_std_with_methods.append(temp_std)

    colors = [('ko', 'k-', 'k'), ('bo', 'b-', 'b'), ('ro', 'r-', 'r'), (('go', 'g-', 'g'))]
    for i in range(len(areas_with_methods)):
        plt.plot(sample_sizes, areas_with_methods[i], colors[i][0])
        plt.plot(sample_sizes, areas_with_methods[i], colors[i][1])
        # plt.errorbar(sample_sizes, area_list, yerr = area_std_list)
        plt.fill_between(sample_sizes, np.array(areas_with_methods[i]) - np.array(areas_std_with_methods[i]),
                          np.array(areas_with_methods[i]) + np.array(areas_std_with_methods[i]), color=colors[i][2], alpha=0.15)

    plt.xlabel('S')
    plt.ylabel('A')
    plt.show()

    return areas_with_methods, areas_std_with_methods


def iterations_vs_num_runs(sample_size, num_samples_list, iterations_list):
    area_array = np.zeros((len(num_samples_list), len(iterations_list)))
    area_std_array = np.zeros((len(num_samples_list), len(iterations_list)))

    for i, num_samples in enumerate(num_samples_list):
        for j, num_iterations in enumerate(iterations_list):
            
            area_temp = np.zeros((num_samples))
            for k in range(num_samples):
                c_arr = complex_random_array(sample_size)
                mand_set = generate_mandelbrot(0, c_arr, num_iterations, 1, 2, sequence=False, heights=False)
                area_temp[k] = calculate_area(len(mand_set), len(c_arr))

            area_mean = np.mean(area_temp)
            area_std = np.std(area_temp)
            area_array[i][j] = area_mean
            area_std_array[i][j] = area_std

    return area_array, area_std_array
    

def plot_runs_iterations_interaction():
    num_runs_list = [2, 3, 10, 20, 30, 50, 60, 70] #[i for i in range(1, 50, 10)]
    iterations_list = [i for i in range(5, 126, 20)][1:]

    area_array, area_std_array = iterations_vs_num_runs(int(1E4), num_runs_list, iterations_list)

    # Calculate step sizes
    sample_step = num_runs_list[1] - num_runs_list[0]
    iteration_step = iterations_list[1] - iterations_list[0]

    plt.imshow(area_array, origin='lower', cmap='Greys',) #extent=[min(iterations_list)-iteration_step/2, max(iterations_list)+iteration_step/2,
                                                          #        min(num_samples_list)-sample_step/2, max(num_samples_list)+sample_step/2])
    plt.colorbar()
    plt.xlabel('Iterations')
    plt.ylabel('Num runs')
    plt.title('Mean area dependence on i and N')
    plt.show()

    # Calculate step sizes
    sample_step = num_runs_list[1] - num_runs_list[0]
    iteration_step = iterations_list[1] - iterations_list[0]

    plt.imshow(area_std_array, origin='lower', cmap='Greys',) #extent=[min(iterations_list)-iteration_step/2, max(iterations_list)+iteration_step/2, 
                                                              #        min(num_samples_list)-sample_step/2, max(num_samples_list)+sample_step/2])
    plt.colorbar()
    plt.xlabel('Iterations')
    plt.ylabel('Num runs')
    plt.title(r'$\sigma(A)$')
    plt.show()



def main():

    res1 = estimate_area(sample_size=int(1E4), num_runs=10, iterations=100, iteration_step=10, method='cutout')
    res2 = estimate_area(sample_size=int(1E4), num_runs=10, iterations=100, iteration_step=10, method='uniform')

    print(res1, '\n', res2)

    # cutout_sample = complex_random_array(100000, method='cutout')
    # print(cutout_sample)
    # plt.scatter(cutout_sample.real, cutout_sample.imag, label="LHS Samples", s=1)
    # plt.show()

    # plot_runs_iterations_interaction()

    #sample_sizes = np.linspace(int(1E2), int(1E3), num=10).astype(int)
    #plot_area_vs_sample_size(sample_sizes, 10, 100, 2, methods=['uniform', 'lhs'])


    #print(calculate_error_over_iterations(best_estimation=1.527021, sample_size=int(1E4), iterations=100, iteration_step=10, num_runs=10))

    #sample_sizes = np.linspace(int(1E3), int(1E5), num=10)
    #area_vs_sample_size(sample_sizes, 30, 100, 2)





if __name__ == '__main__':
    main()




