import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import csv
import multiprocessing
import os
import time


###---------------------------- DATA LOADING AND SAVING ----------------------------------------
def load_graph(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    cities = []
    for line in lines[6:]:
        data = line.split()
        if len(data) == 3:
            cities.append(tuple(map(float, data[1:])))

    return cities

def save_data_old(*args, file_path, column_names=None, header=None):
    data = zip(*args)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if header is not None:
            writer.writerow(header)
        if column_names is not None:
            writer.writerow(column_names)

        writer.writerows(data)

def save_data(data_list, file_path, column_names=None, header=None):
    data = zip(*data_list)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if header is not None:
            writer.writerow([header])
        if column_names is not None:
            writer.writerow(column_names)

        writer.writerows(data)
###-----------------------------------------------------------------------------------------------

###------------------------ DISTANCE CALCULATION -------------------------------------------------
def calculate_distances(cities):
    """
    returns list of distances between cities (graph weights)
    """
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = np.sqrt((cities[i][0] - cities[j][0]) * (cities[i][0] - cities[j][0]) +
                                  (cities[i][1] - cities[j][1]) * (cities[i][1] - cities[j][1]))
            
            distances[i][j] = distances[j][i] = distance

    return distances

def total_tour_distance(tour, distances):
    total = 0
    num_cities = len(tour)
    for i in range(num_cities - 1):
        total += distances[tour[i]][tour[i + 1]]

    total += distances[tour[num_cities - 1]][tour[0]]

    return total

def calculate_optimal_distances():
    script_directory = os.path.dirname(os.path.abspath(__file__))

    local_paths = ['TSP-Configurations/eil51.tsp.txt', 'TSP-Configurations/a280.tsp.txt', 'TSP-Configurations/pcb442.tsp.txt']
    filepaths = [os.path.join(script_directory, path) for path in local_paths]
    
    local_paths_opt = ['TSP-Configurations/eil51.opt.tour.txt', 'TSP-Configurations/a280.opt.tour.txt', 'TSP-Configurations/pcb442.opt.tour.txt']
    filepaths_opt = [os.path.join(script_directory, path) for path in local_paths_opt]
    results = []

    for j in range(len(filepaths_opt)):
        with open(filepaths_opt[j], 'r') as file:
            lines = file.readlines()
        
        tour = lines[5:(len(lines) - 1)]

        for i in range(len(tour) - 1):
            tour[i] = int(tour[i].split()[0]) - 1
        tour[-1] = -1

        cities = load_graph(filepaths[j])
        distances = calculate_distances(cities)
        tour_distance = total_tour_distance(tour, distances)

        results.append(tour_distance)

    return results


###------------------------ PERMUTATION OPERATORS ----------------------------------------
def swap(current_state):
    """
    takes a tour city sequence (list) and swaps two randomly chosen cities
    """
    new_state = current_state.copy()
    index1 = random.randint(0, len(current_state) - 1)
    index2 = random.randint(0, len(current_state) - 1)
    new_state[index1], new_state[index2] = new_state[index2], new_state[index1]
    
    return new_state

def reverse(current_state):
    """
    generates two random indices and reverses the order of all the cities in between
    """
    new_state = current_state.copy()
    index1 = random.randint(0, len(current_state) - 1)
    index2 = random.randint(0, len(current_state) - 1)

    while index2 == index1:
        index2 = random.randint(0, len(current_state) - 1)

    index1, index2 = min(index1, index2), max(index1, index2)

    new_state[index1:index2+1] = list(reversed(new_state[index1:index2+1]))

    return new_state

def insert(current_state):
    """
    inserts a single random node in another random place and removes it from where it was originally
    """
    new_state = current_state.copy()
    if isinstance(current_state, np.ndarray):
        new_state = new_state.tolist()

    index1 = random.randint(0, len(current_state) - 1)
    index2 = random.randint(0, len(current_state) - 1)

    while index2 == index1:
        index2 = random.randint(0, len(current_state) - 1)

    value_to_insert = new_state[index2]
    new_state.insert(index1, value_to_insert)

    # Remove the value at index2 from its original position
    if index2 < index1:
        del new_state[index2]
    else:
        del new_state[index2 + 1]

    return new_state
###--------------------------------------------------------------------------------------------------------

###----------------- ADDITIONAL FUNCTIONS -----------------------------------------------------------------
def tour_to_cities(tour, cities):
    rearranged_cities = [cities[i] for i in tour]
    return rearranged_cities


def count_intersections(cities):
    """
    Count the number of intersecting paths in the tour
    note: Cities are the coordinates of the best_tour_list
    This can be converted by plugging the original  cities and best_tour
    into the function above: tour_to_cities
    """

    def ccw(A, B, C):
        """Check if three points are in a counterclockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A, B, C, D):
        """Check if two line segments AB and CD intersect."""
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    n = len(cities)
    intersect_count = 0

    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if i != 0 or j != n - 1:
                # Avoid checking consecutive edges in the tour
                if intersect(cities[i], cities[i + 1], cities[j], cities[j + 1]):
                    intersect_count += 1

    return intersect_count

#### IS IT NEEDED ?
def two_opt(current_state, distances):
    """
    Apply the 2-opt algorithm to improve the tour.
    """
    improved = True
    while improved:
        improved = False
        for i in range(len(current_state) - 1):
            for j in range(i + 1, len(current_state)):
                if j - i == 1:
                    continue  # Skip adjacent cities
                new_tour = current_state[:i] + current_state[i:j][::-1] + current_state[j:]
                if total_tour_distance(new_tour, distances) < total_tour_distance(current_state, distances):
                    current_state = new_tour
                    improved = True
                    break  # Start over if improvement is found
            if improved:
                break  # Start over if improvement is found
    return current_state
    
###-------------------------------------------------------------------------------------------------

###--------------------------------- Statistics ----------------------------------------------------
def estimate_conf_interval(data, conf_level=0.95):
    return stats.t.interval(conf_level, len(data) - 1, loc=stats.describe(data).mean, scale=stats.sem(data))

def perform_tests():
    return

###-------------------------------------------------------------------------------------------------

###---------------- SIMULATED ANNEALING ------------------------------------------------------------
def generate_neighbor(current_state, method):
    if method == 'swap':
        new_state = swap(current_state)
    if method == 'reverse':
        new_state = reverse(current_state)
    if method == 'insert':
        new_state = insert(current_state)
    
    return new_state

def cool(current_temp, alpha, method, current_step, max_iter, t_max, t_min):
    """
    implementation of cooling schedules
    alpha - cooling rate
    current_step - current step in the cycle 
    return: temperature after a cooling step
    t_max = initial temperature
    t_min = final temperature
    """

    if method not in ['linear_m', 'linear_a', 'quadratic_a', 'quadratic_m', 
                         'exponential_m', 'dynamic_m', 'logarithmic_m']:
        raise Exception(f'cooling schedule {method} is not implemented')    
    
        
    if method == 'linear_m':
        # linear multiplicative cooling
        new_temp = t_max / (1 + alpha * current_step)
        
    if method == 'linear_a':
        # linear additive cooling
        new_temp = t_min + (t_max - t_min) * ((max_iter - current_step)/max_iter)
        
    if method == 'quadratic_m':
        # quadratic multiplicative cooling
        new_temp = t_min / (1 + alpha * current_step**2)
        
    if method == 'quadratic_a':
        # quadratic additive cooling
        new_temp = t_min + (t_max - t_min) * ((max_iter - current_step)/max_iter)**2
        
    if method == 'exponential_m':
        # exponential multiplicative cooling
        new_temp = t_max * alpha**current_step
    
    if method == 'dynamic_m':
        new_temp = current_temp * alpha**current_step

    if method == 'logarithmic_m':
        # logarithmical multiplicative cooling
        new_temp = t_max / (1+alpha * np.log(current_step + 1))
   
    return new_temp

# CURRENTLY NOT IMPLEMENTED
def accept_reject(new_energy, current_energy, new_tour, temperature, current_tour):
    energy_difference = new_energy - current_energy

    if energy_difference < 0 or random.random() < np.exp(-energy_difference / temperature):
        current_tour = new_tour
        current_energy = new_energy

        best_tour = current_tour
        best_energy = current_energy
        
        # # a possible improvement
        # if current_energy < best_energy:
        #     best_tour = current_tour.copy()
        #     best_energy = current_energy
        return best_tour, best_energy
    else:
        return current_tour, current_energy

def perform_annealing(distances, altering_method = 'reverse', cooling_schedule = 'exponential_m', 
                      initial_temp=10000, alpha=0.999, max_iterations=int(1E4), final_temp = 1E-7,
                     chain_length = 1, init_tour = None, output_count = False, mesa = False):
    """
    Optimizes tour length using simulated annealing.
    altering_method -  determines how tour will be changed at each iteration
    init_tour       -  initial order of visiting cities   
    """
    count = 0

    num_cities = len(distances)
    if init_tour == None:
        current_tour = np.arange(num_cities)
    else:
        current_tour = init_tour

    # random.shuffle(current_tour)

    # practically a cost variable, longer tour -> higher energy
    current_energy = total_tour_distance(current_tour, distances)
    
    best_tour = current_tour.copy()
    best_energy = current_energy 

    temperature = initial_temp
    cost_over_iterations = []
    temperature_over_iterations = []

    for k in range(max_iterations):
        count += 1
        if temperature < final_temp:
            # print('final temperature reached')
            # print(f'iterations: {count}')
            break
        
        # explore landscape at this temperature for # of iterations = chain_length
        for _ in range(chain_length):
            new_tour = generate_neighbor(current_tour, altering_method)
            new_energy = total_tour_distance(new_tour, distances)
            energy_difference = new_energy - current_energy

            # apply accept-reject condition according to Metropolis-Hastings
            if energy_difference < 0 or random.random() < np.exp(-energy_difference / temperature):
                current_tour = new_tour
                current_energy = new_energy
                
                if mesa:
                    if current_energy < best_energy:
                        best_tour = current_tour.copy()
                        best_energy = current_energy
                else:
                    best_tour = current_tour
                    best_energy = current_energy

            temperature = cool(temperature, alpha, method=cooling_schedule, current_step=k, max_iter=max_iterations, 
                            t_max=initial_temp, t_min=final_temp)
            temperature_over_iterations.append(temperature)
            cost_over_iterations.append(best_energy)

    if output_count:
        return best_tour, best_energy, cost_over_iterations, temperature_over_iterations, count
    else:
        return best_tour, best_energy, cost_over_iterations, temperature_over_iterations
    

###-----------------------------------------------------------------------------------------------------------------

###-------------------------------- PLOTTERS -----------------------------------------------------------------------
def plot_tour(tour, cities, permutation_method='swap'):
    """
    requires calling plt.show() after
    """
    num_cities = len(tour)
    tour_x = [cities[tour[i]][0] for i in range(num_cities)]
    tour_y = [cities[tour[i]][1] for i in range(num_cities)]
    plt.figure(figsize=(8, 6))
    plt.scatter([city[0] for city in cities], [city[1] for city in cities])
    plt.plot(tour_x + [tour_x[0]], tour_y + [tour_y[0]], 'o-')
    plt.title("TSP Solution for %s"%(permutation_method))
    plt.xlabel("X")
    plt.ylabel("Y")

def plot_dist_and_temp_local(costs, temps, title=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(costs))
    axs[0].set_title('Distance over iterations')
    axs[0].plot(x, costs)
    axs[1].set_title('Temperature')
    axs[1].plot(x, temps)
    if title is not None:
        fig.suptitle(title)
    plt.show()

###-----------------------------------------------------------------------------------------------------------------

###------------------------------ RUNNERS --------------------------------------------------------------------------

def run_simulations(num_runs, distances, output = 'full', **kwargs):
    """
    calls perform_annealing for num_runs times and outputs chosen data
    **kwargs are optional additional parameters that will be passed to perform_annealing

    """
    tours = []
    fitness_lists = []
    temperatures = []
    final_fitnesses = []
    for i in range(num_runs):
        best_tour, _, fitness, temper = perform_annealing(distances=distances, **kwargs)
        tours.append(best_tour)
        fitness_lists.append(fitness)
        temperatures.append(temper)
        final_fitnesses.append(fitness[-1])
    
    if output == 'full':    
        return tours, fitness_lists, temperatures
    elif output == 'final_fitnesses':
        return final_fitnesses
    elif output == 'fitness_statistics':
        return np.mean(final_fitnesses), np.std(final_fitnesses), estimate_conf_interval(data=final_fitnesses)

def run_vary_maxiter(num_runs, distances, max_iterations_list, save_file_path = None, **kwargs):
    """
    performs annealing for several values of max_iterations_list

    return: mean, std and confidence interval corresponding to each value in max_iterations_list
    """
    means, stds, conf_intervals = [], [], []
    for max_i in max_iterations_list:
        mean, std, conf_interval = run_simulations(num_runs=num_runs, distances=distances,
                                                    max_iterations = max_i, output='fitness_statistics', **kwargs)
        means.append(mean)
        stds.append(std)
        conf_intervals.append(conf_interval)

    if save_file_path is not None:
        save_data(means, stds, conf_intervals, file_path=save_file_path, 
                  column_names=['Mean Distance', 'STD', 'CI'])

    return means, stds, conf_intervals

def run_concurrent(func, param_sets):
    # obtains number of cores and threads (# of cores * 2)
    num_processes = multiprocessing.cpu_count() * 2
    
    start_time = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        # submit tasks to the pool for each parameter set
        results = [pool.apply_async(func, kwds=params) for params in param_sets]
        
        # wait for all processes to finish and save output
        output = [res.get() for res in results]

    end_time = time.time()
    time_taken_concurrency = end_time - start_time

    # print(f"Time taken (conc): {time_taken_concurrency} seconds")

    return output

def run_simulations_concurrent(num_runs, distances, output='fitness_statistics', **kwargs):
    """
    Runs simulations concurrently using the run_concurrent approach.
    """
    param_sets = [{'distances': distances, **kwargs} for _ in range(num_runs)]
    
    result = run_concurrent(perform_annealing, param_sets)

    if output == 'full':    
        tours = [elem[0] for elem in result]
        fitness_lists = [elem[2] for elem in result]
        temperatures = [elem[3] for elem in result]
        return tours, fitness_lists, temperatures
    
    elif output == 'final_fitnesses':
        final_dist_list = [elem[2][-1] for elem in result]
        return final_dist_list
    
    elif output == 'fitness_statistics':
        final_dist_list = [elem[2][-1] for elem in result]
        return np.mean(final_dist_list), np.std(final_dist_list), estimate_conf_interval(data=final_dist_list)
    
def run_vary_maxiter_concurrent(num_runs, distances, max_iterations_list, output='fitness_statistics', save_file_path=None, **kwargs):
    """
    performs annealing for several values of max_iterations_list using concurrency

    return: mean, std and confidence interval corresponding to each value in max_iterations_list
    """
    param_sets = []

    for max_i in max_iterations_list:
        params = {
            'num_runs': num_runs,
            'distances': distances,
            'max_iterations': max_i,
            'output': output
        }
        for key, value in kwargs.items():
            params[key] = value
        param_sets.append(params)
    

    sim_output = run_concurrent(run_simulations, param_sets)

    if output == 'fitness_statistics':
        means = [result[0] for result in sim_output]
        stds = [result[1] for result in sim_output]
        conf_intervals = [result[2] for result in sim_output]

        if save_file_path is not None:
            save_data(means, stds, conf_intervals, file_path=save_file_path, column_names=['Mean Distance', 'STD', 'CI'])

        return means, stds, conf_intervals
    
    elif output == 'final_fitnesses':
        return sim_output

def run_vary_maxiter_concurrent_sims(num_runs, distances, max_iterations_list, output='fitness_statistics', save_file_path=None, **kwargs):
    """
    performs annealing for several values of max_iterations_list using concurrency

    return: mean, std and confidence interval corresponding to each value in max_iterations_list
    """
    start_time = time.time()
    param_sets = []

    for max_i in max_iterations_list:
        params = {
            #'num_runs': num_runs,
            #'distances': distances,
            'max_iterations': max_i,
            'output': output
        }
        for key, value in kwargs.items():
            params[key] = value
        param_sets.append(params)
    
    sim_output = []
    for param in param_sets:
        sim_output.append(run_simulations_concurrent(num_runs, distances, **param))

    #sim_output = run_concurrent(run_simulations, param_sets)

    if output == 'fitness_statistics':
        means = [result[0] for result in sim_output]
        stds = [result[1] for result in sim_output]
        conf_intervals = [result[2] for result in sim_output]

        if save_file_path is not None:
            save_data(means, stds, conf_intervals, file_path=save_file_path, column_names=['Mean Distance', 'STD', 'CI'])

        end_time = time.time()
        print(f'Time taken (conc): {end_time - start_time}')
        return means, stds, conf_intervals
    
    elif output == 'final_fitnesses':
        end_time = time.time()
        print(f'Time taken (conc): {end_time - start_time}')
        return sim_output

def run_vary_schedules(num_runs, distances, schedules, save_file_path=None, **kwargs):
    return

def run_vary_chain_length():
    return

def run_vary_parameter_and_maxiter(num_runs, distances, max_iterations_list, variable_parameter_values: dict, **kwargs):
    results = {}
    for param in variable_parameter_values.get(0):
        output = run_vary_maxiter_concurrent(num_runs, distances, max_iterations_list, param, **kwargs)
        results[param] = output
    
    return results

def wrapper_annealing(**kwargs):
    _ , best_energy, costs, temps, count = perform_annealing(**kwargs, output_count=True)
    schedule = kwargs['cooling_schedule']
    print('cooling schedule: ', schedule)
    print('best_energy:', best_energy)
    print('performed iterations:', count)
    plot_dist_and_temp_local(costs, temps, f'Cooling schedule: {schedule}')

def examine_schedules_dynamics(distances, schedules = 
                               ['linear_m', 'exponential_m', 'logarithmic_m', 'linear_a', 'quadratic_a'], **kwargs):
    kwargs_list = []
    for schedule in schedules:
        params = ({   
            'distances': distances,
            'cooling_schedule': schedule,
            }
        )
        for key, value in kwargs.items():
            params[key] = value
        kwargs_list.append(params)


    output = run_concurrent(wrapper_annealing, kwargs_list)
    
    return output

###-----------------------------------------------------------------------------------------------------------------

##--------------------------------- CODE TESTING ------------------------------------------------------------------------

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_directory, 'TSP-Configurations/a280.tsp.txt')

    cities = load_graph(filepath)
    distances = calculate_distances(cities)

    # _ , best_energy, costs, temps  = perform_annealing(distances=distances, cooling_schedule='linear_m', final_temp=1E-5,
    #                                                    alpha = 1 - 1E-4, chain_length=10, max_iterations=10000)
    # print(best_energy)
    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # axs[0].plot(costs)
    # axs[1].plot(temps)
    # plt.show()

    # result = run_simulations(num_runs = 10, distances=distances, output='fitness_statistics', 
    #                                     altering_method = 'reverse', cooling_schedule = 'linear_m', final_temp=1E-6, alpha=1 - 1E-5)
    # print(result)
    
    # results = run_vary_maxiter(20, distances, [100, 1000, 10000])
    # print()
    
    # filepath_gendata = os.path.join(script_directory, 'generated_data/output_test.csv')
    # results = run_vary_maxiter_concurrent(10, distances, [100, 1000, 10000], save_file_path=filepath_gendata,
    #                                       cooling_schedule = 'linear_m', final_temp=1E-7, alpha=1 - 1E-5)
    # print()

    # examine_schedules_dynamics(distances, max_iterations=10000, final_temp=1E-5, alpha=1 - 1E-5)

    # schedules = ['linear_a', 'linear_m']
    # output = run_vary_schedules(num_runs=20, distances=distances, schedules=schedules, 
    #                     max_iterations=1000, final_temp=1E-5, alpha=1 - 1E-5)
    
    # start_time = time.time()
    # run_simulations(20, distances, output='fitness_statistics', max_iterations=10000, 
    #                     final_temp=1E-4, alpha=1-1E-4, cooling_schedule='linear_m')
    # end_time = time.time()
    # time_taken= end_time - start_time
    # print(f"Time taken with NO concurrency: {time_taken} seconds")

    results1 = run_simulations_concurrent(10, distances, output='fitness_statistics', max_iterations=20000, 
                         final_temp=1E-4, alpha=1-(1E-4), cooling_schedule='linear_m', mesa=False)
    results2 = run_simulations_concurrent(10, distances, output='fitness_statistics', max_iterations=20000, 
                         final_temp=1E-4, alpha=1-(1E-4), cooling_schedule='linear_m', mesa=True)
    print()

    #results = run_vary_maxiter_concurrent(20, distances, [100, 1000, 10000], output='final_fitnesses')

    # start = time.time()
    # results2 = run_vary_maxiter_concurrent_sims(20, distances, [100, 1000, 10000], output='final_fitnesses')
    # end = time.time()
    # print(end - start)


if __name__ == '__main__':
    main()