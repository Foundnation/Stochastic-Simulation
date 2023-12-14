import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

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

def load_graph(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    cities = []
    for line in lines[6:]:
        data = line.split()
        if len(data) == 3:
            cities.append(tuple(map(float, data[1:])))

    return cities

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

    new_state[index1:index2+1] = reversed(new_state[index1:index2+1])

    return new_state

def generate_neighbor(current_state, method):
    if method == 'swap':
        new_state = swap(current_state)
    if method == 'reverse':
        new_state = reverse(current_state)
    
    return new_state

def cool(current_temp, alpha, method = 'exponential'):
    """
    implementation of cooling schedules
    alpha - cooling rate
    return: temperature after a cooling step
    """
    if method == 'exponential':
        new_temp = current_temp * alpha    

    return new_temp

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

def perform_annealing(distances, altering_method = 'swap', initial_temp=10000, alpha=0.999, max_iterations=int(1E4), init_tour = None, final_temp = 1E-6):
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

    #random.shuffle(current_state)

    # practically a cost variable, longer tour -> higher energy
    current_energy = total_tour_distance(current_tour, distances)
    
    best_tour = current_tour.copy()
    best_energy = current_energy 

    temperature = initial_temp
    cost_over_iterations = []
    temperature_over_iterations = []

    for _ in range(max_iterations):
        count += 1
        # if temperature < final_temp:
        #     print('temperature reached')
        #     print(count)
        #     break

        new_tour = generate_neighbor(current_tour, altering_method)

        new_energy = total_tour_distance(new_tour, distances)
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

        temperature = cool(temperature, alpha)
        temperature_over_iterations.append(temperature)
        cost_over_iterations.append(best_energy)

    return best_tour, best_energy, cost_over_iterations, temperature_over_iterations

def estimate_conf_interval(data, conf_level=0.95):
    return stats.t.interval(conf_level, len(data) - 1, loc=stats.describe(data).mean, scale=stats.sem(data))


def plot_tour(tour, cities):
    """
    requires calling plt.show() after
    """
    num_cities = len(tour)
    tour_x = [cities[tour[i]][0] for i in range(num_cities)]
    tour_y = [cities[tour[i]][1] for i in range(num_cities)]
    plt.figure(figsize=(8, 6))
    plt.scatter([city[0] for city in cities], [city[1] for city in cities])
    plt.plot(tour_x + [tour_x[0]], tour_y + [tour_y[0]], 'o-')
    plt.title("TSP Solution")
    plt.xlabel("X")
    plt.ylabel("Y")


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

def run_vary_maxiter(num_runs, distances, max_iterations_list, **kwargs):
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

    return means, stds, conf_intervals


def main():
    """ testing """
    import os
    script_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_directory, 'TSP-Configurations/eil51.tsp.txt')

    cities = load_graph(filepath)
    distances = calculate_distances(cities)
    # _ , _, _, _  = perform_annealing(distances=distances, final_temp = 1)
    #result = run_simulations(num_runs = 50, distances=distances, output='fitness_statistics')
    
    results = run_vary_maxiter(20, distances, [100, 1000, 10000])
    print(results)

if __name__ == '__main__':
    main()