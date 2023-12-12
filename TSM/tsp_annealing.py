import random
import matplotlib.pyplot as plt
import numpy as np


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
            distances[i][j] = distance
            distances[j][i] = distance

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
    takes a tour city sequence (list) swap two random cities
    """
    new_state = current_state.copy()
    index1 = random.randint(0, len(current_state) - 1)
    index2 = random.randint(0, len(current_state) - 1)
    new_state[index1], new_state[index2] = new_state[index2], new_state[index1]
    
    return new_state

def generate_neighbor(current_state, method = 'swap'):
    if method == 'swap':
        new_state = swap(current_state)
    
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

def perform_annealing(distances, altering_method = 'swap', initial_temp=10000, final_temp=1E-5, alpha=0.999, max_iterations=int(1E7), init_tour = None):
    """
    Optimizes tour length using simulated annealing.
    altering_method -  determines how tour will be changed at each iteration
    init_tour       -  initial order of visiting cities   
    """
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
    fitness_over_iterations = []

    for _ in range(max_iterations):
        if temperature < final_temp:
            break

        new_tour = generate_neighbor(current_tour)

        new_energy = total_tour_distance(new_tour, distances)
        energy_difference = new_energy - current_energy

        if energy_difference < 0 or random.random() < np.exp(-energy_difference / temperature):
            current_tour = new_tour
            current_energy = new_energy

            if current_energy < best_energy:
                best_tour = current_tour.copy()
                best_energy = current_energy

        temperature = cool(temperature, alpha)
        fitness_over_iterations.append(best_energy)

    return best_tour, best_energy, fitness_over_iterations

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

def main():
    """ testing """
    import os
    script_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_directory, 'TSP-Configurations/eil51.tsp.txt')

    cities = load_graph(filepath)
    print(cities)
    print(calculate_distances(cities))



if __name__ == '__main__':
    main()