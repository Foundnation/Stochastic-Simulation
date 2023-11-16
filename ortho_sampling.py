import numpy as np

def init_genrand(seed):
    np.random.seed(seed)

def permute(arr):
    np.random.shuffle(arr)

def random_orthogonal_sampling(MAJOR, RUNS, x_range=(-2, 1), y_range=(-1, 1)):
    SAMPLES = MAJOR * MAJOR

    x_min, x_max = x_range
    y_min, y_max = y_range
    x_scale = (x_max - x_min) / SAMPLES
    y_scale = (y_max - y_min) / SAMPLES

    scale = 4.0 / SAMPLES
    xlist = np.zeros((MAJOR, MAJOR), dtype=int)
    ylist = np.zeros((MAJOR, MAJOR), dtype=int)

    output = []

    m = 0
    for i in range(MAJOR):
        for j in range(MAJOR):
            xlist[i][j] = ylist[i][j] = m
            m += 1
    
    for k in range(RUNS):
        for i in range(MAJOR):
            permute(xlist[i])
            permute(ylist[i])

        for i in range(MAJOR):
            for j in range(MAJOR):
                x = x_min + x_scale * (xlist[i][j] + np.random.random())
                y = y_min + y_scale * (ylist[j][i] + np.random.random())


                output.append(complex(x, y))
    
    return output

# Initializing random number generator
#init_genrand(3737)

# Perform random orthogonal sampling
#print(random_orthogonal_sampling())
