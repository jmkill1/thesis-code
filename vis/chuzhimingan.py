import numpy as np
from matplotlib import pyplot as plt

# this simple code produces bifurcation diagram for different nonliniear maps.

# range command just accepts integer numbers, this function defines a my_range command
#  that accepts float numbers
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

# this function defines different chaotic maps,
# name argument specifies the name of the map, here I have defined logistic, tent, ecology, and gaussian map
# m is a real number which is known as the control parameter of maps
# x is a real number which is the output of the map.
def maps(name,  m , x):
    if name == 'logistic':
        #in this definition m varies [0,4]
        return m * x * (1.0 - x)
    if name == 'tent':
        if x < 0.5:
            # in this definition m varies [1,2]
            return m* x
        if x > 0.5:
            return m * (1.0 - x)
    if name =='ecology':
        return x * np.exp( m * ( 1.0 - x ) )
    if name == 'gaussian':
        b = 5.0    # b is a free parameter in gaussian map that could be a real number
                   # b could be treated as an other bifurcation parameter of the map
        return np.exp(-b * x * x) + m
    if name == 'tent_v2':
        if x < m:
            return x / m
        else:
            return (1 - x) / (1 - m)

# this function plots the bifurcation diagram for a map
# name specifies the name of the map
# m_min and m_max  represent the m parameter's domain(which is the control parameter of the map)
# step is the step of m parameter



def plot_mg(name, x, m):
    data = []
    iters = []
    map_name = name
    for iter in range(0,24):
        x = maps(map_name, m, x)
        # if iter > 10000:
        #     iters.append(iter)
        #     data.append(x)
        iters.append(iter)
        data.append(x)
    return data, iters


data1, iters = plot_mg('tent', 0.1522, 1.999)
data2, iters = plot_mg('tent', 0.1523, 1.999)
data3, iters = plot_mg('tent', 0.1522, 1.998)
data4, iters = plot_mg('tent', 0.1523, 1.998)
fig, axes = plt.subplots(2,2)
axes[0][0].scatter(iters, data1, c = iters, cmap = "viridis")
axes[0][0].set_title('$x_0 = 0.1522 \ \ r = 1.999$')
axes[0][1].scatter(iters, data2, c = iters, cmap = "viridis")
axes[0][1].set_title('$x_0 = 0.1523 \ \ r = 1.999$')
axes[1][0].scatter(iters, data3, c = iters, cmap = "viridis")
axes[1][0].set_title('$x_0 = 0.1522 \ \ r = 1.998$')
axes[1][1].scatter(iters, data4, c = iters, cmap = "viridis")
axes[1][1].set_title('$x_0 = 0.1523 \ \ r = 1.998$')
plt.show()

