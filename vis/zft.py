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

def plot_zft(name, m):
    data = []
    x = 0.152
    map_name = name
    for iter in range(0, 10001):
        data.append(x)
        x = maps(map_name , m , x)
    return data

# plot_zft('logistic', 4.0)
# plot_zft('tent', 1.89)
# plot_zft('gaussian',1.0)
# plot_zft('tent_v2', 0.499)

logistc_data = plot_zft('logistic', 4.0)
gauss_data = plot_zft('gaussian',-0.5)
tent_data = plot_zft('tent', 1.999)
fig, axes = plt.subplots(1,3)
axes[0].hist(logistc_data, bins=10,color='black')
axes[0].set_title('logistic')
axes[1].hist(gauss_data, bins=10,color='black')
axes[1].set_title('gauss')
axes[2].hist(tent_data, bins=10,color='black')
axes[2].set_title('tent')
plt.show()

