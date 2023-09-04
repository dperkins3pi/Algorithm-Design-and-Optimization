# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Daniel Perkins
MATH 321
09/04/23
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """ Create an (n x n) array of values randomly sampled from the standard
    normal distribution. Compute the mean of each row of the array. Return the
    variance of these means.

    Parameters:
        n (int): The number of rows and columns in the matrix.

    Returns:
        (float) The variance of the means of each row.
    """
    sample = np.random.normal(size=(n,n))  #creates an nxn normal sample
    mean = np.mean(sample, axis=1) #computs the mean of each row
    variance = np.var(mean)
    return variance

def prob1():
    """ Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    n = np.arange(100, 1100, 100)  #n values for impuy
    variance = lambda x: var_of_means(x)  #lambda function to apply var_of_means() to each value
    array_of_variance = np.array([variance(ni) for ni in n]) #array of results
    plt.plot(n, array_of_variance) #plot it
    plt.show()


# Problem 2
def prob2():
    """ Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100) #define the domain
    plt.plot(x, np.sin(x)) #plot the functions
    plt.plot(x, np.cos(x))
    plt.plot(x, np.arctan(x))
    plt.show() #show the plot


# Problem 3
def prob3():
    """ Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x1 = np.linspace(-2, 1, 100, endpoint=False) #[-2,1)
    x2 = np.linspace(1, 6, 100)[1:-1] #(1,6], excluding first point
    y1 = 1 / (x1 - 1)
    y2 = 1 / (x2 - 1)
    plt.plot(x1, y1, "m--", lw=4)
    plt.plot(x2, y2, "m--", lw=4)
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    plt.show()


# Problem 4
def prob4():
    """ Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi], each in a separate subplot of a single figure.
        1. Arrange the plots in a 2 x 2 grid of subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    #overall setup
    x = np.linspace(0, 2*np.pi, 100) #Domain
    fig, axes = plt.subplots(2, 2) #arrange plots into 2x2 grid
    fig.tight_layout(pad = 2) #space out the axes
    plt.setp(axes, xlim=(0, 2*np.pi), ylim=(-2, 2)) #set limits of each subplot to [0,2pi]x[-2,2]
    plt.suptitle("Graphs of asin(kx)")
    
    #sin(x)
    axes[0,0].set_title("sin(x)")
    axes[0,0].plot(x, np.sin(x), "g-")

    #sin(2x)
    axes[0,1].set_title("sin(2x)")
    axes[0,1].plot(x, np.sin(2*x), "r--")

    #2sin(x)
    axes[1,0].set_title("2sin(x)")
    axes[1,0].plot(x, 2*np.sin(x), "b--")

    #2sin(2x)
    axes[1,1].set_title("2sin(2x)")
    axes[1,1].plot(x, 2*np.sin(2*x), "m:")

    plt.show()


# Problem 5
def prob5():
    """ Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    #upload data into an array
    data = np.load("FARS.npy") 
    times = data[:,0]
    longitudes = data[:, 1]
    latitudes = data[:, 2]

    #set up axes
    fig, axes = plt.subplots(2, 1) #arrange plots into 1x2 grid
    fig.tight_layout(pad = 4)

    #scatter plot
    axes[0].set_title("Location of motor vehicle traffic crashes from 2010-2014 (FARS)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_aspect("equal") # x and y scaled the same way
    axes[0].plot(longitudes, latitudes, "ko", ms=0.01) #small marker size to show difference between east and west
    
    #histogram
    plt.setp(axes[1], xlim=(0, 23), ylim=(0, 15000)) #set x and y limits
    axes[1].set_title("Time when the traffic crashes occured 2010-2014 (FARS)")
    axes[1].set_xlabel("Time of day (military time)")
    axes[1].set_ylabel("Frequency of accidents")
    axes[1].hist(times, bins=np.arange(0, 24))

    #show the plot
    plt.show()


# Problem 6
def prob6():
    """ Plot the function g(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of g, and one with a contour
            map of g. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Include a color scale bar for each subplot.
    """
    #Set up domain and function
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = x.copy()
    X, Y = np.meshgrid(x,y)
    Z = (np.sin(X) * np.sin(Y)) / (X * Y)

    #First subplot
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap="magma")
    plt.colorbar()
    plt.xlim(-2*np.pi, 2*np.pi, 100)
    plt.ylim(-2*np.pi, 2*np.pi, 100)

    #Second subplot
    plt.subplot(122)
    plt.contour(X, Y, Z, 50, cmap="coolwarm")
    plt.colorbar()
    
    #show the plots
    plt.show()


#if __name__=="__main__":
    #prob1()
    #prob2()
    #prob3()
    #prob4()
    #prob5()
    #prob6()