import numpy as np
import matplotlib.pyplot as plt
from lab_03.regression import multi_regress

def main():
    
    data = np.loadtxt("../data/M_data.txt")
    mag = data[:,-1]
    time = data[:,0]
    plt.plot(time, mag)
    plt.savefig("../figures/data.png")

if __name__ == "__main__":
    main()