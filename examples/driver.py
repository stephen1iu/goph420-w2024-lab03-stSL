import numpy as np
import matplotlib.pyplot as plt
from lab_03.regression import multi_regress

def main():
    data = np.loadtxt("../data/M_data.txt")
    mag = data[:,-1]
    time = data[:,0]
    plt.figure()
    plt.plot(time, mag)
    plt.savefig("../figures/raw_data.png")

    day_cuts = [3928, 5684, 6974, 9536]

    mag1 = mag[:day_cuts[0]]
    mag2 = mag[day_cuts[0]:day_cuts[1]]
    mag3 = mag[day_cuts[1]:day_cuts[2]]
    mag4 = mag[day_cuts[2]:day_cuts[3]]
    mag5 = mag[day_cuts[3]:]

    magnitudes = [mag1, mag2, mag3, mag4, mag5]

    plt.figure()
    fig, ax = plt.subplots(5, figsize = (20, 20))
    ax[0].plot(time[:day_cuts[0]], mag1, label = "Day 1")
    ax[1].plot(time[day_cuts[0]:day_cuts[1]], mag2, label = "Day 2")
    ax[2].plot(time[day_cuts[1]:day_cuts[2]], mag3, label = "Day 3")
    ax[3].plot(time[day_cuts[2]:day_cuts[3]], mag4, label = "Day 4")
    ax[4].plot(time[day_cuts[3]:], mag5, label = "Day 5")    
    plt.savefig("../figures/daily_data.png")

    cutoffs = [34, 46, 72, 96]

    plt.figure(figsize=(20,20))
    plt.plot(time, mag, 'or')
    plt.vlines(cutoffs, -2, 2, colors="k")
    plt.savefig("../figures/vline_data.png")

    i = 0
    index = np.zeros(0, dtype = int)
    for cut in cutoffs:
        while cut > time[i]:
            i += 1
        index = np.append(index, i)

    M = np.linspace(-0.25 , 1, 25)

    y0 = mag[:index[0]]
    y1 = mag[index[0]: index[1]]
    y2 = mag[index[1]: index[2]]
    y3 = mag[index[2]: index[3]]
    y4 = mag[index[3]:]

    N0 = [sum(1 for j in y0 if j > M[k]) for k in range(len(M))]
    N1 = [sum(1 for j in y1 if j > M[k]) for k in range(len(M))]
    N2 = [sum(1 for j in y2 if j > M[k]) for k in range(len(M))]
    N3 = [sum(1 for j in y3 if j > M[k]) for k in range(len(M))]
    N4 = [sum(1 for j in y4 if j > M[k]) for k in range(len(M))]
    N_arr = [N0, N1, N2, N3, N4]

    row1 = np.ones_like(N0)
    Z = np.ones_like(N0)*-M
    Z = np.column_stack((row1, Z))
    residuals = [[], [], [], [], []]

    for i, N in enumerate(N_arr):
        plt.figure()
        a, e, rsq = multi_regress(np.log10(N), Z)
        rgr = 10**np.dot(Z, a)
        residuals[i].append(e)
        plt.semilogy(M, N, label = "Number of events")
        eq = f"N = {a[0]:.2f} - {a[1]:.2f}M \n $R^2$ = {rsq:.3}"
        plt.plot(M, rgr, label=eq)
        plt.xlabel("Magnitudes")
        plt.ylabel("Number of Events")
        plt.legend()
        plt.savefig(f"../figures/Period {i+1}.png")

    plt.figure()
    for j, res in enumerate(residuals):
        plt.plot(M, res[0], label=f"Period {j}")
        plt.xlabel("Magnitudes")
        plt.ylabel("Number of Events")
        plt.legend()
        plt.savefig("../figures/rsq.png")

if __name__ == "__main__":
    main()