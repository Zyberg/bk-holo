import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.optimize import leastsq


def get_tenengrad(directory_path):
    distances = []
    tenengrads = []
    
    # Traverse the directory and find JSON files
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                distances.append(data.get("distance", float('-inf')))
                tenengrads.append(data.get("tenengrad", float('-inf')))
                
    return distances, tenengrads

def sine_func(x, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase)

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

if __name__ == "__main__":
    ## ------------------------------
    # Read input 
    ## ------------------------------
    parser = argparse.ArgumentParser(description="Find JSON file with highest 'distance' value")
    parser.add_argument("output_path", type=str, help="Output JSON file path")
    parser.add_argument("directory_path", type=str, help="Directory path containing JSON files")
    args = parser.parse_args()

    ## ------------------------------
    # Process 
    ## ------------------------------
    

    t, data = get_tenengrad(args.directory_path)

    t = np.array(t)
    data = np.array(data)

    # res = fit_sin(distances, tenengrads)

    # popt, pcov = curve_fit(sine_func, distances, tenengrads)

    # amplitude, frequency, phase = popt

    # period = 1 / frequency

    guess_mean = np.mean(data)
    guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
    guess_phase = 0
    guess_freq = 1
    guess_amp = 1

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - data
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean

    # recreate the fitted curve using the optimized parameters

    fine_t = np.arange(0,max(t),0.1)
    data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean

   
    ## ------------------------------
    # Persist results
    ## ------------------------------
    fig, ax = plt.subplots()

    # ax.scatter(distances, tenengrads, color='red')

    ax.plot(t, data, '.')
    ax.plot(t, data_first_guess, label='first guess')
    ax.plot(fine_t, data_fit, label='after fitting')
    ax.legend()

    # plt.plot(tt, yynoise, "ok", label="y with noise")
    # plt.plot(distances, res["fitfunc"](distances), "r-", label="y fit curve", linewidth=2)
    # plt.legend(loc="best")

    # ax.plot(distances, sine_func(distances, *popt), 'r-', label='Fitted function')

    # ax.title('Fitted Sine Function')

    # ax.xlabel('Propagation distance (m)')
    # ax.ylabel('Tenegrad coefficient')

    fig.savefig(args.output_path, dpi=300)
    
    plt.close(fig)

    # print("Amplitude:", amplitude)
    # print("Frequency:", frequency)
    # print("Phase:", phase)
    # print("Period:", period)


    print("Tenengrad plot saved successfully to:", args.output_path)

