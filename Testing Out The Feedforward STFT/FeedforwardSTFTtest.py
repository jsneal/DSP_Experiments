"""
    This module is for testing out the Feedforward STFT refererenced in 
    M. Garrido, "The Feedforward Short-Time Fourier Transform," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 63, no. 9, pp. 868-872, Sept. 2016.
    doi: 10.1109/TCSII.2016.2534838

    I am trying to implement the algorithm to check Figure 5 in "The Feedforward Short-Time Fourier Transform" which compares the method of computing the STFT
    column by column with separate STFTs in MATLAB with using the proposed Feedfoward STFT algorithm. I implemented both in MATLAB and actually got results contrary
    to what the paper describes, with the columnwise technique being faster than the Feedforward technique. I have a suspicion that this is related to MATLAB's use of FFTW.

    I want to test out whether this suspicion is true by comparing this Python implementation with my MATLAB implementation. Here I will use scipy.fftpack(), which does not use
    an "adaptive FFT program" like FFTW.

    I'm using an Intel Core i7 8th Gen processor. The results in the paper were achieved on an Intel Core i3.

    The Feedforward algorithm used in this file is pretty much straight out of the paper. I'm not creatively responsibile for it in any way.

    As usual, I will use my speech file "OneMustHaveAMindOfWinter.wav" in which I read the opening line of Wallace Steven's "The Snow Man."

    Update:

    After running this experiment, I recieved roughly the same results that I recieved when performing this experiment in MATLAB. The fft-based method 
    (columnwise computation of FFT's from scratch) was significantly faster than the Feedforward Technique. I am stumpted as to why, but I think it has to do with fetching from memory
    that is performed when using the Feedforward technique as opposed to using scipy.fftpack(). What confuses me is that I thought FFT-based method was faster because it used FFTW which adapts
    to the architecture being used. Perhaps fftpack considers the hardware-aspect of the problem of FFT computation is well, though without the adaptive programming.

    All in all, I am unsure as to how Figure 5 was produced in this paper ("The Feedforward Short-Time Fourier Transform"). Perhaps it also has something to do with the architecture involved.
    My next step is to investigate the Intel i3 further and compare it with the i7.

    This python script should work out-of-the-box if you have all of the dependencies and my audio file (OneMustHaveAMindOfWinter.py). 
    If something is wrong, submit an issue and I will try to address it when I can.

"""
import time
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib
import math

[Fs, speechSample] = wavfile.read('OneMustHaveAMindOfWinter.wav', mmap=False) 

fftBasedTimeIncrements = np.zeros(100, dtype='float')
FeedForwardTimeIncrements = np.zeros(100, dtype='float')



FFTbasedSTFT = np.zeros(shape=(100,8), dtype='complex')

for i in range(0, 100):
    t = time.time()
    FFTbasedSTFT[i, :] = fftpack.fft(speechSample[i:i+8])
    duration = time.time() - t
    fftBasedTimeIncrements[i] = duration

N = 8
inputSignal = speechSample[0:100]
FFSTFT = np.zeros(shape=(100,8), dtype='complex')

# FeedForward Technique
b10 = np.zeros(shape=(N/2, 1), dtype='complex')
b20 = np.zeros(shape=(N/2, 1), dtype='complex')
b21 = np.zeros(shape=(N/2, 1), dtype='complex')
b30 = np.zeros(shape=(N/2, 1), dtype='complex')
b31 = np.zeros(shape=(N/2, 1), dtype='complex')
b32 = np.zeros(shape=(N/2, 1), dtype='complex')
b33 = np.zeros(shape=(N/2, 1), dtype='complex')
signalLength = len(inputSignal)
for n in range(0, signalLength):
    t = time.time()
    Input = inputSignal[n]
    x00 = Input             # Input sample
    m1 = n % (N/2)    # Stage 1
    x10 = b10[m1] + x00
    x11 = b10[m1] - x00
    b10[m1] = x00
    m2 = n % (N/4)    # Stage 2
    xr11 = (-1j)*x11
    x20 = b20[m2] + x10
    x21 = b20[m2]- x10
    x22 = b21[m2] + xr11
    x23 = b21[m2] - xr11
    b20[m2] = x10
    b21[m2] = x11
        
    m3 = n % (N/8)       # Stage 3
    xr21 = (-1j)*x21
    xr22 = (.7071 - .7071j)*x22
    xr23 = (-.7071 - .7071j)*x23
    x30 = b30[m3] + x20
    x31 = b30[m3] - x20
    x32 = b31[m3] + xr21
    x33 = b31[m3] - xr21
    x34 = b32[m3] + xr22
    x35 = b32[m3] - xr22
    x36 = b33[m3] + xr23
    x37 = b33[m3] - xr23
    b30[m3] = x20
    b31[m3] = x21
    b32[m3] = x22
    b33[m3] = x23
      
    FFSTFT[n, 0] = x30
    FFSTFT[n, 1] = x34
    FFSTFT[n, 2] = x32
    FFSTFT[n, 3] = x36
    FFSTFT[n, 4] = x31
    FFSTFT[n, 5] = x35
    FFSTFT[n, 6] = x33
    FFSTFT[n, 7] = x37
    duration = time.time() - t
    FeedForwardTimeIncrements[n] = duration

"""
------------
Plots
------------
"""

plt.figure(1)
plt.plot(fftBasedTimeIncrements)
plt.plot(FeedForwardTimeIncrements)
plt.title('Speed Over 100 iterations')
plt.axis([0, 120, 0, .00012])
plt.ylabel('Time (s)')
# Each iteration corresponding to the computation of 1 column of the STFT
plt.xlabel('Iterations')
plt.legend(('FFT-based', 'Feedforward Technique'))
plt.show()

"""
fftBasedMean = np.mean(fftBasedTimeIncrements);
print('The Average Time to Compute an STFT column for FFT-based Technique is ' + str(fftBasedMean))
feedforwardMean = np.mean(FeedForwardTimeIncrements);
print('The Average Time to Compute an STFT column for Feedforward Technique is ' + str(feedforwardMean))
"""
