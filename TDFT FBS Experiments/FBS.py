"""
I used Audacity to read record the audio at 44.1 kHz (recommended by a google search for audio processing):

Source: http://www.lavryengineering.com/pdfs/lavry-white-paper-the_optimal_sample_rate_for_quality_audio.pdf

Source: https://en.wikipedia.org/wiki/44,100_Hz

I purposefully exported this as a wavfile so that I could read it into a Numpy array and use scipy.signal and scipy.fftpack for manipulations:

Source: https://docs.scipy.org/doc/scipy-0.14.0/reference/index.html

"""

import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib
import math

# 	Reads the recorded audio (.wav format) of the first line of Wallace Steven's "The Snow Man" and
# 	returns a Numpy array representation and the sample rate of the audio.

"""
def stereoToMono(audiodata): # https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
    newaudiodata = []
    audiodata = audiodata.astype(float)
    for i in range(len(audiodata)):
        d = (audiodata[i][0]/2 + audiodata[i][1]/2)
        newaudiodata.append(d)
s
    return np.array(newaudiodata, dtype='int16')
"""

[sampleRate, data] = wavfile.read('OneMustHaveAMindOfWinterElevenHertz.wav', mmap=False) # data is 1 x 88576 numpy array (int16) (Not anymore)
data = np.append(data, np.zeros(220, dtype='float'))
print(len(data))
"""
Why does this make sense that the numpy array is 2 X 1774107? (old)

-- Well we sampled at 44.1 kHz and the audio length is roughly 40.23 sseconds. 44100*40.23 = 1,774,143
-- So each 2 element array is equivalent to 1 sample

Why a two element array?
-- I recorded the audio in Stereo which has two inputs. I should have recorded in Mono: https://stackoverflow.com/questions/30401042/stereo-to-mono-wav-in-python
"""

"""
------------------------------ This area shows how the stereoToMono function does not produce the same Mono as Audacity.
print(str(data))

data = stereoToMono(data)

print(str(data))

[sampleRate, data] = wavfile.read('TheSnowManMono.wav', mmap=False) # data is a a 2 X 1,774,107 numpy array

print(str(data))
-----------------------------

"""

"""
Here I wondered how to interpret the Numpy array that came from the .wav file. Questions in my mind: 

What does the amplitude represent? Voltage

Source: https://au.mathworks.com/matlabcentral/answers/86258-what-is-the-amplitude-unit-when-we-plot-wav-files

How is the time measured?

-- Well since we recorded with a sample rate of 441 kHz, the time unit must be 1/441000 seconds or 1/441 ms.
"""

dftList = [] # initializing out list of DFT's for the time representation of the TDFT.
winLength = 220
M = 220
N = len(data) # length of our Nw point time varying signal
boxWindow = np.append(np.ones(winLength, dtype='float'), np.zeros(N-winLength, dtype='float')).transpose()

dftList = [] # initializing out list of DFT's for the time representation of the TDFT.
M = 220 # length of the window
N = len(data) # length of our Nw point time varying signal
L = 4 		# To define Temporal Decimation Factor so as to achieve critical sampling rate


for n in range(0, (N/L)-winLength):

	dftList.append(fftpack.fft(data[(n*L):(winLength+(n*L))]*boxWindow[0:winLength]))

filterBank = np.zeros((N/L), dtype='float')
filterBankList = []

for k in range(0, M):
	filterBank = np.zeros((N/L), dtype='float')
	for n in range(0, (N/L)-winLength):
		filterBank[n] = dftList[n][k]
	filterBankList.append(filterBank)

output = np.zeros(N/4, dtype='complex')


for k in range(0, M):
	output = filterBankList[k][0:N] + output

output = (M/2)*output.astype('int32')

wavfile.write('outputFBSBadL4.wav', sampleRate/4, output) # ValueError: Unsupported data type 'complex128'

