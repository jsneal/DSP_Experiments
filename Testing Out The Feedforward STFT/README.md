# This folder contains three files related to experimenting with the Feedforward STFT computation method as shown in
M. Garrido, "The Feedforward Short-Time Fourier Transform," in IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 63, no. 9, pp. 868-872, Sept. 2016.
    doi: 10.1109/TCSII.2016.2534838

OneMustHaveAMindOfWinter.wav -> my standard audio file for testing. I read the opening line of Wallace Steven's "The Snow Man."

FeedforwardSTFTtest.py -> Python script that runs the test. Make sure you have the imported dependencies and either my test audio file or your own. If you use your own, be sure to change the code.

FeedforwardPythonImplementation.png -> A plot of the results of running the test. "Iterations" corresponds to which column of the STFT is being computed. "Time" corresponds to the time it took to compute a particular column. The FFT-based method succeeds over the Feedforward method, which is contrary to the results presented in the paper. My suspicion is that they did not use a built in fft() function, but simply wrote the Cooley-Turkey algorithm for a radix-2 8 point FFT, and that is why the Feedforward operates faster. See FeedforwardSTFTtest.py for more details about the test I ran.