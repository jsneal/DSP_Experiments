## This file contains experiments that were run when writing my paper, "Why Would You Ever Set L = 1? The Advantage of High Temporal Bin Resolution," for EC 716 at BU.

## The sentence read in the original speech file "OneMustHaveAMindOfWinter.wav" is the opening line of a Wallace Stevens poem "The Snow Man." I used this speech file to compute the TDFT where M (frequency sampling factor) = window length, and L (temporal decimation factor) is set to various values.

## The files titled "output___.wav" are the results of either performing filter bank summation using FBS.py, or generalized filter bank summation using GFBS.py. I wrote the scripts myself. The comments in those scripts are not for the purpose of others understanding how they work. This will be provided in the future.

## FBS was used in the cases were L = 1, L = 2, and L = 4, while GFBS was used on the critical sampling case where L = M, and L = M/2.

## What is key to note is that the output files that use FBS, but where L =/= 1 still synthesize the original signal. I sugges that this is probably because in the grand scheme of the entire signal the amount of information lost is approximately nothing, though one will notice it sounds as if the signal is being lowpass filtered in the case where L = 4.

## For the GFBS cases, one notices a "buzzing" in the background.