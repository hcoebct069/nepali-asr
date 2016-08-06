import soundfile as sf
import numpy
import matplotlib.pyplot as plt

from pprint import pprint
from statistics import *

from math import *

#For MFCC
from scipy.fftpack import dct



stereo_audio_data, sample_rate = sf.read('../recordings/all.ogg')
mono_audio_data = stereo_audio_data[:,1] #uses 2nd channel :) 
data = list(mono_audio_data)

#Convert data into matrix of frames
#

print(len(data))

frames = []

for i in range(0, len(data), 200):
	frames.append(data[i:i+200])


lastFrame = len(frames) -1
padder = 200 - len(frames[lastFrame]) % 200
pprint(padder)
frames[lastFrame].extend([0] * padder)
print(len(frames))

window = 1/200 #frame size  ==> w(n) = 1/2N

def sgn(val):
	if(val >= 0):
		return 1;
	return -1;

zcr = []

for each_row in frames:
	zcr_inner = 0;
	for index, each_record in enumerate(each_row):
		if (index == 0):
			continue
		zcr_inner = zcr_inner + abs(sgn(each_record) - sgn(each_row[index-1]))/400
	zcr.append(zcr_inner);	

ste = []

for each_row in frames:
	ste_inner = 0;
	for index, each_record in enumerate(each_row):
		ste_inner = ste_inner + ((each_record*(0.54 - 0.46 * cos(2*pi*(index+1)/199)))**2)
	ste.append(ste_inner)


assert(len(ste) == len(zcr))


#calculate multiplier
#

multiplier = [];

for index, each_record in enumerate(zcr):
	if(each_record <= ste[index]):
		multiplier.append(1)
	else:
		multiplier.append(0)

print(multiplier);

total = []

for each_multiplier in multiplier:
	total.extend([each_multiplier] * 200)

padded_data = data;
padded_data.extend([0]*padder)
assert(len(padded_data) == len(total))

multiplied = [x * y for x, y in zip(padded_data, total)]


#First plot the multiplier
#

plt.plot(numpy.asfarray(padded_data))
plt.plot(numpy.asfarray(total))
plt.savefig('amono_audio_data_all_processed_superimposed.png')
plt.clf()

plt.plot(numpy.asfarray(multiplied))
plt.savefig('amono_audio_data_all_processed.png')
plt.clf()

sf.write('final.wav', multiplied, sample_rate)

# Frame "multiplied" (which is already padded) with some overlap and calculate MFCC for each frame.
# You get an vectors of MFCC coefficients
# Store that in some db, look for how to build HMM based clasifier based on those MFCCs
# Tell others to find how to use the classifer to get MFCC as input and get output as sequence of words
# or phonemes
# Build a phonetic dictionary
# See YAHMM

indexes = [ i for i, (x, y) in enumerate(zip(multiplier[:-1],multiplier[1:])) if x!=y]

pprint(indexes)

##Framing routine
#
#Do not neeed indexes, just do the thing in multiplied. 
#
#Use indexes for comparision of accuracy in Total number of words recognized Vs Actual number of words
#
#

#Framing, each frame starts from 80th sample, with size 200

overlap_frames = []
for i in range(0, len(multiplied), 80):
	overlap_frames.append(multiplied[i:i+200])

last_overlap_frame = len(overlap_frames) - 1
if(len(overlap_frames[last_overlap_frame]) < 200):
	overlap_frames[last_overlap_frame].extend([0] * (200 - len(overlap_frames[last_overlap_frame])))


pprint(overlap_frames)
#pprint(len(overlap_frames[last_overlap_frame]));

assert(len(overlap_frames[last_overlap_frame]) == 200)



#Each array in a 2D overlap_frames array has the signal, Use that to compute MFCC 
#Store in an vector

def melFilterBank(blockSize):
	numBands = int(numCoefficients)
	maxMel = int(freqToMel(maxHz))
	minMel = int(freqToMel(minHz))


	#Create a matrix for triangular filter, one row per filter
	#
	
	filterMatrix = numpy.zeros((numBands, blockSize))	
	melRange = numpy.array(range(numBands + 2))

	melCenterFilters = melRange * (maxMel - minMel) / (numBands + 1 ) + minMel

	#each array index represent the center of each triangular filter
	aux = numpy.log(1 + 1000.0 / 700.0) / 1000.0
	aux = (numpy.exp(melCenterFilters * aux) - 1) / 22050 
	aux = 0.5 + 700 * blockSize * aux
	aux = numpy.floor(aux)
	centerIndex = numpy.array(aux, int) #Get int values

	for i in range(numBands):
		start, center, end = centerIndex[i:i+3]
		k1 = numpy.float32(center - start)
		k2 = numpy.float32(end - center)
		up = (numpy.array(range(start, center)) - start) / k1 
		down = (end - numpy.array(range(center, end))) / k2 
		filterMatrix[i][start:center] = up 
		filterMatrix[i][center:end] = down 

	return filterMatrix.transpose()

def freqToMel(freq):
	return 1127.01048 * log(1 + freq /  700.0)

def melToFreq(mel):
	return 700 * (exp(mel / 1127.01048 )-1)

print("============================")

numCoefficients = 13
minHz = 0
maxHz = 22.000

for each_frame in overlap_frames:
	complexSpectrum = numpy.fft.fft(each_frame)
	powerSpectrum = abs(complexSpectrum) ** 2
	filteredSpectrum = numpy.dot(powerSpectrum, melFilterBank(200)) # <- melFilterBank is where the problem is.
	logSpectrum = numpy.log(filteredSpectrum)
	dctSpectrum = dct(logSpectrum, type = 2)

	pprint(dctSpectrum) #Hopefully, this contains 13 MFCCs


#need motivation, do not have any :/
#
#
#Problem is in filterbank creation
#https://github.com/jameslyons/python_speech_features <-- Use this 