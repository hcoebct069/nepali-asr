import soundfile as sf
import numpy
import matplotlib.pyplot as plt

from pprint import pprint
from statistics import *

from math import *

#For MFCC
from scipy.fftpack import dct

from python_speech_features import mfcc

from yahmm import *



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

#Using MFCC library, we can eradicate the following code:
#

mfcc_feat = mfcc(numpy.asarray(multiplied), sample_rate);
pprint(mfcc_feat);
pprint(len(mfcc_feat[0]))

'''



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

'''

'''
Create HMM model using yahmm

Each number's model will have 13 state with Normal Distribution
Randomly assign all transition values and Normal Distribution Parameter

Use MFCCs as training data and train the model to improve performance

BONUS points: Store the trainned HMMs (in file?)

To test, calculate the log probability for each word and choose maximum. 

Project Completed! (Give me one night of pure coding!)

Ok, I got it wrong, no big deal! 

I got it right again,

Don't create 13 states, create 3 or 4, but assign each state a GMM  (Multivariate) of dimension 13
and then train matching the numbers. 

[ [[13],[13],[13]],[[13],[13],[13]],[[13],[13],[13]],[[13],[13],[13]] ]



'''

# mfcc_feat contains the 13-dimensional array
# 

# In future replace the following code to search a directory for predefined library and do stuffs
# 

model = [];
model.append(Model(name = "zero")) #0
model.append(Model(name = "one")) #1
model.append(Model(name = "two")) #2
model.append(Model(name = "three")) #3
model.append(Model(name = "four")) #4
model.append(Model(name = "five")) #5
model.append(Model(name = "six")) #6
model.append(Model(name = "seven")) #7
model.append(Model(name = "eight")) #8
model.append(Model(name = "nine")) #9

# The word length of every nepali digits can be accumulated on 3 words, so selecting 3 states
# Assigning Flat initial probabilities (transition, and pdfs)
# Assuming Gaussian Mixture Model (Multi[13]-Dimensional Gaussian (Normal) Distribution <-- because the number of cepstral coefficent is 13)
# 

for each_model in model:
	s1 = State(MultivariateDistribution([NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2)]))
	s2 = State(MultivariateDistribution([NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2)]))
	s3 = State(MultivariateDistribution([NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2),NormalDistribution(.3,.2)]))
	each_model.add_states([s1, s2, s3])
	each_model.add_transition(each_model.start, s1, 1)
	each_model.add_transition(s1, s1, 0.3)
	each_model.add_transition(s1,s2, 0.7)
	each_model.add_transition(s2,s2,0.4)
	each_model.add_transition(s2,s3,0.6)
	each_model.add_transition(s3,s3,0.45)
	each_model.add_transition(s3, each_model.end, 0.55)
	each_model.bake()



#Start Comment Block
'''
# For Debugging Purposes
# Wow, So this means that we can train with the given flat MFCC data 
# and test with that too.1

check_list = [1,2,3,4,5,6,7,8,9,10,11,12,13];
for each_model in model:
	each_model.train([[check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list]])
	a = each_model.log_probability([check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list,check_list])
	pprint("Model Log Probability for given Sequence")
	pprint("=================")
	pprint(a)



'''
#Stop Comment Block

#Start Comment Block
'''
# For Debugging Purposes
# Since this is an infinte generator, the sample size is random. 
# This generally should mean yahmm supports multi-dimensional stuff
# And based on previous debugging test, it is now known that we can test and train on the given sequence of MFCC directly
# :)

for each_model in model:
	pprint("Printing Model now ")
	pprint("=============")
	pprint(len(each_model.sample()))

'''
#Creation of HMMs complete, now train it
#
#Check for folder, data > training > (zero, ... nine)
#Use loop to train
#
#
#For testing
#
#Check for folder, data > testing > (zero, ... nine)
#
#Test how many files in zero is correctly predicted as zero, and so on.. 

##After training, look in yahmm docs and find a way to store those in a file
#Write a seperate module in GUI to load those HMM files and use that in testing data 




