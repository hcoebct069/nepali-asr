import soundfile as sf
import numpy
import matplotlib.pyplot as plt
#from itertools import zip_longest
from pprint import pprint
from statistics import *


# def grouper(iterable, n, fillvalue=None):
# 	args = [iter(iterable)] * n
# 	return zip_longest(*args, fillvalue=fillvalue)

stereo_audio_data, sample_rate = sf.read('../recordings/all.ogg')
mono_audio_data = stereo_audio_data[:,1] #uses 2nd channel :) 

plt.plot(mono_audio_data)
plt.savefig('mono_audio_data_all.png')
plt.clf()

list_data = list(mono_audio_data)
#list_data = (list(filter(lambda x: x > 0.1 or x < -0.1, mono_audio_data)))
data = numpy.asfarray(list_data)

#plt.plot(data)
#plt.savefig('processed_audio_data_all.png')
#plt.clf()

#Frame the signal
#sound signal = data

#hardcoded for now, 
#Actual calculation: 8KHz * 10ms <== Frame Step
#					 8KHz * 25ms <== Frame size
frame_step = 80
frame_size = 200

#padding 0 to the end if required
#total_sample_length = len(data)
#padding = (total_sample_length) % 80
#padder = [0] * padding

# Pad after framing

#print(len(list_data))

#list_data.append(padder)

#print(len(list_data))

#init frame list
#frames = grouper(list_data, frame_size)
#NOT USING grouper for now 
#

frames = []

for i in range(0, len(list_data), frame_step):
	frames.append(list_data[i:i+frame_size])

lastFrame = len(frames) -1
padder = len(frames[lastFrame]) % 80
frames[lastFrame].extend([0] * padder)

#pprint(frames)

#Use STE and ZCR on each frame. 

#For ZCR 
#Function: sgn 

def sgn(n):
	if n>=0:
		return 1
	return -1

zs1 = []
ps1 = []

for frame_index, frame in enumerate(frames):
	zs1.append(0)
	ps1.append(0)
	for index, sample in enumerate(frame):
		ps1[frame_index] = ps1[frame_index] + (sample)**2 
		try:
			zs1[frame_index] = zs1[frame_index] + abs(sgn(sample) - sgn(frame[index-1]))/2 
		except IndexError:
			zs1[frame_index] = zs1[frame_index] + sgn(sample)
	ps1[frame_index] = ps1[frame_index]/(index + 1)
	zs1[frame_index] = zs1[frame_index]/(index +1)

scaling_factor = 1  #educated Guess
ws1 = [(x * (1 - y))*scaling_factor for x, y in zip(ps1, zs1)]

assert(len(frames) == len(ws1)) #this is where things can go haywire

miu_w = mean(ws1)
delta_w = variance(ws1)

pprint(miu_w)
pprint(delta_w)
alpha = 0.95 #educated Guess

trigger_threshold = miu_w + alpha * delta_w
pprint(trigger_threshold);
#pprint(ws1);

#Write list comprehension or generator instead of this loop

vad_multiplier = [];
for each_ws1 in ws1:
	if(each_ws1 >= trigger_threshold):
		vad_multiplier.append(1)
	else:
		vad_multiplier.append(0)




assert(len(vad_multiplier) == len(frames))
pprint(len(frames))
#Multiply the frame with vad_multiplier to get points 
#
#Wanna see the plots? :D
#
#need to calculate (recreate signal plot from frame size and overlap size calculation)
#this will/should give indexes .... ask anup pokhrel

#plot frame

pprint(vad_multiplier)

multiplied = [x*y for x, y in zip(frames, vad_multiplier)]

pprint(multiplied)

#plt.plot(multiplied) # <= convert this to frame
plt.plot(vad_multiplier)
plt.savefig('vad_multiplier_all.png')
plt.clf()


# plt.plot(data)
# plt.savefig('processed_audio_data_zero.png')
# plt.clf()



#
#
#
# G - (down, down , up)*2 , C, G - (down, down:up, down:up), C, G - (down, down:up, down:up), F, D(down, down:up down:up)
# ^^ * 2
# 
# 
# Dark Side: 
# 
# 
# Solo:
# 
# Open G * 3 ; E-flat, B-flat, Open G
# E-flat : DString, (1) at First Fret 
# B-flat : BString, (1) at Third Fret
# 