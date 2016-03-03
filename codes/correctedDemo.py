import soundfile as sf
import numpy
import matplotlib.pyplot as plt
#from itertools import zip_longest
from pprint import pprint


# def grouper(iterable, n, fillvalue=None):
# 	args = [iter(iterable)] * n
# 	return zip_longest(*args, fillvalue=fillvalue)

stereo_audio_data, sample_rate = sf.read('../recordings/0.ogg')
mono_audio_data = stereo_audio_data[:,1] #uses 2nd channel :) 

plt.plot(mono_audio_data)
plt.savefig('mono_audio_data_zero.png')
plt.clf()


list_data = (list(filter(lambda x: x > 0.1 or x < -0.1, mono_audio_data)))
data = numpy.asfarray(list_data)

plt.plot(data)
plt.savefig('processed_audio_data_zero.png')
plt.clf()

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

