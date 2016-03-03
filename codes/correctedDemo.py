import soundfile as sf
import numpy
import matplotlib.pyplot as plt

stereo_audio_data, sample_rate = sf.read('../recordings/0.ogg')
mono_audio_data = stereo_audio_data[:,1] #uses 2nd channel :) 

plt.plot(mono_audio_data)
plt.savefig('mono_audio_data_zero.png')
plt.clf()

data = numpy.asfarray(list(filter(lambda x: x > 0.1 or x < -0.1, mono_audio_data)))

plt.plot(data)
plt.savefig('processed_audio_data_zero.png')
plt.clf()

#write code to calculate envelope
#and then achieve a threshold
#create blocks of signals where value is greater than threshold

