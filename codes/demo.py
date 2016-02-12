import numpy
import soundfile as sf
import matplotlib.pyplot as plt

data, samplerate = sf.read('../recordings/0.ogg')
a = numpy.ndarray(1)
a = data[0][0]
for i in range(1, len(data)):
	a = numpy.append(a, data[i][0])
#conversion from 2-D stero sound to mono sound complete
plt.plot(a)
plt.savefig('signal_zero.png')
b = numpy.ndarray(1)
b[0] = 0
for i in range(0, len(a)):
	if(a[i]>=0.1):
		b = numpy.append(b, a[i])

plt.clf()
plt.plot(b)
plt.savefig('signal_zero_cut.png')
#c = numpy.ndarray(1)
#c[0] = b[0]
#for i in range(1, len(b)):
#	c = numpy.append(c, a[i])
#sf.write('newfile.wav', c, samplerate)
newdata = data
for i in range(0, len(data)):
	if(data[i][0] <0.1):
		newdata[i] = [0, 0]

print(len(newdata))
print(samplerate)
sf.write('newfile1.wav', data, samplerate)
