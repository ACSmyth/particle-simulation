import imageio
import os

path = 'C:/Users/Alex/git/particle-simulation'
images = []
files = os.listdir(path)
for i in range(len(files)):
	filename = files[i]
	if filename.endswith('.png'):
		images.append([imageio.imread(filename), filename])
		print(i)

images.sort(key=lambda fn: int(fn[1][fn[1].index('img')+3:fn[1].index('.')]))
print([e[1] for e in images])

images = [ele[0] for ele in images]
imageio.mimsave('C:/Users/Alex/git/particle-simulation/particles.gif', images, format='GIF', duration=1/500)