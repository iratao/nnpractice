import os, sys
import numpy as np 
# from scipy.misc import imread, imsave, imresize
from imageio import imread
import PIL.Image as Image

def normalize(X, low, high, dtype=None):
	X = np.asarray(X)
	minX, maxX = np.min(X), np.max(X)
	X = X - float(minX) # why float?
	X = X / float(maxX - minX)
	X = X * (high - low)
	X = X + low
	if dtype is None:
		return np.asarray(X)
	return np.asarray(X, dtype = dtype)


def read_images(path, sz=None):
	c = 0
	X, y = [], []
	for dirname, dirnames, filenames in os.walk(path):
		for subdirname in dirnames:
			subject_path = os.path.join(dirname, subdirname)
			for filename in os.listdir(subject_path):
				try:
					print(os.path.join(subject_path, filename))
					im = imread(os.path.join(subject_path, filename))
					X.append(np.asarray(im, dtype=np.uint8))
					y.append(c)
				except IOError:
					print 'I/O error({0}): {1}'.format(errno, strerror)
				except:
					print 'Unexpected error:', sys.exc_info()[0]
					raise
			c = c + 1
	return [X, y]


def read_image(path):
	img = imread(path)
	return img

def asRowMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0, X[0].size), dtype=X[0].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
	return mat

def asColumnMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
	for col in X:
		mat = np.hstack((mat, np.asarray(col).reshape(-1,1)))
	return mat


if __name__ == '__main__':
	[X, y] = read_images('./faces')
	[n, d] = np.asarray(X).shape
	print(d)
	# matrix = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
	# print(asColumnMatrix(matrix))
	# image = read_image('./faces/yalefaces/subject01.happy')
	# print(image.shape)


		