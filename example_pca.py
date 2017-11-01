import sys
import numpy as np 
from subspace import pca
from util import normalize, asRowMatrix, read_images
from visual import subplot

import matplotlib.cm as cm

[X, y] = read_images('./faces')
print(np.asarray(X).shape)
print(asRowMatrix(X).shape)
# [D, W, mu] = pca(asRowMatrix(X), y)

# E=[]
# for i in xrange(min(len(X), 16)):
# 	e = W[:, i].reshape(X[0].shape)
# 	E.append(normalize(e, 0, 255))

# subplot(title='Eigenface', images=E, rows=4, cols=4, sptitle='Eigenface', colormap=cm.jet, filename='python_pca_eigenfaces.png')