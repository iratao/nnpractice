import numpy as np 

def pca(X, y, num_components=0):
	[n,d] = X.shape
	if (num_components <= 0) or (num_components>n):
		num_components = n
	mu = X.mean(axis=0)
	X = X - mu
	if n>d:
		C = np.dot(X.T,X)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
	else:
		C = np.dot(X,X.T)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
		for i in xrange(n):
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	# or simply perform an economy size decomposition
	# eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
	# sort eigenvectors descending by their eigenvalue
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	# select only num_components
	eigenvalues = eigenvalues[0:num_components].copy()
	eigenvectors = eigenvectors[:,0:num_components].copy()
	return [eigenvalues, eigenvectors, mu]

def pca2(X, y, num_components=0):
	[n, d] = X.shape
	if num_components <= 0 or num_components >n:
		num_components = n
	mu = X.mean(axis=0)
	X = X - mu
	if n > d:
		S = np.dot(X.T, X)
		print('before eigen')
		[eigenvalues, eigenvectors] = np.linalg.eigh(S)
		print('after eign')
	else:
		S = np.dot(X, X.T)
		print('before eigen')
		[eigenvalues, eigenvectors] = np.linalg.eigh(S)
		print('after eign')
		eigenvectors = np.dot(X.T, eigenvectors)
		for i in xrange(n): # why n?
			eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:, idx]
	# select only num_components
	eigenvalues = eigenvalues[0:num_components].copy()
	eigenvectors = eigenvectors[:, 0:num_components].copy()
	return [eigenvalues, eigenvectors, mu]
	
def project(W, X, mu=None):
	if mu is None:
		return np.dot(X, W)
	return np.dot(X - mu, W)



