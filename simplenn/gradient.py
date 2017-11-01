import numpy as np
from data_prep import features, features_test, targets, targets_test

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    """
    # Derivative of the sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))


np.random.seed(42)
n_records, n_features = features.shape
last_loss = None

# reshape your data - necessary??
targets = targets.values.reshape(-1, 1)
targets_test = targets_test.values.reshape(-1, 1)

# Initialize weights
w = np.random.normal(scale=1/n_features**0.5, size=n_features)
# w.shape (6,)
w = w.reshape(-1, 1)
learnrate = 0.5
epochs = 1000

# train model
# features.shape (360, 6)
# targets.shape (360, 1)
# features_test (40, 6)
# targets_test (40, 1)
for i in np.arange(epochs):
	del_w = np.zeros(w.shape)
	h = np.matmul(features, w) # h.shape = (360, 1)
	nn_output = sigmoid(h) # nn_output.shape = (360, 1)
	error = targets - nn_output # error.shape = (360, 1)
	error_term = error * sigmoid_prime(h) # error_term.shape = (360, 1)
	del_w += np.dot(features.T, error_term) 
	w += learnrate * del_w / n_records

	if i % (epochs / 10) == 0:
		out = sigmoid(np.dot(features, w))
		loss = np.mean((out - targets) ** 2)
		if last_loss and last_loss < loss:
			print("Train loss: ", loss, "  WARNING - Loss Increasing")
		else:
			print("Train loss: ", loss)
		last_loss = loss


# test model
h = np.matmul(features_test, w)
test_output = sigmoid(h)
predictions = test_output > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))