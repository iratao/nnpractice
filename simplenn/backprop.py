import numpy as np
from data_prep import features, features_test, targets, targets_test
import pdb

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
	"""
    Derivative of the sigmoid function
    """
	return sigmoid(x) * (1 - sigmoid(x))


n_record, n_features = features.shape
n_hidden = 2
learnrate = 0.005
epochs = 900
last_loss = None

weights_input_hidden = np.random.normal(scale=1 / n_features ** .5, size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5, size=n_hidden) # can not understand scale here??

for i in np.arange(epochs):
	delta_w_i_h = np.zeros(weights_input_hidden.shape)
	delta_w_h_o = np.zeros(weights_hidden_output.shape)

	hidden_layer_input = np.dot(features, weights_input_hidden) # hidden_layer_input (360, 2), features (360, 6), weights_input_hidden (6, 2)
	hidden_layer_output = sigmoid(hidden_layer_input) # (360, 2)

	output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) # output_layer_input (360, )
	output_layer_output = sigmoid(output_layer_input) # nn output
	# pdb.set_trace()


	# Backward pass
	error = targets - output_layer_output
	output_error_term = error * sigmoid_prime(output_layer_input) # (360, )
	hidden_error = np.dot(output_error_term[:, None], weights_hidden_output[None, :])
	hidden_error_term = hidden_error * sigmoid_prime(hidden_layer_input)
	# pdb.set_trace()

	delta_w_h_o = learnrate * np.dot(output_error_term, hidden_layer_output) / n_record
	delta_w_i_h = learnrate * np.dot(features.T, hidden_error_term) / n_record
	# pdb.set_trace()

	weights_input_hidden += delta_w_i_h
	weights_hidden_output += delta_w_h_o
	# pdb.set_trace()

	if i % (epochs / 10) == 0:
		hidden_output = sigmoid(np.dot(features, weights_input_hidden))
		out = sigmoid(np.dot(hidden_output, weights_hidden_output))
		loss = np.mean((out - targets)**2)
		if last_loss and last_loss < loss:
			print("Train loss: ", loss, "  WARNING - Loss Increasing")
		else:
			print("Train loss: ", loss)
		last_loss = loss

hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))


# ## Forward pass
# hidden_layer_input = np.dot(x, weights_input_hidden)
# hidden_layer_output = sigmoid(hidden_layer_input)

# output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
# output = sigmoid(output_layer_in)

# ## Backwards pass
# ## TODO: Calculate output error
# error = target - output

# # TODO: Calculate error term for output layer
# output_error_term = error * output * (1 - output)

# # TODO: Calculate error term for hidden layer
# hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
#                     hidden_layer_output * (1 - hidden_layer_output)

# # TODO: Calculate change in weights for hidden layer to output layer
# delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# # TODO: Calculate change in weights for input layer to hidden layer
# delta_w_i_h = learnrate * hidden_error_term * x[:, None]

# print('Change in weights for hidden layer to output layer:')
# print(delta_w_h_o)
# print('Change in weights for input layer to hidden layer:')
# print(delta_w_i_h)
