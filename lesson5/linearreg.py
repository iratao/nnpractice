import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
from sklearn.datasets import load_boston


def predictChallengeDataSet():
	#read data
	dataframe = pd.read_csv('challenge_dataset.csv')
	x_values = dataframe['A'].values.reshape(-1, 1)
	y_values = dataframe['B'].values.reshape(-1, 1)
	# print(x_values)
	# print(y_values)

	#train model on data
	body_reg = linear_model.LinearRegression()
	body_reg.fit(x_values, y_values)

	plt.scatter(x_values, y_values)
	plt.plot(x_values, body_reg.predict(x_values))
	plt.show()

def predictBIMLife():
	#read data
	bmi_life_data = pd.read_csv('bmi_and_life_expectancy.csv')
	bmi_life_data['CountryCode'] = pd.Categorical(bmi_life_data['Country']).codes
	x_values = bmi_life_data.loc[:,['CountryCode', 'BMI']].values
	y_values = bmi_life_data['Life expectancy'].values.reshape(-1, 1)

	#train model
	bmi_life_model = linear_model.LinearRegression()
	bmi_life_model.fit(x_values, y_values)
	print(bmi_life_model.predict(x_values))

	#visualize
	fig = plt.figure(1)
	ax = fig.add_subplot(211, projection='3d')
	ax.plot_surface(x_values[:, 0].reshape(-1,1), x_values[:, 1].reshape(-1,1), bmi_life_model.predict(x_values))
	ax.set_xlabel('Country')
	ax.set_ylabel('BMI')
	ax.set_zlabel('Life expectancy')
	ax = fig.add_subplot(212, projection='3d')
	ax.plot_wireframe(x_values[:, 0].reshape(-1,1), x_values[:, 1].reshape(-1,1), bmi_life_model.predict(x_values))
	ax.set_xlabel('Country')
	ax.set_ylabel('BMI')
	ax.set_zlabel('Life expectancy')
	plt.show()

def predictHousePrice():
	#read data
	boston_house_data = load_boston()
	x_values = boston_house_data['data']
	y_values = boston_house_data['target']

	#train model
	model = LinearRegression()
	model.fit(x_values, y_values)

	# Mak a prediction using the model
	sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
	                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
	                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
	prediction = model.predict(sample_house)

# predictBIMLife()
predictHousePrice()
