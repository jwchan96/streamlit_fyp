import pandas as pd
import tensorflow 
# import keras 
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
# from tensorflow.keras.models import Sequential
from tensorflow.python.keras.engine.sequential import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.python.keras.models import save_model
# from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn import  metrics
import numpy as np



df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train2_modified.csv')
train_X = df.drop(['Purchase','Product_ID','User_ID'], axis=1)
train_Y = df['Purchase']

df_test = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\test2_modified.csv')
X_test = df_test.drop(['Purchase','Product_ID','User_ID'], axis=1)
Y_test = df_test['Purchase']

# print(train_X.head())
# print(train_Y.head())

'''
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(9, input_shape=(9,), kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error')
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def wider_model():
 # create model
	NN_model = Sequential()
	# The Input Layer :
	NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))

	# The Hidden Layers :
	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

	# The Output Layer :
	NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
	# compile model 
	NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

	return NN_model 

'''
# first testing model 
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# # # Compile the network :
# NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# NN_model.summary()

# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]

# NN_model.fit(train_X, train_Y, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
# # after callback get the best checkpoint which is Weights-460--2155.82959.hdf5

wights_file = 'Weights-460--2155.82959.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

predictions = NN_model.predict(X_test)
train_df_predictions = NN_model.predict(train_X)


# print(predictions)
# save model and architecture to single file
# save_model(NN_model,r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\NN_model.h5")
# print("Saved model to disk")

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, Y_test)))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_Y).values, train_df_predictions)))

# Mean Absolute Error : 2154.736724258204
# RMSE result are 2986 

# second testing model 

# def wider_model():
#  # create model
# 	NN_model = Sequential()
# 	# The Input Layer :
# 	NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train_X.shape[1], activation='relu'))

# 	# The Hidden Layers :
# 	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# 	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# 	NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# 	# The Output Layer :
# 	NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
# 	# compile model 
# 	NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

# 	return NN_model 

# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(model=wider_model, epochs=100, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = cross_val_score(pipeline, train_X, train_Y, cv=kfold, scoring='neg_mean_squared_error')
# print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))











# load and evaluate a saved model
# from numpy import loadtxt
# from tensorflow.python.keras.models import load_model
 
# # load model
# model = load_model(r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\NN_model.h5")
# # summarize model.
# model.summary()

