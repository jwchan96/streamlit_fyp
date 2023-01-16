import dice_ml
from dice_ml import Dice
import pandas as pd 
import xgboost 
import pickle 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import  metrics
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from xgboost import cv 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import fasttreeshap
#Define target and ID columns:
target = 'Purchase'
IDcol = ['User_ID','Product_ID']



#model building 
df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\BF_modified.csv')
train_df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train2_modified.csv')
test_df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\test2_modified.csv')

predictors = train_df.columns.drop(['Purchase','Product_ID','User_ID'])
test_x = test_df.columns.drop(['Purchase','Product_ID','User_ID'])

# xgb_reg = xgboost.XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

# xgb_reg.fit(train_df[predictors], train_df[target])
# xgb_y_pred = xgb_reg.predict(test_df[predictors])
# print(mean_absolute_error(test_df[target], xgb_y_pred))
# print(mean_squared_error(test_df[target], xgb_y_pred))
# print(r2_score(test_df[target], xgb_y_pred))
# from math import sqrt
# print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(test_df[target], xgb_y_pred)))




# frames = [train_df, test_df]
# full_dataset = pd.concat(frames)
# scaler = StandardScaler()
# x_trainScaled = scaler.fit_transform(train_df[predictors])
# X_testScaled = scaler.fit_transform(test_df[predictors])
# params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10}

# cv_results = xgboost.cv(dtrain=full_dataset, params=params, nfold=3,
#                     num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

X = df.drop(['Purchase','Product_ID','User_ID'], axis=1)

y = df['Purchase']

data_dmatrix = xgboost.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# print(X_train.head())
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.head())

# params = {
#             'objective':"reg:linear",
#             'colsample_bytree': 0.3,'learning_rate': 0.5,
#                 'max_depth': 5, 'alpha': 10
#         }         
           
          
# instantiate the classifier 
# xgb_clf = xgboost.XGBRegressor(**params)


# fit the classifier to the training data
# xgb_clf.fit(X_train, y_train)
# print(xgb_clf)
# train_df_predictions = xgb_clf.predict(X_train)
# # make predictions
# predictions = xgb_clf.predict(X_test)

# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_df[target])))
# print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_df[target]).values, train_df_predictions)))

# preds = xgb_clf.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, preds))
# print("RMSE: %f" % (rmse))


## result 
# Fitting 5 folds for each of 54 candidates, totalling 270 fits
# Best parameters: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 500}
# Lowest RMSE:  2852.385257531773

# params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10}

# xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3,
#                     num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)


target = 'Purchase'
my_model = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10,colsample_bytree=0.7)
my_model.fit(train_df[predictors], train_df[target], early_stopping_rounds=5, 
             eval_set=[(test_df[predictors], test_df[target])], verbose=False)

# save model using pickle 
# filename = r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost_lastest.sav'
# pickle.dump(my_model, open(filename, 'wb'))
# print('done saving model')


 #Predict training set:
train_df_predictions = my_model.predict(train_df[predictors])
# make predictions
predictions = my_model.predict(test_df[test_x])

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_df[target])))
print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error((train_df[target]).values, train_df_predictions)))

# First result  without tune parameter is Mean Absolute Error : 2167.1927689994577    RMSE : 2862
# Second result with tune parameter is Mean Absolute Error : 2150.6792587178056       RMSE : 2775





# find the best parameter of the model (XGBoost) 
'''
params = { 'max_depth': [3,6,10],
           'learning_rate': [0.01, 0.05, 0.1],
           'n_estimators': [100, 500, 1000],
           'colsample_bytree': [0.3, 0.7]}
xgbr = xgboost.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='neg_mean_squared_error', 
                   verbose=1)
clf.fit(X, y)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
'''

## result 
# Fitting 5 folds for each of 54 candidates, totalling 270 fits
# Best parameters: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 500}
# Lowest RMSE:  2852.385257531773


# second method Random Search method 
'''
from sklearn.model_selection import RandomizedSearchCV

params = { 'max_depth': [3, 5, 6, 10, 15, 20],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
           'n_estimators': [100, 500, 1000]}
xgbr = xgboost.XGBRegressor(seed = 20)
clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring='neg_mean_squared_error',
                         n_iter=25,
                         verbose=1)
clf.fit(X, y)
print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
'''

#result 
# Best parameters: {'subsample': 0.7, 'n_estimators': 500, 'max_depth': 20, 'learning_rate': 0.01, 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7}
# Lowest RMSE:  2867.5428338885567





# preds = my_model.predict(test_df[test_x])
# rmse = np.sqrt(mean_squared_error(test_df[target], preds))
# print("RMSE: %f" % (rmse))





