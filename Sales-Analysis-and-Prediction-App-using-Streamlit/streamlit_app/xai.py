import fasttreeshap 
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
import os 
import shap
import matplotlib.pyplot as plt 
import time 
import dice_ml
from dice_ml import Dice
 

# compute SHAP values/SHAP interaction values via TreeSHAP algorithm with version "algorithm_version"
# (parallel on "n_jobs" threads)
def run_fasttreeshap(model, data, sample, interactions, algorithm_version, n_jobs, num_round, num_sample, shortcut = True):
    shap_explainer = fasttreeshap.TreeExplainer(
        model, data = data , algorithm = algorithm_version, n_jobs = n_jobs, shortcut = shortcut, feature_perturbation='interventional')
    run_time = np.zeros(num_round)
    for i in range(num_round):
        start = time.time()
        shap_values = shap_explainer(sample.iloc[:num_sample], interactions = interactions).values
        run_time[i] = time.time() - start
        print("Round {} takes {:.3f} sec.".format(i + 1, run_time[i]))
    print("Average running time of {} is {:.3f} sec (std {:.3f} sec){}.".format(
        algorithm_version, np.mean(run_time), np.std(run_time), " (with shortcut)" if shortcut else ""))

def memory_estimate_v2(shap_explainer, num_sample, num_feature, n_jobs):
    max_node = max(shap_explainer.model.num_nodes)
    max_leaves = (max_node + 1) // 2
    max_combinations = 2**int(shap_explainer.model.max_depth)
    phi_dim = num_sample * (num_feature + 1) * shap_explainer.model.num_outputs
    n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
    memory_1 = (max_leaves * max_combinations + phi_dim) * 8 * n_jobs
    memory_2 = max_leaves * max_combinations * shap_explainer.model.values.shape[0] * 8
    memory = min(memory_1, memory_2)
    if memory < 1024:
        print("Memory usage of FastTreeSHAP v2 is around {:.2f}B.".format(memory))
    elif memory / 1024 < 1024:
        print("Memory usage of FastTreeSHAP v2 is around {:.2f}KB.".format(memory / 1024))
    elif memory / 1024**2 < 1024:
        print("Memory usage of FastTreeSHAP v2 is around {:.2f}MB.".format(memory / 1024**2))
    else:
        print("Memory usage of FastTreeSHAP v2 is around {:.2f}GB.".format(memory / 1024**3))


#Define target and ID columns:
target = 'Purchase'
IDcol = ['User_ID','Product_ID']

num_sample = 10000 # number of samples to be explained
n_jobs = -1

#model building 
df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\BF_modified.csv')
train_df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train2_modified.csv')
test_df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\test2_modified.csv')

predictors = train_df.columns.drop(['Purchase','Product_ID','User_ID'])
test_x = test_df.columns.drop(['Purchase','Product_ID','User_ID'])

X = df.drop(['Purchase','Product_ID','User_ID'], axis=1)
test = test_df.drop(['Purchase','Product_ID','User_ID'], axis=1)
# print(train_df[predictors][:4])
y = df['Purchase']
print(test_df[predictors][:1])
data_dmatrix = xgboost.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

target = 'Purchase'
# my_model = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10,colsample_bytree=0.7)
# my_model.fit(train_df[predictors], train_df[target], early_stopping_rounds=5, 
#              eval_set=[(test_df[predictors], test_df[target])], verbose=False)
# v1 = my_model.predict(test_df[predictors][:1])
# print(v1)
# print('************************************************')

model_load = r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost_lastest.sav"
loaded_model = pickle.load(open(model_load, 'rb'))
# v2 = loaded_model.predict(test_df[predictors][:1])
# print(v2)

# shap_explainer = fasttreeshap.TreeExplainer(loaded_model)
# val = shap_explainer(train_df[predictors][:4]).values
# print(val.shape)
# num_leaves = sum(shap_explainer.model.num_nodes) - sum(sum(shap_explainer.model.children_left > 0))
# print("Total number of leaves is {}.".format(num_leaves))

# memory_estimate_v2(shap_explainer, num_sample, test.shape[1], n_jobs)  check_additivity = False, feature_perturbation='interventional'
shap_explainer = fasttreeshap.TreeExplainer(loaded_model, data=train_df[predictors] ,algorithm = "v2", n_jobs = n_jobs, shortcuts = True, feature_perturbation='interventional')
# shap_values_v2 = shap_explainer(train_df[predictors][:4])
# print(shap_values_v2)

# shap explainer 
# explainer = shap.Explainer(loaded_model)
# shaPvalues = explainer(train_df[predictors][:3])
# shap.plots.waterfall(shap_values_v2[0])

num_round = 3
num_sample = 100 

run_fasttreeshap(
    model = loaded_model, data = train_df[predictors], sample = test_df[predictors], interactions = False, algorithm_version = "v2", n_jobs = n_jobs,
    num_round = num_round, num_sample = num_sample, shortcut = True)




# Implementation of DICE_ML (Diverse Counterfactual Model)

my_model = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=10,colsample_bytree=0.7)
my_model.fit(train_df[predictors], train_df[target], early_stopping_rounds=5, 
             eval_set=[(test_df[predictors], test_df[target])], verbose=False)

#  #Predict training set:
# train_df_predictions = my_model.predict(train_df[predictors])
# # make predictions
# predictions = my_model.predict(test_df[test_x])


 # XAI counterfactual DICE ML Test1 

# data preparation
features_BF = test_df.columns.drop(['Product_ID','User_ID'])
continuous_features_BF = df.drop(['Purchase','Product_ID','User_ID'], axis=1).columns.tolist()
test_x = test_df.columns.drop(['Purchase','Product_ID','User_ID'])
test_x2 = pd.DataFrame(test_df[test_x])



d_housing = dice_ml.Data(dataframe=df[features_BF] , continuous_features=continuous_features_BF , outcome_name=target)
# We provide the type of model as a parameter (model_type)
m_housing = dice_ml.Model(model=my_model, backend="sklearn", model_type='regressor')
exp_genetic_BF = Dice(d_housing, m_housing, method="genetic")

query_instances_BF = test_x2[2:4]
# print(query_instances_BF)
# print( "++++++++++=")
# print(df[continuous_features_BF[:][0]])
genetic_BF = exp_genetic_BF.generate_counterfactuals(query_instances_BF, total_CFs=2, desired_range=[8000.0, 9000.0])
genetic_BF.visualize_as_dataframe(show_only_changes=True)