import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO 
from sklearn.preprocessing import LabelEncoder
import xgboost
import pickle
import dice_ml
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split

# st.set_page_config(layout="wide")

def data_info(data):
    data.info()
    
def check_duplicates(data):
    idsUnique = len(set(data.User_ID))
    idsTotal = data.shape[0]
    idsDupli = idsTotal - idsUnique
    print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries") 

def feature_engineering(data):
    # turn gender to binary 
    gender_dict = {'F':0, 'M':1}
    data["Gender"] = data["Gender"].apply(lambda line: gender_dict[line])
    # turn age to numeric values
    age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
    data["Age"] = data["Age"].apply(lambda line: age_dict[line])
    #convert city_catgeory to binary
    city_dict = {'A':0, 'B':1, 'C':2}
    data["City_Category"] = data["City_Category"].apply(lambda line: city_dict[line])
    #convert stay_in_current_city_years to binary 
    le = LabelEncoder()
    #New variable for outlet
    data['Stay_In_Current_City_Years'] = le.fit_transform(data['Stay_In_Current_City_Years'])
    #Dummy Variables:
    data = pd.get_dummies(data, columns=['Stay_In_Current_City_Years'])
    print(data.dtypes)

def app():
    html1 = '''
            <style>
            #heading{
              color: #E65142;
              text-align:top-left;
              font-size: 45px;
            }
            </style>
            <h1 id = "heading"> Data Pre-processing</h1>
        '''
    st.markdown(html1, unsafe_allow_html=True)
    train = pd.DataFrame()
    test = pd.DataFrame()
    uploaded_files = st.file_uploader("Choose train and test CSV file", accept_multiple_files=True)
    if uploaded_files is not None:
      for uploaded_file in uploaded_files:
        if(uploaded_file.name.startswith("train")):
          train = pd.read_csv(uploaded_file)
          st.write(train)
        if(uploaded_file.name.startswith("test")):
          test = pd.read_csv(uploaded_file)
          st.write(test)
        
    button1 = st.button("Pre-process data", key=None, help=None, on_click=None, args=None, kwargs=None, disabled=False) 
    if(button1):
      if train.empty & test.empty:
        st.write('Please insert csv files')
      else:
        st.write('pre-porcess data')
        idsUnique = len(set(train.User_ID))
        idsTotal = train.shape[0]
        idsDupli = idsTotal - idsUnique
        print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")
        train['source']='train'
        test['source']='test'
        data = pd.concat([train,test], ignore_index = True, sort = False)
        data["Product_Category_2"]= \
        data["Product_Category_2"].fillna(-2.0).astype("float")
        data["Product_Category_3"]= \
        data["Product_Category_3"].fillna(-2.0).astype("float")
        #Get index of all columns with product_category_1 equal 19 or 20 from train
        condition = data.index[(data.Product_Category_1.isin([19,20])) & (data.source == "train")]
        data = data.drop(condition)
        data.apply(lambda x: len(x.unique()))
        category_cols = data.select_dtypes(include=['object']).columns.drop(["source"])
        #Print frequency of categories
        for col in category_cols:
        #Number of times each value appears in the column
          frequency = data[col].value_counts()
          st.write(frequency)
        feature_engineering(data)
        print('done converting')
        train = data.loc[data['source']=="train"]
        test = data.loc[data['source']=="test"]
        #Drop unnecessary columns:
        test.drop(['source'],axis=1,inplace=True)
        train.drop(['source'],axis=1,inplace=True)
        #Export files as modified versions:
        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">After processed train data</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write(train.head())
        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">After processed test data</p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.write(test.head())
  
        train.to_csv(r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train2_mofified.csv",index=False)
        test.to_csv(r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\test2_modified.csv",index=False)
        st.warning('processed dataset saved !!!') 
        st.write('processed dataset saved !!!')


        #model building 
        train_df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train_modified.csv')
        test_df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\test_modified.csv')
        predictors = train_df.columns.drop(['Purchase','Product_ID','User_ID'])
        target = 'Purchase'
        # my_model = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.05)
        # my_model.fit(train_df[predictors], train_df[target])
        # filename = r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost2.sav'
        # pickle.dump(my_model, open(filename, 'wb'))
        
        # DT_model = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
        # modelfit(DT_model, train_df, test_df, predictors, target, IDcol, 'DT.csv')
        # coef3 = pd.Series(DT_model.feature_importances_, predictors).sort_values(ascending=False)
        # coef3.plot(kind='bar', title='Feature Importances')
        # filename = r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost.sav'
        # pickle.dump(DT_model, open(filename, 'wb'))
