
import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from config import features, salesDependingFeatures, model_load
from sklearn.preprocessing import LabelEncoder
from numpy import loadtxt
from tensorflow.python.keras.models import load_model
import dice_ml
from dice_ml import Dice
import fasttreeshap
import shap

# st.set_page_config(layout="wide")
#feature update
def feature_update(li,features):
    final_features = []
    for i in li:
        for j in features:
         if(i == j):
                final_features.append(features.index(j))
    return final_features

def feature_engineering(data):
    # turn gender to binary 
    gender_dict = {'F':0, 'M':1}
    data["Gender"] = data["Gender"].apply(lambda line: gender_dict[line])
    city_dict = {'A':0, 'B':1, 'C':2}
    data["City_Category"] = data["City_Category"].apply(lambda line: city_dict[line])
    #convert stay_in_current_city_years to binary 
    return data


def cf_ml(df, features_BF, continuous_features_BF, target, model, var_explain, num_cf, desired_target):
    dice_data = dice_ml.Data(dataframe=df[features_BF], continuous_features=continuous_features_BF, outcome_name=target)
    # We provide the type of model as a parameter (model_type)
    dice_model = dice_ml.Model(model=model, backend="sklearn", model_type='regressor')
    exp_genetic_BF = Dice(dice_data, dice_model, method="genetic")

    genetic_BF = exp_genetic_BF.generate_counterfactuals(var_explain, total_CFs=num_cf, desired_range=desired_target)
    genetic_BF.visualize_as_dataframe(show_only_changes=True)

    return genetic_BF.cf_examples_list[0].final_cfs_df

def fast_treeshap(loaded_model, train_df, predictors):
    shap_explainer = fasttreeshap.TreeExplainer(loaded_model, data=train_df[predictors] ,algorithm = "v2", n_jobs = n_jobs, shortcuts = True, feature_perturbation='interventional')
    shap_values_v2 = shap_explainer(train_df[predictors][:4])
    st.image(shap.plots.waterfall(shap_values_v2[0]))

#page functioning
def app():

    html1 = '''
                <style>
                #heading{
                  color: #E65142;
                  text-align:top-left;
                  font-size: 45px;
                }
                </style>
                <h1 id = "heading"> Sales Data Prediction</h1>
            '''
    st.markdown(html1, unsafe_allow_html = True)
    
    html5='''<p style="font-size: 15px">
    # User can <b>manually input a specific user characteristics</b> by selecting different features. 
     The data will then go through <b>XGBoost regression</b> to predict that user Big Sales Day spending</p>'''

    st.markdown(html5,unsafe_allow_html=True)


    li = st.multiselect("Select the feature/features whose value can be manually updated ",features)
    list = feature_update(li,features)
    value = []
    for i in list:
        if features[i] == 'Gender':
            string = st.text_input("Enter the values " +features[i]+" (M/F)")
            print(string)
            value.append(string)
        elif features[i]=='City_Category':
            string = st.text_input("Enter the values " +features[i] +" (A/B/C)")
            value.append(string.upper())
        elif features[i] == 'Age':
            number = st.number_input("Enter the values " +features[i])
            print(number)
            print(type(number))
            if number <= 17:
                value.append(0)
            elif number > 17 and number <= 25:
                value.append(1)
            elif number > 25 and number <= 35:
                value.append(2)
            elif number > 35 and number <= 45:
                value.append(3)
            elif number > 45 and number <= 50:
                value.append(4)
            elif number > 51 and number <= 55:
                value.append(5)
            else :
                value.append(6)
        else:
            number = st.number_input("Enter the values " +features[i])
            value.append(number)
    
    for i in range(len(list)):
        salesDependingFeatures[list[i]] = value[i]
    
    salesdata = {} 
    if len(salesDependingFeatures) == len(features):  
        for j in range(len(salesDependingFeatures)):
            salesdata[features[j]] = salesDependingFeatures[j]
    
    print(salesdata)
    df = pd.DataFrame (salesdata, index=[0]) 
    sale_predict_data = feature_engineering(df)
    print('INPUT DATA CHECKING')
    print(sale_predict_data)
    products_list = sale_predict_data.values.tolist()
    print(products_list)
    
    model_list = ["XGBoost","Deep Neural Network Model"]
    selected_list = st.selectbox("Select the type of model that u want to use",model_list)
    if (selected_list == "XGBoost"):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print("Enter process part")
        print(sale_predict_data)
        model_load = r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost_lastest.sav"
        loaded_model = pickle.load(open(model_load, 'rb'))
        # train_df_shap = pd.read_csv(
        #     r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train2_modified.csv')
        # predictors = train_df_shap.columns.drop(['Purchase', 'Product_ID', 'User_ID'])
        XGresult = loaded_model.predict(sale_predict_data)
        st.subheader("The Predicted Value using XGBoost regressor")
        st.write(XGresult)

        #XAI part for counterfactual
        st.subheader('Counterfactual Input')
        cf_range = [8000,9000]
        values = st.slider('Select a range of values', 2000, 20000, (8000, 9000))
        if st.button('Process'):
            st.success('Amount range confirmed. Processing......')  # displayed when the button is clicked
            print(values)
            cf_range[0]= values[0]
            cf_range[1] = values[1]
            df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\BF_modified.csv')
            features_BF = df.columns.drop(['Product_ID', 'User_ID'])
            continuous_features_BF = df.drop(['Purchase', 'Product_ID', 'User_ID'], axis=1).columns.tolist()
            target = 'Purchase'
            query_instances_BF = sale_predict_data
            model_load = r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost_lastest.sav"
            loaded_model = pickle.load(open(model_load, 'rb'))
            cf_result = cf_ml(df, features_BF, continuous_features_BF, target, loaded_model, query_instances_BF, 2,cf_range)

        # dice_data = dice_ml.Data(dataframe=df[features_BF], continuous_features=continuous_features_BF, outcome_name=target)
        # dice_model = dice_ml.Model(model=loaded_model, backend="sklearn", model_type='regressor')
        # exp_genetic_BF = Dice(dice_data, dice_model, method="genetic")
        #
        # genetic_BF = exp_genetic_BF.generate_counterfactuals(query_instances_BF, total_CFs=2,
        #                                                      desired_range=[8000.0, 9000.0])
        # st.write(genetic_BF.visualize_as_dataframe(show_only_changes=True))
        # counterfactual_result = genetic_BF.cf_examples_list[0].final_cfs_df
            st.subheader("Explained counterfactual result ")
            st.dataframe(cf_result)
        st.subheader("Implementation of SHAP testing")
        train_df_shap = pd.read_csv(
            r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train2_modified.csv')
        predictors = train_df_shap.columns.drop(['Purchase', 'Product_ID', 'User_ID'])
        n_jobs = -1
        model_load = r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost_lastest.sav"
        loaded_model = pickle.load(open(model_load, 'rb'))
        shap_explainer = fasttreeshap.TreeExplainer(loaded_model, data=train_df_shap[predictors], algorithm="v2",
                                                    n_jobs=n_jobs, shortcuts=True,
                                                    feature_perturbation='interventional')
        shap_values_v2 = shap_explainer(sale_predict_data)
        # shap.plots.waterfall(shap_values_v2[0])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(shap.plots.waterfall(shap_values_v2[0]))
    else:
        print("execute dnn here")
        NN_model = load_model(r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\NN_model.h5")
        NNresult = NN_model.predict(sale_predict_data)
        st.subheader("The Predicted Value using XGBoost regressor")
        st.write(NNresult)

    d ={"Feature ":features, "Value for Prediction": products_list[0]}
    st.subheader("Default values- ")
    st.write(pd.DataFrame(d))
    model_load = r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost_lastest.sav"
    loaded_model = pickle.load(open(model_load, 'rb'))
    model_ = loaded_model.predict(sale_predict_data)
    st.subheader("The Predicted Value using XGBoost regressor")
    st.write(model_)
    html6 = '''
    <hr class="rounded">
                <style>
                #heading{
                  color: #E65142;
                  text-align:top-left;
                  font-size: 45px;
                }
                </style>
                <h1 id = "heading"> XAI model</h1>
            '''
    st.markdown(html6, unsafe_allow_html = True)
    
    # XAI implmentation example
    # df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\BF_modified.csv')
    # features_BF = df.columns.drop(['Product_ID', 'User_ID'])
    # continuous_features_BF = df.drop(['Purchase', 'Product_ID', 'User_ID'], axis=1).columns.tolist()
    # target = 'Purchase'
    # query_instances_BF = sale_predict_data
    # model_load = r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost_lastest.sav"
    # loaded_model = pickle.load(open(model_load, 'rb'))
    # cf_result = cf_ml(df,features_BF,continuous_features_BF,target,loaded_model,query_instances_BF,2,[8000.0,9000.0])
    #
    # # dice_data = dice_ml.Data(dataframe=df[features_BF], continuous_features=continuous_features_BF, outcome_name=target)
    # # dice_model = dice_ml.Model(model=loaded_model, backend="sklearn", model_type='regressor')
    # # exp_genetic_BF = Dice(dice_data, dice_model, method="genetic")
    # #
    # # genetic_BF = exp_genetic_BF.generate_counterfactuals(query_instances_BF, total_CFs=2,
    # #                                                      desired_range=[8000.0, 9000.0])
    # # st.write(genetic_BF.visualize_as_dataframe(show_only_changes=True))
    # # counterfactual_result = genetic_BF.cf_examples_list[0].final_cfs_df
    # st.subheader("Explained counterfactual result ")
    # st.dataframe(cf_result)


    #ftshap implementation trying
    # st.subheader("Implementation of SHAP testing")
    # train_df = pd.read_csv(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train2_modified.csv')
    # predictors = train_df.columns.drop(['Purchase','Product_ID','User_ID'])
    # n_jobs = -1
    # model_load = r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xgboost_lastest.sav"
    # loaded_model = pickle.load(open(model_load, 'rb'))
    # shap_explainer = fasttreeshap.TreeExplainer(loaded_model, data=train_df[predictors] ,algorithm = "v2", n_jobs = n_jobs, shortcuts = True, feature_perturbation='interventional')
    # shap_values_v2 = shap_explainer(sale_predict_data)
    # # shap.plots.waterfall(shap_values_v2[0])
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.pyplot(shap.plots.waterfall(shap_values_v2[0]))
    #     explainer = shap.Explainer(model)
#     shap_values = explainer(train_df[predictors])
#     shap.plots.waterfall(shap_values[0])
#     shap.plots.bar(shap_values)
#     shap.plots.beeswarm(shap_values)
#     shap.plots.force(shap_values[:100])




