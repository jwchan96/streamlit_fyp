import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# st.set_page_config(layout="wide")
sale_data =  r"C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\train.csv"
train = pd.read_csv(sale_data)
    
def data_preprocessing():
    body = ''
    st.caption(body, unsafe_allow_html=False)



def duo_variable(first_var, second_var):
    pivot = train.pivot_table(index=first_var, values=second_var, aggfunc=np.mean)
    pivot.plot(kind='bar', color='blue',figsize=(12,7))
    plt.xlabel(first_var)
    plt.ylabel(second_var)
    title = first_var + 'and' + second_var + 'Analysis'
    plt.title(title)
    plt.xticks(rotation=0)
    plt.show()
    st.pyplot()
    
def matrix():
    numeric_features = train.select_dtypes(include=[np.number])
    corr = numeric_features.corr()
    f, ax = plt.subplots(figsize=(20, 9))
    sns.heatmap(corr, vmax=.8,annot_kws={'size': 20}, annot=True);
    st.pyplot()

def plot_purchase_amount_distribution(train):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(12,7))
    sns.distplot(train.Purchase, bins = 25)
    plt.xlabel('Amount spent in Purchase')
    plt.ylabel('Number of Buyers')
    plt.title('Purchase amount Distribution')
    st.pyplot()


def univaraite_plot(ele):
    sns.countplot(ele)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    

# date specific plotting
def dateplot(x, y, df, k):
    sns.catplot(x=x, y=y, data=df, kind=k,height=7, aspect=11.7/7)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

#plotting the Sales vs StoreType
def Univariate_Analysis():
    col1, col2 = st.columns(2)
    type_chart = col1.selectbox("Select the Chart type", ('Bar','None'))
    data1 = col2.selectbox("Select the distribution of variable", ('Purchase', 'Occupation', 'Gender', 'Marital Status', 
                            'Product_Category_1', 'Product_Category_2','Product_Category_3','Age','City_Category','Stay_In_Current_City_Years','Purchase'))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if (data1 == 'Purchase'):
        plot_purchase_amount_distribution(train)
    elif (data1 == 'Occupation'):
        univaraite_plot(train.Occupation)
    elif (data1 == 'Gender'):
        univaraite_plot(train.Gender)
    elif (data1 == 'Marital Status'):
        univaraite_plot(train.Marital_Status)
    elif (data1 == 'Product_Category_1'):
        univaraite_plot(train.Product_Category_1)
    elif (data1 == 'Product_Category_2'):
        univaraite_plot(train.Product_Category_2)
    elif (data1 == 'Product_Category_3'):
        univaraite_plot(train.Product_Category_3)
    elif (data1 == 'Age'):
        univaraite_plot(train.Age)
    elif (data1 == 'City_Category'):
        univaraite_plot(train.City_Category)
    else:
        univaraite_plot(train.Stay_In_Current_City_Years)

#plotting correlation matrix 
def Correlation():
    matrix()


#plotting the month vs sales
def occupation_purchase():
    duo_variable('Occupation','Purchase')

def product_cat_purchase():
    duo_variable('Product_Category_1','Purchase')

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
            <h1 id = "heading"> Sales Data Analyses</h1>
        '''
    st.markdown(html1, unsafe_allow_html=True)

    b1 = st.selectbox("Select the Analyses Point",('Univariate Analysis','Correlation between Numerical Predictors and Target variable','Occupation vs Purchase','Product Category 1 vs Purchase'))
    if(b1 == 'Univariate Analysis'):
        Univariate_Analysis()
    elif(b1 == 'Correlation between Numerical Predictors and Target variable'):
        Correlation()
    elif(b1 == 'Occupation vs Purchase'):
        occupation_purchase()
    elif(b1 == 'Product Category 1 vs Purchase'):
        product_cat_purchase()






