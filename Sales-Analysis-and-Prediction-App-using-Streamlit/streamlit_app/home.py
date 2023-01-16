import pandas as pd
import streamlit as st
from PIL import Image
from config import datafields

st.set_page_config(layout="wide")
home_image = 'https://drive.google.com/file/d/1XkY4zibeFhBrHBCNi6s6SPErLT-F48Ue/view?usp=sharing'
#page functioning
def app():

    #heading and text information
    html1 = '''
    <style>
    #pred_image{
    
    }
    #heading{
      color: #E65142;
      text-align:top-left;
      font-size: 45px;
    }
    #sub_heading1{
    color: #E65142;
    text-align: right;
    font-size: 30px;
    }
    #sub_heading2{
    color: #E65142;
    text-align: left;
    font-size: 30px;
      }
    #usage_instruction{
    text-align: left;
    font-size : 15px;
    }
    #data_info{
    text-align : left;
    font-sixe : 15px;
    }
    /* Rounded border */
    hr.rounded {
        border-top: 6px solid #E65142;
        border-radius: 5px;
    }
    </style>
    <h1 id = "heading"> <i>Explainable AI</i> on spending behavior </h1>
    <h3>Online platform sales prediction during Mega Sale day.<br>
    Use XAI for understanding 'Black Box' behind the model.<br>
    This website works on the data extracted from the Kaggle<br>
    <a href = "https://www.kaggle.com/datasets/llopesolivei/blackfriday" target="_blank"><i><b>'Kaggle Black Friday dataset collected from a retail store’s purchase transactions.'</i></b></a>
    </h3>
    '''
    st.markdown(html1, unsafe_allow_html=True)
    # original = Image.open(r'C:\Users\L\fyp_demo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\banner3.jpg')
    # st.image(original, use_column_width=True)
    # st.image(home_image) #putting image 
    html2 = '''
    <hr class="rounded">
    <h3 id = "sub_heading1">Usage Description&emsp;&emsp;&emsp;&emsp;</h3>
    <p id = "usage_instruction"><br>The UI/UX for the app glides using the <b>Sidebar</b> to the left.&emsp;&emsp;<br><br>
    Access all the features of the app using it.The web app comes&nbsp;<br>
    with the features including -&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp; <br>
    <b>1. Sales Analyses using <i> Data Visulations</i> based on <i>Seaborn</i><br>
    <b>2. Data Pre-processing for <i>train data</i>&emsp;&emsp;<br>
    <b>3. Future Sale Prediction using cDecision Tree Algorithm</i>&emsp;&emsp;<br>
    <b>4. Result interprebailty  using <i>Explainable AI method</i>&emsp;&emsp;<br>
    <h3 id ="sub_heading2">Data Description&emsp;&emsp;&emsp;</h3>
    <p id ="data_info">The data is the <i>sales data</i> from <i> retailer shop<br> 
    </i> The data attributes can be viewed by pressing data fields button </p>
    '''
    st.markdown(html2, unsafe_allow_html=True)

    df = pd.read_csv(datafields)
    # data = [["Quarter1", "Jan, 2014 - Mar, 2014"], ["Quarter2", "Apr, 2014 - Jun, 2014"],["Quarter3", "Jul, 2014 - Sep, 2014"], 
    # ["Quarter4", "Oct, 2014 - Dec, 2014"]]
    # df2 = pd.DataFrame(data, columns=["Quarters", "Range"])

    #data description
    col1, col2 = st.columns(2)
    
    button1 = col1.button("Data Fields")
    if(button1):
        st.table(df)
    # button2 = col2.button("Data Breakup")
    # if(button2):
    #   st.table(df2)

    
