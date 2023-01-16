import streamlit as st
from PIL import Image
import home
import data_visulization
import model
import Preprocessing
import test_page
import info
from streamlit_option_menu import option_menu
import dice_ml

# selected = option_menu(
#     menu_title=None,
#     options=["Home", "Visualization", " Pre-processing", "Predictions", "About Me"],
#     icons= ["house","bar-chart","terminal","piggy-bank-fill","person-lines-fill"],
#     menu_icon= "cast",
#     default_index=0,
#     orientation="horizontal",

#     # style={
#     #     "container": "{"padding":"0!important", "background": "transparent"}
#     #     :icon: {""}
#     # }
# )


selected = option_menu(menu_title="Final Year Project", options= ["Home", "Visualization", "Pre-processing", "Predictions","Global Explanations", "About Me"],
                    icons=["house","bar-chart","terminal","piggy-bank-fill","answer","person-lines-fill"],
                    menu_icon="app-indicator", default_index=0, orientation="horizontal",
                    styles={
"container": { "background-color": "#fafafa","width": "100%",},
"icon": {"color": "orange", "font-size": "20px"}, 
"nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#eee"},
"nav-link-selected": {"background-color": "#02ab21"},
}
)

fsktm_image = r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\download.png'
fbe_image = r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\UM_faculty_economics_business.jpeg'
#Pages in the app
PAGES = {
    "Home": home,
    "Visualization": data_visulization,
    "Pre-processing": Preprocessing,
    "Predictions": model,
    "Global Explanations": test_page,
    "About Me": info
    
    
}

page = PAGES[selected]
page.app()

# #background styling
# page_bg = '''
# <style>
# body {
# background-color : #f4f4f4;
# }
# </style>
# '''
# st.markdown(page_bg, unsafe_allow_html=True)

# #navbaar styling
# st.markdown(
#     """
# <style>
# .sidebar .sidebar-content {
#     background-image: linear-gradient(#292929,#E65142;9);
#     color: black;
#     align-text: center;
# }
# hr.rounded {
#         border-top: 6px solid #E65142;
#         border-radius: 5px;
#     }
# </style>
# """, unsafe_allow_html=True,
# )

# #inseting image in the sidebar
# fsktm = Image.open(fsktm_image)
# fbe = Image.open(fbe_image)
# st.sidebar.image(fsktm)
# st.sidebar.image(fbe)
# #navbar content-1
# html3 = '''
# <h2 style="text-align: center;">Final Year Project 1</h2>
# <p style="text-align: center; font-size: 15px">Chan Jia Wei 17204964/1<br>
# Ng Yong Zhang 17205238/1<br>
# This project is about using <i>explainable AI</i> on prediction model for end user to understand the top drivers that effecting the result.</p>
# <hr class="rounded">
# '''
# st.sidebar.markdown(html3, unsafe_allow_html=True)

# st.sidebar.title("Project Modules")

# #radio selection for the pages
# selection = st.sidebar.radio("", list(PAGES.keys()))
# page = PAGES[selection]
# page.app()

# #navbar content-2
# html4 = '''
# <br>
# <br>
# <p><b>Data Visualization -</b> Analyze the sales data accross various quarters of financial year 2014 for <i>Rossmann Stores</i> using 
# multiple charts-box, line, bar, time series etc.</p>
# <br>
# <p><b>Data Prediction -</b> Predict the future sales set to different features or columns. Predictions can be made using default entries
# or select one or multiple fields to be manually edited for predictions.</p>
# '''  
# st.sidebar.markdown(html4, unsafe_allow_html=True)
