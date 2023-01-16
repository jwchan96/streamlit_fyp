import requests
import streamlit as st
from PIL import Image
import fasttreeshap
import shap
import time
import os

# st.set_page_config(layout="wide")
# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
# st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

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

def fast_treeshap(loaded_model, train_df, predictors):
    shap_explainer = fasttreeshap.TreeExplainer(loaded_model, data=train_df[predictors] ,algorithm = "v2", n_jobs = n_jobs, shortcuts = True, feature_perturbation='interventional')
    shap_values_v2 = shap_explainer(train_df[predictors][:4])
    st.image(shap.plots.waterfall(shap_values_v2[0]))

def kernelshap(NN_model, X_train):
    X_train_summary = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(model = NN_model.predict, data = X_train_summary)
    sv = explainer.shap_values(xxx[:4])
    shap.summary_plot(sv, xxx[:4])


def app():
    # number = st.text_input('Number', value=20)
    # st.write(number)
    # list = []
    # if st.button("Form"):
    #     st.subheader("User Info Form")
    #
    #     # name = st.text_input("Name")
    #     with st.form(key='user_info'):
    #         st.write('User Information')
    #
    #         num1 = st.text_input(label="Name 📛")
    #         num2 = st.number_input(label="Age 🔢")
    #         # email = st.text_input(label="Email 📧")
    #         # phone = st.text_input(label="Phone 📱")
    #         # gender = st.radio("Gender 🧑", ("Male", "Female", "Prefer Not To Say"))
    #
    #         submit_form = st.form_submit_button(label="Register", help="Click to register!")
    #
    #         st.write(submit_form)
    #
    #         # Checking if all the fields are non empty
    #         if submit_form:
    #             st.write(submit_form)
    #
    #             # if name and age and email and phone and gender:
    #             #     # add_user_info(id, name, age, email, phone, gender)
    #             #     st.success(id + "\n" + age + "\n" + email + "\n" + phone + "\n" + gender)
    #             # else:
    #             #     st.warning("Please fill all the fields")

    b1 = st.selectbox("Select the type of global analysis graph", (
    'Feature Importance', 'Beeswarm', 'Waterfall', 'Forceplot'))
    if (b1 == 'Feature Importance'):
        html2 = '''
            <h3 id = "sub_heading1">Global XAI in feature importance &emsp;&emsp;&emsp;&emsp;</h3>
            '''
        st.markdown(html2, unsafe_allow_html=True)
        feature_importance = Image.open(
            r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xai_feature_importance2.png')
        st.image(feature_importance, use_column_width=True)

    elif (b1 == 'Beeswarm'):
        html2 = '''
            <h3 id = "sub_heading1">Global XAI in Beeswarm format &emsp;&emsp;&emsp;&emsp;</h3>
            '''
        st.markdown(html2, unsafe_allow_html=True)
        beeswarm = Image.open(
            r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xai_beeswarm.png')
        st.image(beeswarm, use_column_width=True)

    elif (b1 == 'Waterfall'):
        html2 = '''
            <h3 id = "sub_heading1">Global XAI in Beeswarm format &emsp;&emsp;&emsp;&emsp;</h3>
            '''
        st.markdown(html2, unsafe_allow_html=True)
        waterfall = Image.open(
            r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xai_waterfall.png.png')
        st.image(waterfall, use_column_width=True)

    elif (b1 == 'Forceplot'):
        html2 = '''
            <h3 id = "sub_heading1">Global XAI in Beeswarm format &emsp;&emsp;&emsp;&emsp;</h3>
            '''
        st.markdown(html2, unsafe_allow_html=True)
        forceplot = Image.open(
            r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xai_waterfall.png.png')
        st.image(forceplot, use_column_width=True)

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
    st.markdown(html1, unsafe_allow_html=True)

    html5 = '''<p style="font-size: 15px">
        #Select the range of your prefered spending/b> 
         The data will then go through <b>Diverse Counterfactual Explanations
         </b> to suggest which attributes need to be adjusted to get the prefered spending amount</p>'''

    st.markdown(html5, unsafe_allow_html=True)



    html7='''<p style="font-size: 15px">
    # The XAI model that we choose to explain the prediction of customer spending is <b>SHAP</b>.
     SHAP (SHapley Additive exPlanations) is a game theoretic approach to <b>explain the output of our machine learning model</b></p>'''

    st.markdown(html7,unsafe_allow_html=True)
    html2 = '''
    <h3 id = "sub_heading1">Explainable AI example waterfall&emsp;&emsp;&emsp;&emsp;</h3>
    '''
    st.markdown(html2, unsafe_allow_html=True)
    waterfall = Image.open(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xai_waterfall.png')
    st.image(waterfall, use_column_width=True)
    html3 = '''
    <hr class="rounded">
    <h3 id = "sub_heading1">Explainable AI example force plot &emsp;&emsp;&emsp;&emsp;</h3>
    '''
    st.markdown(html2, unsafe_allow_html=True)
    force_plot = Image.open(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xai_forceplot.png')
    st.image(force_plot, use_column_width=True)

    html4 = '''
    <hr class="rounded">
    <h3 id = "sub_heading1">Explainable AI example- beeswarm graph &emsp;&emsp;&emsp;&emsp;</h3>
    '''
    st.markdown(html4, unsafe_allow_html=True)
    beeswarm = Image.open(r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\xai_beeswarm.png')
    st.image(beeswarm, use_column_width=True)





# # ---- PROJECTS ----
# with st.container():
#     st.write("---")
#     st.header("My Projects")
#     st.write("##")
#     image_column, text_column = st.columns((1, 2))
#     with image_column:
#         st.image(img_lottie_animation)
#     with text_column:
#         st.subheader("Integrate Lottie Animations Inside Your Streamlit App")
#         st.write(
#             """
#             Learn how to use Lottie Files in Streamlit!
#             Animations make our web app more engaging and fun, and Lottie Files are the easiest way to do it!
#             In this tutorial, I'll show you exactly how to do it
#             """
#         )
#         st.markdown("[Watch Video...](https://youtu.be/TXSOitGoINE)")
# with st.container():
#     image_column, text_column = st.columns((1, 2))
#     with image_column:
#         st.image(img_contact_form)
#     with text_column:
#         st.subheader("How To Add A Contact Form To Your Streamlit App")
#         st.write(
#             """
#             Want to add a contact form to your Streamlit website?
#             In this video, I'm going to show you how to implement a contact form in your Streamlit app using the free service ‘Form Submit’.
#             """
#         )
#         st.markdown("[Watch Video...](https://youtu.be/FOULV9Xij_8)")