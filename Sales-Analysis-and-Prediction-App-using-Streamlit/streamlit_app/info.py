import requests
import streamlit as st
from PIL import Image

# st.set_page_config(layout="wide")
def app():

    fsktm_image = r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\download.png'
    fbe_image = r'C:\Users\Chan Jia Wei\venv\fypdemo\Sales-Analysis-and-Prediction-App-using-Streamlit\data\UM_faculty_economics_business.jpeg'
    fsktm = Image.open(fsktm_image)
    fbe = Image.open(fbe_image)
    with st.container():
        st.subheader("Final year Project")
        st.title("Explainable AI on spending behavior")
        col1, mid, col2 = st.columns([20,1,20])
        with col1:
            st.image(fsktm, width=500)
        with col2:
            st.image(fbe,width=500)
        st.write("Supervisor: Dr Aznul Qalid Md Sabri ")
        st.write("Collobarator: Dr Phoong Seuk Wai")
        st.write("[Learn More >](https://fpe.um.edu.my/)")
   
    with st.container():
        st.subheader("Hi, I am Chan Jia Wei :wave:")
        st.title("A UM Computer Science Student majoring in Artificial Intelligence")
        st.write(
            "I am currently doing my Final Project 1 with the title Explainable AI in Spending Behaviour."
        )
        # st.write("[Learn More >](https://pythonandvba.com)")


            # ---- WHAT I DO ----
    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("What I do")
            st.write("##")
            st.write(
                """
                What I have done in FYP:
                - discover and understand the concept of Explainable AI and how it works to implement in our AI system.
                - come out with a proposal to trying to solve problems existed in market.
                - Apply skills of data analysis and machine learning knowledge to build a prototype.
                - Analysis and design the methodology of my project for Final Year Project 2.
                
                """
            )


        # ---- CONTACT ----
    with st.container():
        st.write("---")
        st.header("Get In Touch With Me!")
        st.write("##")
        st.caption("Phone no. : +60177432989 \n Email: chanjiawei2000@gmail.com")
        st.caption("For more information, contact me on [Linkedin](https://www.linkedin.com/in/chan-jiawei-20a623193/),"
        "[Instagram](),"
        "[Facebook]()")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        # contact_form = """
        # <form action="https://formsubmit.co/YOUR@MAIL.COM" method="POST">
        #     <input type="hidden" name="_captcha" value="false">
        #     <input type="text" name="name" placeholder="Your name" required>
        #     <input type="email" name="email" placeholder="Your email" required>
        #     <textarea name="message" placeholder="Your message here" required></textarea>
        #     <button type="submit">Send</button>
        # </form>
        # """
        # left_column, right_column = st.columns(2)
        # with left_column:
        #     st.markdown(contact_form, unsafe_allow_html=True)
        # with right_column:
        #     st.empty()


# def load_lottieurl(url):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


# # Use local CSS
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# local_css("style/style.css")

# ---- LOAD ASSETS ----
# lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
# img_contact_form = Image.open("images/yt_contact_form.png")
# img_lottie_animation = Image.open("images/yt_lottie_animation.png")




    

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

