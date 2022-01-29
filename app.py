from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("Species Detection")

#----------------------------------------------------------
#Form

form = st.empty()
with form.container():
    # Form
    with st.form(key='species_form'):
        
        # Wall section input for where the shellfish was found.
        # I want to have like an animation of the selection process. So if 1 section 1 will light up
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
        wall_section = st.radio("Wall Section",("1","2","3"))


        #Take Picture
        picture = st.camera_input("Take a picture")
        # or use file uploader
        picture = st.file_uploader("Upload an image")
        st.caption("Unable to use the camera? Use the file uploader and select take photo.")

        submit_button = st.form_submit_button(label='Submit')

if submit_button == True:
    form.empty()

    # AI --------------------------------------

    # Save Image
    img = Image.open(picture)
    img = img.save("img.jpg")

    # Read Image
    img = cv2.imread("img.jpg")

    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

    ## slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    ## save 
    cv2.imwrite("green.png", green)

    # -------------------------------------------------------
    # Info write / Display

    date = datetime.now().strftime("%d/%m/%Y")
    time = datetime.now().strftime("%H:%M:%S")

    #Write the data into database
    # body = [date, time, age, alge, shape, review]
    # wks.append_row(body, table_range="A1:G1")

    #Gets rid of form so user can't submit twice
    st.success(f"Submission Time: {date} {time}. Thank you for filling out the form!")
    st.balloons()
    
    # Display info to user
    st.header("So what did you detect?")
    st.image(img, use_column_width=True, clamp= True)
    st.image(green, use_column_width=True,clamp = True)

    st.header("Here is information on your species")
    st.header("Also here is a graph of how many times your species was detected at our park")
    st.markdown("Hope you enjoyed the park!")


