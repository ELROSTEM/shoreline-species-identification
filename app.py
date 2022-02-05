import os
from datetime import datetime

import cv2 as cv
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

st.title("Species Detection")


# Take Picture
picture = st.camera_input("Take a picture")
# or use file uploader
if picture is None:
    picture = st.file_uploader("Upload an image")
    st.caption("Unable to use the camera? Use the file uploader and select take photo.")

if picture is not None:
    # Read Image as PIL
    img = Image.open(picture)

    # Load Model
    model = torch.jit.load('./model/shellfish_model_scripted.pt')
    model.eval()

    ## Preprocess to tensor
    preprocess =  transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model_img = preprocess(img)
    
    # Generate prediction
    data = [model_img]
    prediction = model(data[0].unsqueeze(0))
    # # Predicted class value using argmax
    _, preds = torch.max(prediction, 1)

    st.success("Indentified!")

    # Display info to user
    st.header("So what did you detect?")
    if preds == 1:
        st.session_state['animal'] = 'Hard Clam'
        st.write("Hard Clam")
    else:
        st.session_state['animal'] = 'Atlantic Bay Scallop'
        st.write("Atlantic Bay Scallop")
    st.image(img, use_column_width=True, clamp= True)

    st.header("Here is information on your species")
    st.header("Also here is a graph of how many times your species was detected at our park")


#----------------------------------------------------------
#Form

if 'animal' in st.session_state:
    form = st.empty()
    with form.container():
        # Form
        with st.form(key='species_form'):

            st.header("Sighting")
            st.write(f"Animal Sighted: {st.session_state.animal}")
            
            st.header("Location")
            # Wall section input for where the shellfish was found.
            # I want to have like an animation of the selection process. So if 1 section 1 will light up
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
            st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
            wall_section = st.radio("Wall Section",("1","2","3"))

            # Submit button
            submit_button = st.form_submit_button(label='Submit')

    if submit_button == True:
        form.empty()

        date = datetime.now().strftime("%d/%m/%Y")
        time = datetime.now().strftime("%H:%M:%S")

        # Write in database
            # Code

        st.success(f"Submission Time: {date} {time}. Thank you for filling out the form! Hope you enjoyed the park!")
        st.balloons()
        



# Green detection using OpenCv
    # Save Image
    # save_img = img.save("img.jpg")

    # # Green
    # # # Read Image
    # img = cv.imread("img.jpg")
    # ## convert to hsv
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv.inRange(hsv, (36, 25, 25), (70, 255,255))
    # ## slice the green
    # imask = mask>0
    # green = np.zeros_like(img, np.uint8)
    # green[imask] = img[imask]
    # # ## save 
    # cv.imwrite("green.png", green)
    # st.image(green, use_column_width=True,clamp = True)

    #Write the data into database
    # body = [date, time, age, alge, shape, review]
    # wks.append_row(body, table_range="A1:G1")
