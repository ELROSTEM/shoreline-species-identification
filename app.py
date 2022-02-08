import os
from datetime import datetime

# import cv2 as cv
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from streamlit_echarts import st_echarts


def animal_sighting_data():
    option = {
    "legend": {},
    "tooltip": {"trigger": "axis", "showContent": False},
    "dataset": {
        "source": [
            ["shellfish", "2012", "2013", "2014", "2015", "2016", "2017"],
            ["Hard Clam", 56.5, 82.1, 88.7, 70.1, 53.4, 85.1],
            ["Soft Shell Clam", 51.1, 51.4, 55.1, 53.3, 73.8, 68.7],
            ["Eastern Oyster", 40.1, 62.2, 69.5, 36.4, 45.2, 32.5],
            ["Atlantic Bay Scallop", 25.2, 37.1, 41.2, 18, 33.9, 49.1],
        ]
    },
    "xAxis": {"type": "category"},
    "yAxis": {"gridIndex": 0},
    "grid": {"top": "55%"},
    "series": [
        {
            "type": "line",
            "smooth": True,
            "seriesLayoutBy": "row",
            "emphasis": {"focus": "series"},
        },
        {
            "type": "line",
            "smooth": True,
            "seriesLayoutBy": "row",
            "emphasis": {"focus": "series"},
        },
        {
            "type": "line",
            "smooth": True,
            "seriesLayoutBy": "row",
            "emphasis": {"focus": "series"},
        },
        {
            "type": "line",
            "smooth": True,
            "seriesLayoutBy": "row",
            "emphasis": {"focus": "series"},
        },
        {
            "type": "pie",
            "id": "pie",
            "radius": "30%",
            "center": ["50%", "25%"],
            "emphasis": {"focus": "data"},
            "label": {"formatter": "{b}: {@2012} ({d}%)"},
            "encode": {"itemName": "shellfish", "value": "2012", "tooltip": "2012"},
        },
    ],
}
    st_echarts(option, height="500px", key="echarts")

def  atlantic_bay_scallop():
    """Info for atlantic bay scallop"""
    st.session_state['animal'] = 'Atlantic Bay Scallop'

    st.subheader("Atlantic Bay Scallop")
    st.image(img, use_column_width=True, clamp= True)

    st.subheader("Info:")
    st.markdown("""
        Atlantic bay scallops are also known simply as "Bay Scallops". Their shells are ribbed and have a distinctive wing-like hinge. They can grow to approximately three inches in length. The shells vary in color and can be blue-black, orange, reddish-brown, or white. Within the shells is a single adductor muscle that closes the two shells tightly together. This muscle allows the scallop to clap its shells quickly and strongly which propels the animal through the water. It's the adductor muscle that is the only part of the scallop commonly eaten. Unlike most bivalves, the bay scallop does not have a muscular foot for digging or a siphon for water intake.

        Along the edge of a bay scallop shell are 30 to 40 bright blue eyes. Each eye has a cornea, a lens, an optic nerve, and a retina which enables it to see movements and shadows. This allows them to detect predators. In addition to eyes they also have tentacles along the edge of their shells. The tentacles contain cells that are sensitive to chemicals in the water. These cells help the animal react to its environment. Bay scallops grow quickly, and rarely live past three years of age.

        Habitat: East Coast from Cape Cod to the Gulf of Mexico. On Long Island, Bay Scallops are mostly found in the small bays and harbors, most notably in the Peconic Bay Estuary which lies on the eastern end of Long Island. Bay scallops have also been found in Great South Bay, Moriches Bay, and Shinnecock Bay. Their preferred habitats are within eelgrass beds on sandy and sandy-mud bottoms. Juvenile bay scallops use byssal threads to attach themselves to aquatic plants and rocks to keep them away from predators. They prefer salinities ranging from 31 ppt to 32.8 ppt.
    """)
    st.caption("Info from: https://www.dec.ny.gov/animals/117470.html")

    st.subheader("Data:")
    animal_sighting_data()

def hard_clam():
    """Info for Hard Clam"""
    st.session_state['animal'] = 'Hard Clam'

    st.subheader("Hard Clam")
    st.image(img, use_column_width=True, clamp= True)

    st.subheader("Info:")
    st.markdown("""
        Hard clams are also known as "Chowder Clams", "Northern Quahogs" and "Round Clams". These clams have hard, thick shells that can grow to about four inches in length. The shell's color is pale brown with a distinctive purple stain on the interior. They have a muscular foot which they use to burrow deep into the sediment.

        Hard clams are a commercially important species that are harvested both by commercial fishermen (baymen) and recreational clammers. They have an average lifespan of four to eight years, however specimens as old as 40 years of age have been found. Hard clams are also given common names based on their size:

        Littlenecks: Smallest hard clams able to harvest, 1" thickness (vertically). Often served raw on the half-shell.

        Middlenecks/Topnecks: Average 2" across the shell (horizontally). Can be served raw, steamed, or grilled.

        Cherrystones: Approximately 2.5" across. Versatile use for raw, cooked, and baked.

        Chowders: Larger than 3" across. Best used chopped and cooked in soups or as fried clams.

        Habitat: East coast of the United States, ranging from the Gulf of St. Lawrence to the Gulf of Mexico. They are found in the sand and mud habitats of the intertidal and sheltered subtidal zones. The intertidal zone is the area of marine shoreline that is exposed to air at low tide and covered with seawater at high tide. The subtidal zone is the area of marine shoreline that is below the intertidal zone and is submerged most of the time, portions of this zone may be exposed briefly during extreme low tides around full and new moons. Hard clams can tolerate salinities ranging from 15 ppt to 35 ppt.
    """)
    st.caption("Info from: https://www.dec.ny.gov/animals/117470.html")

    st.subheader("Data:")
    animal_sighting_data()





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
    if preds == 0:
        atlantic_bay_scallop()
    elif preds == 1:
        hard_clam()



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
