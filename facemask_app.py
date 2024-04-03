import requests  # type: ignore
import streamlit as st  # type: ignore
from PIL import Image, ImageDraw, ImageFont, ImageOps  # type: ignore
import io
import os
import random

FONT = ImageFont.truetype("img/Arial.ttf", 33)
API_URL = "https://bigml.io/andromeda/"

#For Production
API_USERNAME = st.secrets['BIGML_USERNAME']
API_KEY = st.secrets['BIGML_API_KEY']

#For testing environment
#API_USERNAME = os.getenv("BIGML_USERNAME")
#API_KEY = os.getenv("BIGML_API_KEY")

#API_USERNAME="prestonkhiev"
#API_KEY="e662fd42619b89fa2442c267c8ab694cd3a61f60"
API_AUTH = f"username={API_USERNAME};api_key={API_KEY}"

#V2 Model. Base dataset with resize to 640x640
MODEL = "deepnet/66027485478150ec58f66eb6"

#V7 Model. No Resize. Augmentations for rotation, sheer, blur
#MODEL = "deepnet/6607a6f0506f7b1096918782"

#V9 Model. Base Dataset, no resize
#MODEL = "deepnet/6607a903478150ec58f6734d"

PREDICTION_THRESHOLD = 0.1
MASK_CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def resize(img, width):
    """ Resize an image to a given width maintaining aspect ratio """
    percent = width / float(img.size[0])
    return img.resize((width, int((float(img.size[1]) * float(percent)))))

def detection(uploaded_file):
    # Upload image to BigML as a source
    source_response = requests.post(
        f"{API_URL}source?{API_AUTH}",
        files={"file": ("plant_image", uploaded_file)}
    )
    source = source_response.json()["resource"]
    #Generate predictions
    data = {"model": MODEL, "input_data": {"000002": source}}
    response = requests.post(f"{API_URL}prediction?{API_AUTH}", json=data)
    regions = response.json()["prediction"].get("000000", [])
    # Delete the source after it's no longer needed
    requests.delete(f"{API_URL}{source}?{API_AUTH}")
    return [r for r in regions if r[5]>PREDICTION_THRESHOLD]


def draw_predictions(pil_image, boxes):
    """ Draw BigML detected objects in the image"""
    w, h = pil_image.size
    draw = ImageDraw.Draw(pil_image)
    for box in boxes:
        label, xmin,ymin, xmax, ymax, confidence = box
        draw.rectangle(((xmin*w, ymin*h), (xmax*w, ymax*h)), width=9, outline="#eee")
        draw.text(
            (xmin*w+20, ymin*h+random.randint(10, 40)),
            f"{label}: {str(confidence)[:3]}", font=FONT, fill="red"
        )
    return ImageOps.expand(pil_image ,border=50,fill='black')


def gen_message(boxes):
    """ Generate output message for predictions """
    labels = set([box[0] for box in boxes])
    mask_classes = labels.intersection(set(MASK_CLASSES))
    if len(mask_classes) <= 0:
        st.warning('Nothing detected')
    if 'with_mask' in labels:
        st.success('Subject(s) is wearing a mask!')
    if 'mask_weared_incorrect' in labels:
        st.warning('Subject(s) is wearing mask incorrectly')
    if 'without_mask' in labels:
        st.error('Subject(s) is not wearing a mask!!')
    #if len(mask_classes) > 0:
        #st.success("Objects Detected: " + str(labels))


st.set_page_config(
    layout="wide",
    page_title="BigML Facemask Detector",
    page_icon="😷",
)

# Sidebar information
description = """ Uses a BigML deepnet to detect if a subject is wearing a facemask correctly, wearing a facemask incorrectly, or not wearing a facemask.  """
image = Image.open('img/rayray.jpg')
st.sidebar.image(image, width=100)
st.sidebar.write(description)
st.sidebar.write("Powered by [BigML](https://bigml.com)")
#st.sidebar.write(os.getenv("BIGML_USERNAME"))

# Page title
st.title("😷 BigML Face Mask Detection")


#with st.expander("⚠️ Disease detection model was trained with a small dataset. It can be inaccurate sometimes. It should be able to find the following classes: "):
    #st.write(classes) 


left, right = st.columns(2)
#right = st.columns(1)

# Example images
examples = {
    "Example 1": 'img/ex1.jpg',
    "Example 2": 'img/ex2.jpg',
    "Example 3": 'img/ex3.jpg',    
    "Example 4": 'img/ex4.jpg',
    "Example 5": 'img/ex5.jpg',
    "Example 6": 'img/ex6.jpg',
    "Example 7": 'img/ex7.jpg',
}

with left.expander()#(label=f"{API_AUTH}", expanded=True):
    option = st.selectbox('Choose one example image...', examples.keys(),index=0)
    clicked = st.button("Evaluate selected image")
    if clicked:
        example_file = open(examples[option], 'rb')

# File uploader
msg = "Or upload your own image..."
with right.form("submit", clear_on_submit=True):
    uploaded_file = st.file_uploader(msg, type=["png ", "jpg", "jpeg"])
    submitted = st.form_submit_button("Evaluate uploaded image")


file_to_predict = None
if clicked and example_file:
    file_to_predict = example_file
elif uploaded_file and submitted:
    file_to_predict = uploaded_file


# Prediction Output
if file_to_predict:
    st.subheader("Detection result")
    with st.spinner('Diagnose in progress. Please wait...'):
        boxes = detection(file_to_predict)
        image = resize(Image.open(file_to_predict), 1000)
        output_image = draw_predictions(image, boxes)
        gen_message(boxes)
        st.image(output_image, width=700)
        uploaded_file = None
