from dotenv import load_dotenv

load_dotenv()  # Load all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import io

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load Google Gemini Pro Vision API And get response

def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Resize the image to 250x150 pixels
        image_resized = image.resize((250, 150))
        
        # Convert resized image to bytes
        img_byte_array = io.BytesIO()
        image_resized.save(img_byte_array, format='JPEG')
        bytes_data = img_byte_array.getvalue()

        image_parts = [
            {
                "mime_type": 'image/jpeg',  # Assuming JPEG format after resizing
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

## Initialize our Streamlit app

st.set_page_config(page_title="Gemini Social Media post description App")

st.header("Gemini Social Media post description App")
input_prompt = st.text_input("Input Prompt:", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Write the Description")

input_prompt = """Determine Image Content:

Look closely at the uploaded image.
Identify and describe what is prominently featured in the image. Is it a person, animal, or something else? Provide as much detail as possible about the content.
Area Detection:

Consider the background and setting of the image.
Describe the location or scene where the image was likely taken. Is it indoors or outdoors? If outdoors, is it in a natural or urban environment? Specify if it's a house, street, restaurant, shop, office, or another relevant area."""

## If submit button is clicked

if submit:
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt, image_data, input_prompt)
    st.subheader("The Response is")
    st.write(response)
