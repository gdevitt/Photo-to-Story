"""_summary_
Name: Photo To Story
Description:
Chain together 3 LLM using Langchain / Hugging Face.
 
Pipeline Steps:
1. Upload photo and get LLM to create a description of the photo
2. Pass description of photo as context to a LMM to generate a short story.
3. Pass story to LLM to to generate a voice over of the story.

Returns:
    audio: Output a flac audio file.
"""
from os import getenv

from transformers import pipeline
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
import requests

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
# from langchain_openai import OpenAI

import streamlit as st


load_dotenv(find_dotenv(usecwd=True))
HUGGINGFACE_API_TOKEN = getenv("hf_api_key")
OPENAI_API_TOKEN = getenv("openai_api_key")


# img2text
def img2text(url):
    """_summary_
    Create a description of an image using the Hugging Face API.
    Args:
        url (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        image_to_text = pipeline(
            "image-to-text", model="Salesforce/blip-image-captioning-large"
        )

        text = image_to_text(url)[0]["generated_text"]

        print(text)
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# llm
def generate_story(scenario):
    """
    _summary_
    Create a short story based on the image description
    Returns:
        _type_: _description_
    """
    
    
    template = """
    You are a storyteller,
    You can generate a short story based on a single narration, the story should be no more than 20 words.
    
    CONTEXT: {scenario}
    STORY:
    """

    prompt_template = PromptTemplate.from_template(template=template)
    story_llm = LLMChain(
        llm=Ollama(model="llama2"), prompt=prompt_template, verbose=True        
    )
    story = story_llm.predict(scenario=scenario)

    print(story)
    return story


# text to speech
def text2speech(message):
    """_summary_
    Create a speech file from the text description of an image
    Args:
        message (_type_): _description_
    """
    
    API_URL = (
        "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    )
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}  # Corrected
    payloads = {
        "inputs": message,
    }

    response = requests.post(API_URL, headers=headers, json=payloads)

    if response.ok:
        with open("audio.flac", "wb") as file:
            file.write(response.content)  # Corrected to use .content
    else:
        print("Error in text-to-speech conversion:", response.text)

def main():
    """_summary_
    Wrap pipelines together into a Streamlit app.
    App will allow a user to upload a photo and generate a text 
    description of the photo and output a audio file.
    """
    st.set_page_config(page_title="Image to Audio story")

    st.header("Turn image into audio story.")
    uploaded_file = st.file_uploader("Upload an image", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()

        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)

        st.audio("audio.flac")


if __name__ == "__main__":
    main()
