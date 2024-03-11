"""_summary_
Name: Photo To Story
Reference: AI Jason https://www.youtube.com/watch?v=_j7JEDWuqLE
Description:
Chain together 3 LLM using Langchain / Hugging Face.

Pipeline Steps:
1. Upload photo and get LLM to create a description of the photo
2. Pass description of photo as context to a LMM to generate a short story.
3. Pass story to LLM to to generate a voice over of the story.

Note: to run app, first update the config file with your own API keys and secrets.

Returns:
    audio: Output a flac audio file.
"""
