# little_medical_reader
This app will help you read your patient's notes.
It will convert the notes content to a visual format after extracting the data.
Always anotate the code to explain in detail what it does.
Change the code in small verifiable steps and implement only one feature at a time.


# python code
Always use uv to manage the project and its dependencies.
Verify the imports and paths before implementing a feature.
At the end of the script create small tests to test the features.

# how it works
The patient's notes file will be in one of these 3 formats:
- pdf - the app will convert it to markdown
- markdown
- txt

# graphical interface
The app will have a UI built with streamlit.
The code for the UI is in the folder /UI

# workflow chain
Workflow chains using LLM and AI agents will use langchain

# LLM and AI agents
The app uses google genai agents. The model that will be used for testing is gemini-2.0-flash due to the quality and low price.
Local models using ollama will be used as local offline option.

