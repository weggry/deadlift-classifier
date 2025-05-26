# Deadlift Classification Prototype

Requirements:
- Python version 3.9 was used to be compatible with keras and tensorflow versions 2.10.0 (windows GPU compatability for machine learning was deprecated past this version). This was solely for GPU accelerated machine learning and app code may function with different versions of keras and tensorflow. 
- All required libraries are listed in "requirements.txt" of the Code folder.
- Webcam for the app to function.

Notes:
- Gemini LLM requires a private API key to function. My own personal API key is removed from the app code. See https://ai.google.dev/api?lang=python to get and provide a private API key for the API_key variable if you want to test the code. 

How to run app:
1. Start your python environment.
2. Install the needed libraries from requirements.txt.
3. Run the app as "python app.py".
4. Move at a length where your entire body is seen by the webcam feed. 
5. Perform deadlift with the webcam recording your body from the side. The recorder logic should trigger once your hip angle drops below the threshold, and it should end once you've stood up and dropped down below the same angle threshold (aligning with the execution of one repetition).
6. Hit P to make prediction once the message states so. Sometimes P has to be hit repeatedly to run the prediction.
7. Hit Escape to exit window.
