# Deadlift Classification Prototype
A prototype app developed in an attempt to differentiate between the deadlift styles of conventional, Romanian, and sumo, while also evaluating execution form between the classes of correct, early hip elevation, back overextension, and rounded back. 

Requirements:
- Python version 3.9 was used to be compatible with keras and tensorflow versions 2.10.0 (windows GPU compatability for machine learning was deprecated past this version). This was solely for GPU accelerated machine learning and app code may function with different versions of keras and tensorflow. 
- All required libraries are listed in "requirements.txt".
- Webcam for the app to function.
- Gemini LLM requires a private API key to function. My own personal API key is removed from the app code. See https://ai.google.dev/api?lang=python to get and provide a private API key for the API_key variable if you want to test the code. 

Contents:
- MoveNet Models contains Google's MoveNet Thunder and Lightning models for human pose estimation and keypoint generation. See https://www.tensorflow.org/hub/tutorials/movenet for more information.
- app.py is the main app script.
- utils.py contains functions for MoveNet Thunder inference, recorder logic and interpolation of keypoint arrays which are loaded into app.py.
- KPS Annotated MP4 Videos are the deadlift clips WITH keypoint and vector overlay generated with MoveNet Thunder Model.
- KPS arrays are the raw keypoint generated values generated with MoveNet Thunder model.
- Raw MP4 Videos are the raw video clips of deadlift executions.
- Base Classifications/LSTM Base Classification contains a jupyter notebook file from LSTM base classification training, the saved training history in the form of a pickle file, and the saved model itself.
- Subclass Conv Classifications/LSTM Conv Subclass Classification contains a jupyter notebook file from LSTM Conventional subclass training, the saved training history, and the saved model.
- Subclass Romanian Classifications/LSTM R Subclass Classification contains a jupyter notebook file from LSTM Romanian subclass training, the saved training history, and the saved model.
- Subclass Sumo Classifications/LSTM S Subclass Classifications contains a jupyter notebook file from LSTM Sumo subclass training, the saved training history, and the saved model. 

How to run app:
1. Start your python environment.
2. Install the needed libraries from requirements.txt.
3. Run the app as "python app.py".
4. Move at a length where your entire body is seen by the webcam feed. 
5. Perform deadlift with the webcam recording your body from the side. The recorder logic should trigger once your hip angle drops below the threshold, and it should end once you've stood up and dropped down below the same angle threshold (aligning with the execution of one repetition).
6. Hit P to make prediction once the message states so. Sometimes P has to be hit repeatedly to run the prediction.
7. Hit Escape to exit window.
