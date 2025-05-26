import tensorflow as tf
import keras
import numpy as np

#matplotlib.use("TkAgg")  # Enables GUI window popups for plots
# Video handling
import cv2
import time

# Import utility functions for MoveNet Inference and Angle Calculations
from utils import movenet, run_inference, init_crop_region, determine_crop_region, draw_prediction_on_image
from utils import return_angle, DeadliftRepDetector
from utils import interpolate_kps

# Gemini LLM
API_key = "" # Gemini LLM requires a private API key to function. See https://ai.google.dev/api?lang=python
from google import genai
client = genai.Client(api_key=API_key)
import textwrap

# Initialize the TFLite interpreter
input_size = 256 # 192 for lightning, 256 for thunder

import os

model_path = os.path.join(os.path.dirname(__file__), "MoveNet Models", "singlepose_thunder_tflite_f16.tflite")
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

# Classification Models
# Label dictionaries
base_classes = {0: "Romanian", 1: "Sumo", 2: "Conventional"}
conv_classes = {0: "Correct", 1: "Early hip elevation", 2: "Overextension", 3: "Rounded back"}
sumo_classes = {0: "Correct", 1: "Early hip elevation", 2: "Overextension", 3: "Rounded back"}
romanian_classes = {0: "Correct", 1: "Overextension", 2: "Rounded back"}

# Loading of models
# Build model paths relative to current script
base_dir = os.path.dirname(__file__)

base_model_path = os.path.join(
    base_dir, "Base Classifications", "LSTM Base Classification", "CNN-LSTM base.keras"
)
R_model_path = os.path.join(base_dir, "Subclass Romanian Classifications", "LSTM R Subclass Classification", "LSTM R Classifier.keras")
conv_model_path = os.path.join(base_dir, "Subclass Conv Classifications", "LSTM Conv Subclass Classification", "LSTM Conv Classifier.keras")
S_model_path = os.path.join(base_dir, "Subclass Sumo Classifications", "LSTM S Subclass Classification", "LSTM Sumo Classifier.keras")

base_model = keras.models.load_model(base_model_path)
R_model = keras.models.load_model(R_model_path)
conv_model = keras.models.load_model(conv_model_path)
S_model = keras.models.load_model(S_model_path)

# Open the webcam and rescale window to cater for text outputs
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
display_width = frame_width + 1200
display_height = frame_height + 600

# OpenCV print statements formatting
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.8
gemini_font_scale = 2.2
text_thickness = 2
gemini_text_thickness = 2
line_spacing = 30
y_offset = 150 # Starting pos of gemini feedback

# Get video properties
fps_start = time.time()
frame_count = 0

# Recording logic
start_threshold = 85 # Less than this triggers record start (and also stop after top_threshold is reached)
top_threshold = 160 # Hip angle needs to surpass this before angles drop below end_threshold
end_threshold = 85 # Once start_ and top_ have been reached, falling below this threshold stops recording
add_frames = 2 # Add additional frames once end recording logic is reached to not cut off repetitions too early
min_conf_score = 0.1
recorder = DeadliftRepDetector(start_threshold, top_threshold, end_threshold, add_frames)
make_pred = False # Converted to True when 'P' is hit during opencv loop
done_pred = False # Logic to keep while loop from making subsequent predictions

# Store KPS and frames in memory to predict and visualize
kps = []
gif_index = 0
frames_for_gif = []

# Function to take 10 frames and print these
def retrieve_frames(frames_for_gif):
   indices = np.linspace(0, len(frames_for_gif) - 1, 10, dtype=int)
   selected = [frames_for_gif[i] for i in indices]
   return selected


# Initial crop region
crop_region = init_crop_region(frame_height, frame_width)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1
    elapsed_time = time.time() - fps_start
    if elapsed_time > 0:
        estimated_fps = frame_count / elapsed_time

    movenet_input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    movenet_input_image = tf.image.resize_with_pad(movenet_input_image, input_size, input_size)

    # Run MoveNet Inference
    keypoints_with_scores = run_inference(movenet, movenet_input_image, crop_region, crop_size=[input_size, input_size], interpreter=interpreter)
    
    # Draw KPS on frames resized for CNN (matching 480x480 size of dataset)
    movenet_output_overlay = draw_prediction_on_image(
      movenet_input_image.numpy().astype(np.int32),
      keypoints_with_scores,
      close_figure=True,
      output_image_height=480)
    
    # Update cropping region for next frame
    crop_region = determine_crop_region(keypoints_with_scores, image_height=frame_height, image_width=frame_width)

    # Calculate angles
    angle_1, angle_2 = return_angle(keypoints_with_scores, min_conf_score) # Returns hip angle for both sides of the body ('angle_1', 'angle_2')

    # Rep detection class
    recorder.update(angle_1, angle_2)

    if recorder.recording:
        frames_for_gif.append(movenet_output_overlay)
        kps.append(keypoints_with_scores[0, 0]) # (17, 3), every append == one frame

    # Displaying webcam feed with inference
    # frame_output = cv2.cvtColor(frame_output, cv2.COLOR_RGB2BGR)
    frame_display = np.zeros((display_height, display_width, 3), dtype=np.uint8) # Canvas for app
    frame_output_bgr = cv2.cvtColor(movenet_output_overlay, cv2.COLOR_RGB2BGR)
    frame_output_resized_display = cv2.resize(frame_output_bgr, (frame_width, frame_height))
    frame_display[:frame_height, :frame_width] = frame_output_resized_display
    
    if make_pred and not done_pred:
      # Preparing GIF visualization
      frames_for_gif = retrieve_frames(frames_for_gif)

      # Preparing array for prediction
      prepared_array = interpolate_kps(np.array(kps))
      prepared_array = np.expand_dims(prepared_array, axis=0)

      base_prediction = base_model.predict(prepared_array)

      predicted_number = np.argmax(base_prediction)
      base_label = base_classes[predicted_number]

      if base_label == "Conventional":
        form_prediction = conv_model.predict(prepared_array)
        form_label = conv_classes[form_prediction.argmax()]
      elif base_label == "Sumo":
        form_prediction = S_model.predict(prepared_array)
        form_label = sumo_classes[form_prediction.argmax()]
      elif base_label == "Romanian":
        form_prediction = R_model.predict(prepared_array)
        form_label = romanian_classes[form_prediction.argmax()]

      # Gemini prompt for LLM model response
      gemini_prompt = f"""
        You are given an input of style and form, which are outputs of a machine learning model for deadlift classification in two steps.
        The first deadlift classification step classifies the style as either Conventional, Sumo, or Romanian. 
        The second deadlift classification step classifies the form as either Correct, Early Hip Elevation, Overextension, or Rounded Back.
        Clues to what determines the different styles and forms are given below, which should be used to help generate the needed feedback.
        
        You are a technically skilled but supportive strength coach. Your task is to provide concise, biomechanically-informed feedback based on the predicted deadlift style and form quality.

        Start the feedback with a clear acknowledgment of the lift style and form, where you provide clues to how to rectify form if its classified as 
        Rounded back, Early hip elevation, or Overextension, if the form is classified as Correct then you should provide positive feedback and encourage the user 
        to continue the good work. 

        Available deadlift styles:
        - Conventional: Feet shoulder-width, grip outside knees. Requires a neutral spine and coordinated hip-knee extension.
        - Sumo: Wide stance, externally rotated feet, grip inside knees. Prioritizes vertical torso and hip abduction.
        - Romanian (RDL): Similar stance to conventional, slight knee flexion (~15°), lift initiated with a hip hinge. Emphasis on posterior chain and spinal alignment.

        Detected form issues and feedback:
        - Rounded back: Keep a neutral spine. Engage your core and lats to avoid spinal flexion under load. Try to look upwards.
        - Early hip elevation: Do not let your hips shoot up first. Brace your core and lift your chest with the hips to keep the movement synchronized.
        - Overextension: Avoid leaning back at lockout. Stop when your spine and hips reach a neutral upright position.
        - Correct: Excellent form! Keep reinforcing proper technique, tension, and alignment throughout the lift.

        Now respond to this:
        Style: {base_label}
        Form: {form_label}
      """

      gemini_response = client.models.generate_content(model="gemini-2.0-flash", contents=gemini_prompt)
      done_pred = True

    if done_pred:
      gif_frame = frames_for_gif[gif_index % len(frames_for_gif)]
      gif_index += 1

      gif_frame_bgr = cv2.cvtColor(gif_frame, cv2.COLOR_RGB2BGR)
      gif_frame_resize = cv2.resize(gif_frame_bgr, (304, 304))

      # Manual hard-coded position
      x_start = 168
      y_start = 600

      # Draw Labels
      cv2.putText(frame_display, "Repetition Playback", (x_start + 50, y_start - 60), font, 0.8, (0, 255, 0), 1)
      if "base_label" in globals() and "form_label" in globals():
        cv2.putText(frame_display, f"Style: {base_label}", (x_start + 70, y_start - 40), font, 0.7, (0, 255, 0), 1)
        cv2.putText(frame_display, f"Execution: {form_label}", (x_start + 70, y_start - 20), font, 0.7, (0, 255, 0), 1)

      # Paste the GIF frame
      frame_display[y_start:y_start + 304, x_start:x_start + 304] = gif_frame_resize


    # Print Recorder Logic
    if recorder.status_msg:
      cv2.putText(frame_display, recorder.status_msg, (frame_width + 10, 100), font, font_scale * 0.8, (0, 255, 0), text_thickness)

    # Print prediction text once predictions are done
    if recorder.done_recording and "base_label" in globals() and "form_label" in globals():
      wrapped_lines = textwrap.wrap(gemini_response.text, width=80) # Wrapping at 40 character length without splitting words
      y_offset = 250
      for line in wrapped_lines:
        cv2.putText(frame_display, line, (frame_width + 10, y_offset), font, gemini_font_scale*0.4, (0, 255, 0), gemini_text_thickness)
        y_offset += line_spacing

    cv2.imshow('MoveNet Webcam Inference', frame_display)

    # Press "escape" to quit, "P" for prediction once done recording
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
      break
    elif key == ord("p") and recorder.done_recording and not done_pred:
      make_pred = True

      
# Cleanup
cap.release()
cv2.destroyAllWindows()