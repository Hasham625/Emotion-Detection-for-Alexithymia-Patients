
import cv2
from deepface import DeepFace
from collections import defaultdict
# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Define response statements based on detected emotions
response_statements = {
    # Define your response statements here...
}

# Initialize emotion counters
emotion_counts = defaultdict(int)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Update emotion counts
        emotion_counts[emotion] += 1

        # Get the response statement based on the detected emotion
        response_statement = response_statements.get(emotion, "I'm not sure how you're feeling.")

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 245), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display response statement
        cv2.putText(frame, response_statement, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Generate report with count of each emotion detected
print("Emotion Counts:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count}")

# Display the emotion with the highest count
if emotion_counts:
    max_emotion = max(emotion_counts, key=emotion_counts.get)
    print(f"\nEmotion with Highest Count: {max_emotion} ({emotion_counts[max_emotion]} occurrences)")
else:
    print("No emotions detected.")

response_statements = {
    'admiring': "Express admiration and appreciation. Respond with gratitude or reciprocate the compliment.",
    'amused': "Acknowledge the humor and enjoy the moment. Respond with laughter or a light-hearted comment.",
    'angry': "Acknowledge the frustration but remain calm. Respond with understanding or seek to address the issue.",
    'annoyed': "Acknowledge the annoyance and try to address the underlying issue. Respond with patience and understanding.",
    'approving': "Acknowledge the agreement and positivity. Respond with appreciation or reciprocate the validation.",
    'aware': "Acknowledge the understanding of the situation. Respond with clarity or further information if needed.",
    'confident': "Acknowledge the certainty and assurance. Respond with affirmation or support.",
    'confused': "Acknowledge the confusion and try to clarify. Respond with patience and willingness to explain.",
    'curious': "Acknowledge the eagerness to learn. Rpip install deepfaceespond with information or encourage further discussion.",
    'eager': "Acknowledge the enthusiasm and readiness to engage. Respond with encouragement or active participation.",
    'disappointed': "Acknowledge the letdown and try to address the issue. Respond with empathy or a solution-oriented approach.",
    'disapproving': "Acknowledge the negative judgment and seek to understand the concerns. Respond with openness to feedback.",
    'embarrassed': "Acknowledge the self-consciousness and try to reassure. Respond with empathy and understanding.",
    'excited': "Acknowledge the excitement and share in the enthusiasm. Respond with encouragement or further excitement.",
    'fearful': "Acknowledge the fear and provide reassurance. Respond with empathy and support.",
    'grateful': "Acknowledge the appreciation and reciprocate the gratitude. Respond with kindness or a supportive gesture.",
    'joyful': "Acknowledge the happiness and share in the joy. Respond with celebration or further positivity.",
    'loving': "Acknowledge the affection and reciprocate the warmth. Respond with love or a kind gesture.",
    'mournful': "Acknowledge the sorrow and offer condolences. Respond with empathy and support.",
    'neutral': "Acknowledge the impartiality and maintain objectivity. Respond with factual information or a neutral stance.",
    'optimistic': "Acknowledge the hopefulness and share in the optimism. Respond with encouragement or positive reinforcement.",
    'relieved': "Acknowledge the relief and share in the positivity. Respond with reassurance or further relief.",
    'remorseful': "Acknowledge the regret and offer forgiveness. Respond with empathy or a willingness to move forward.",
    'repulsed': "Acknowledge the aversion and try to address the issue. Respond with understanding or a solution-oriented approach.",
    'sad': "Acknowledge the sadness and offer support. Respond with empathy or a comforting gesture.",
    'worried': "Acknowledge the concern and provide reassurance. Respond with empathy and a willingness to address the worries.",
    'surprised': "Acknowledge the astonishment and share in the surprise. Respond with curiosity or further exploration.",
    'sympathetic': "Acknowledge the understanding and offer support. Respond with empathy or a willingness to listen.",
}

if emotion_counts:
    max_emotion = max(emotion_counts, key=emotion_counts.get)
    print(f"\nEmotion with Highest Count: {max_emotion} ({emotion_counts[max_emotion]} occurrences)")
    print("Response Statement:")
    print(response_statements.get(max_emotion, "No response statement available for this emotion."))
else:
    print("No emotions detected.")