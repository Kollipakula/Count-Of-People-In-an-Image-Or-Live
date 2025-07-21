from mtcnn import MTCNN
import cv2

# Initialize the MTCNN face detector
detector = MTCNN()

# Function to process video frames and display face count dynamically
def process_video(video_path, confidence_threshold=0.90):
    # Capture video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        detections = detector.detect_faces(frame_rgb)

        # Initialize a counter for detected faces
        face_count = 0

        # Draw bounding boxes for detected faces
        for detection in detections:
            x, y, width, height = detection['box']
            confidence = detection['confidence']

            # Only consider faces with high confidence
            if confidence > confidence_threshold:
                face_count += 1
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the current frame with bounding boxes and face count
        cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Face Detection in Video", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Path to your video
# video_path = '"C:\Users\Nripesh\Downloads\855564-hd_1920_1080_24fps.mp4"'  # Your video path
video_path = 0


# Process the video
process_video(video_path, confidence_threshold=0.90)




