from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt

# Initialize the MTCNN face detector
detector = MTCNN()

# Read the input image
image_path = 'D:\Dwdm_project\People1.jpg'  # Replace with your image path
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MTCNN

# Perform face detection
detections = detector.detect_faces(img_rgb)

# Initialize a counter for detected faces
face_count = 0

# Draw bounding boxes for detected faces
for detection in detections:
    x, y, width, height = detection['box']
    confidence = detection['confidence']
    
    # Only consider faces with high confidence
    if confidence > 0.90:
        face_count += 1
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(img, f"{confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Print the total count of faces
print(f"Number of faces detected: {face_count}")

# Convert BGR to RGB for displaying with matplotlib
img_rgb_with_faces = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
plt.figure(figsize=(16, 12))
plt.imshow(img_rgb_with_faces)
plt.axis('off')
plt.title(f"Detected Faces: {face_count}")
plt.show()

# Optionally, save the result image
cv2.imwrite('/content/faces_detected.jpg', img)



