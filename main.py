import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


class CustomFaceDetectionModel(nn.Module):
    def __init__(self):
        super(CustomFaceDetectionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 4)  # Assuming output is (x, y, w, h)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the pre-trained custom face detection model
model_path = 'yolov5/data.yaml'  # Ensure this is the correct path to your model file
face_detection_model = CustomFaceDetectionModel()

try:
    face_detection_model.load_state_dict(torch.load(model_path))
    face_detection_model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {model_path}")
except Exception as e:
    print(f"An error occurred: {e}")

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# MediaPipe face detection setup
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Convert the RGB image back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face detection annotations on the image
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)

                # Ensure coordinates are within the frame bounds
                if x < 0 or y < 0 or x + w > iw or y + h > ih:
                    continue

                # Crop and process the face region with the custom model
                face_region = image[y:y + h, x:x + w]
                face_region = cv2.resize(face_region, (224, 224))
                face_region = face_region.transpose((2, 0, 1))  # HWC to CHW
                face_region = torch.tensor(face_region, dtype=torch.float32).unsqueeze(0) / 255.0

                print(f"Face region shape: {face_region.shape}")  # Debug statement

                with torch.no_grad():
                    output = face_detection_model(face_region)
                    print(f"Model output: {output}")  # Debug statement

                # Draw the refined bounding box
                x, y, w, h = output[0].numpy()
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw the MediaPipe detected bounding box for comparison
                mp_drawing.draw_detection(image, detection)

        # Display the image
        cv2.imshow('Face Detection', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
