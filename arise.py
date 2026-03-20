import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Use fixed landmark index numbers (stable)
        shoulder = landmarks[11]  # LEFT_SHOULDER
        ear = landmarks[7]        # LEFT_EAR
        hip = landmarks[23]       # LEFT_HIP

        shoulder = [shoulder.x, shoulder.y]
        ear = [ear.x, ear.y]
        hip = [hip.x, hip.y]

        angle = calculate_angle(ear, shoulder, hip)

        # Display angle
        cv2.putText(image, f'Angle: {int(angle)}',
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # Detect bad posture
        if angle < 160:
            cv2.putText(image, "BAD POSTURE DETECTED",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

            print('\a')  # Mac beep sound

        else:
            cv2.putText(image, "GOOD POSTURE",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 3)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Posture Corrector', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
