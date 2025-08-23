import mediapipe as mp
import cv2
import os
import pygame
import math

# mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Resolution variables
X = 640
Y = 480

# camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, X)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Y)

# pygame setup
pygame.init()
screen = pygame.display.set_mode((X, Y))
clock = pygame.time.Clock()
running = True
font = pygame.font.Font('freesansbold.ttf', 32)
display_surface = pygame.display.set_mode((X, Y))

# Stickman skeleton connections
stickman_connections = [
    (11, 12),           # chest
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), # torso
    (23, 24),           # hips
    (23, 25), (25, 27), # left leg
    (24, 26), (26, 28), # right leg
    (27, 29), (29, 31), # left foot
    (28, 30), (30, 32), # right foot
]

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window (game window only)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: # ESC also works
                running = False

    # fill the screen with a color to wipe away anything from last frame
    display_surface.fill("purple")

    # Read frame and display message when camera is not available
    success, image = cap.read()
    if not success:
        text = font.render('No frame detected, please check your camera and restart the game.', True, "white", "purple")
        textRect = text.get_rect()
        textRect.center = (X // 2, Y // 2)
        display_surface.blit(text, textRect)

    # Mediapipe stuff goes here
    # Convert the BGR image to RGB
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        # calculate torso hight so the head radius stays proportional
        left_shoulder  = (landmarks[11].x * X, landmarks[11].y * Y)
        left_hip = (landmarks[23].x * X, landmarks[23].y * Y)
        shoulder_dist = math.dist(left_shoulder, left_hip)

        # head circle
        head_x, head_y = landmarks[0].x * X, landmarks[0].y * Y
        pygame.draw.circle(screen, "White", (head_x, head_y), shoulder_dist * 0.25 , 4)

        # body lines
        for start_idx, end_idx in stickman_connections:
            pygame.draw.line(screen, "White", (landmarks[start_idx].x * X, landmarks[start_idx].y * Y), (landmarks[end_idx].x * X, landmarks[end_idx].y * Y), 4)


        # Print coordinates on the terminal and display landmarks on the pygame window
        os.system('clear')
        for i, landmark in enumerate(landmarks):
            print(f"Landmark #{i}: x: {round(landmark.x, 3)}, y: {round(landmark.y, 3)}, z: {round(landmark.z, 3)}")

            # display landmarks in pygame
            pygame.draw.circle(display_surface, "Green", (landmark.x * X, landmark.y * Y), 5)

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit (on the camera window)
        running = False

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(30)  # limits FPS to 30

cap.release()
cv2.destroyAllWindows()
pygame.quit()