import mediapipe as mp
import cv2
import os
import pygame

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

# camera setup
cap = cv2.VideoCapture(0)

# pygame setup
pygame.init()
X = 1280
Y = 720
screen = pygame.display.set_mode((X, Y))
clock = pygame.time.Clock()
running = True
font = pygame.font.Font('freesansbold.ttf', 32)
display_surface = pygame.display.set_mode((X, Y))

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

        # Print coordinates on screen
        os.system('clear')
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            print(f"Landmark #{i}: x: {round(landmark.x, 3)}, y: {round(landmark.y, 3)}, z: {round(landmark.z, 3)}")

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit (on the camera window)
        running = False

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(30)  # limits FPS to 30

cap.release()
cv2.destroyAllWindows()
pygame.quit()