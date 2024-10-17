import cv2
import mediapipe as mp
import numpy as np
import pygame
import random

# Initialize MediaPipe's Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize Pygame
pygame.init()
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Gesture-Based Snake Game")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Snake settings
snake_block = 20
initial_speed = 5  # Starting speed (frames per second)
max_speed = 15     # Maximum speed
speed_increase = 0.5  # Speed increase per food eaten

# Initialize snake
snake_list = [[window_width // 2, window_height // 2]]  # Start at the center of the screen
snake_length = 1

# Initialize direction
direction = 'RIGHT'

# Initialize food
food_x = round(random.randrange(0, window_width - snake_block) / snake_block) * snake_block
food_y = round(random.randrange(0, window_height - snake_block) / snake_block) * snake_block

# Game over flag and text
game_over = False
game_over_text = pygame.font.Font('freesansbold.ttf', 64).render('Game Over', True, WHITE)

# Restart button
restart_button_text = pygame.font.Font('freesansbold.ttf', 32).render('Restart', True, WHITE)
restart_button_rect = restart_button_text.get_rect()
restart_button_rect.center = (window_width // 2, window_height // 2 + 50)

# Score and speed
score = 0
current_speed = initial_speed
score_font = pygame.font.Font('freesansbold.ttf', 24)

def draw_snake(snake_list):
    for block in snake_list:
        pygame.draw.rect(window, GREEN, [block[0], block[1], snake_block, snake_block])

def show_score_and_speed():
    score_text = score_font.render(f"Score: {score}", True, WHITE)
    speed_text = score_font.render(f"Speed: {current_speed:.1f}", True, WHITE)
    window.blit(score_text, [10, 10])
    window.blit(speed_text, [10, 40])

# Game loop
running = True
clock = pygame.time.Clock()

# Capturing video from webcam:
cap = cv2.VideoCapture(0)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and game_over:
            mouse_pos = pygame.mouse.get_pos()
            if restart_button_rect.collidepoint(mouse_pos):
                # Restart the game
                game_over = False
                snake_list = [[window_width // 2, window_height // 2]]  # Reset to center
                snake_length = 1
                direction = 'RIGHT'
                score = 0
                current_speed = initial_speed
                food_x = round(random.randrange(0, window_width - snake_block) / snake_block) * snake_block
                food_y = round(random.randrange(0, window_height - snake_block) / snake_block) * snake_block

    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Detecting hand landmarks and recognizing gestures
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger landmarks
            index_finger_landmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_landmarks.x * frame.shape[1])
            index_finger_y = int(index_finger_landmarks.y * frame.shape[0])

            # Recognize gesture based on index finger position
            if index_finger_x < frame.shape[1] // 3 and direction != 'RIGHT':
                direction = 'LEFT'
            elif index_finger_x > frame.shape[1] * 2 // 3 and direction != 'LEFT':
                direction = 'RIGHT'
            elif index_finger_y < frame.shape[0] // 3 and direction != 'DOWN':
                direction = 'UP'
            elif index_finger_y > frame.shape[0] * 2 // 3 and direction != 'UP':
                direction = 'DOWN'

            # Draw a circle at the index finger position
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)

            # Draw the recognized direction on the frame
            cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if not game_over:
        # Move the snake
        if direction == 'LEFT':
            snake_x = snake_list[-1][0] - snake_block
            snake_y = snake_list[-1][1]
        elif direction == 'RIGHT':
            snake_x = snake_list[-1][0] + snake_block
            snake_y = snake_list[-1][1]
        elif direction == 'UP':
            snake_x = snake_list[-1][0]
            snake_y = snake_list[-1][1] - snake_block
        elif direction == 'DOWN':
            snake_x = snake_list[-1][0]
            snake_y = snake_list[-1][1] + snake_block

        # Check if the snake has hit the boundaries
        if snake_x >= window_width or snake_x < 0 or snake_y >= window_height or snake_y < 0:
            game_over = True

        # Add new block to snake
        snake_head = [snake_x, snake_y]
        snake_list.append(snake_head)

        # Remove extra blocks
        if len(snake_list) > snake_length:
            del snake_list[0]

        # Check if the snake has hit itself
        for block in snake_list[:-1]:
            if block == snake_head:
                game_over = True

        # Check if the snake has eaten the food
        if snake_x == food_x and snake_y == food_y:
            food_x = round(random.randrange(0, window_width - snake_block) / snake_block) * snake_block
            food_y = round(random.randrange(0, window_height - snake_block) / snake_block) * snake_block
            snake_length += 1
            score += 1
            # Increase speed, but cap it at max_speed
            current_speed = min(current_speed + speed_increase, max_speed)

        # Clear the window
        window.fill(BLACK)

        # Draw the food
        pygame.draw.rect(window, RED, [food_x, food_y, snake_block, snake_block])

        # Draw the snake
        draw_snake(snake_list)

        # Show the score and speed
        show_score_and_speed()

    # Display "Game Over" text if game over
    if game_over:
        window.blit(game_over_text, 
                    (window_width // 2 - game_over_text.get_width() // 2,
                     window_height // 2 - game_over_text.get_height() // 2))
        pygame.draw.rect(window, (0, 0, 255), restart_button_rect)
        window.blit(restart_button_text, restart_button_rect)

    # Update the display
    pygame.display.update()

    # Control game speed
    clock.tick(current_speed)

cap.release()
cv2.destroyAllWindows()
pygame.quit()