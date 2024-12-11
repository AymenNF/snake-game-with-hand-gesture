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
pygame.display.set_caption("Snake Game - Contrôle Clavier")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Snake settings
snake_block = 20
initial_speed = 5
max_speed = 15
speed_increase = 0.5

# Initialize snake
snake_list = [[window_width // 2, window_height // 2]]
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

def show_score_and_info():
    score_text = score_font.render(f"Score: {score}", True, WHITE)
    speed_text = score_font.render(f"Vitesse: {current_speed:.1f}", True, WHITE)
    controls_text = score_font.render("Utilisez votre doigt", True, WHITE)
    
    window.blit(score_text, [10, 10])
    window.blit(speed_text, [10, 40])
    window.blit(controls_text, [10, 70])

# Game loop
running = True
clock = pygame.time.Clock()
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
                snake_list = [[window_width // 2, window_height // 2]]
                snake_length = 1
                direction = 'RIGHT'
                score = 0
                current_speed = initial_speed
                food_x = round(random.randrange(0, window_width - snake_block) / snake_block) * snake_block
                food_y = round(random.randrange(0, window_height - snake_block) / snake_block) * snake_block

    # Handle keyboard input
    keys = pygame.key.get_pressed()
    if not game_over:
        if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and direction != 'RIGHT':
            direction = 'LEFT'
        elif (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and direction != 'LEFT':
            direction = 'RIGHT'
        elif (keys[pygame.K_UP] or keys[pygame.K_w]) and direction != 'DOWN':
            direction = 'UP'
        elif (keys[pygame.K_DOWN] or keys[pygame.K_s]) and direction != 'UP':
            direction = 'DOWN'

    # Handle webcam and hand detection (display only)
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand position for display
            index_finger_landmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_landmarks.x * frame.shape[1])
            index_finger_y = int(index_finger_landmarks.y * frame.shape[0])

            # Draw hand detection visualization
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Detection', frame)
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

        if not game_over:
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
                current_speed = min(current_speed + speed_increase, max_speed)

    # Drawing
    window.fill(BLACK)
    
    if not game_over:
        pygame.draw.rect(window, RED, [food_x, food_y, snake_block, snake_block])
        draw_snake(snake_list)
        show_score_and_info()
    else:
        window.blit(game_over_text, 
                    (window_width // 2 - game_over_text.get_width() // 2,
                     window_height // 2 - game_over_text.get_height() // 2))
        pygame.draw.rect(window, (0, 0, 255), restart_button_rect)
        window.blit(restart_button_text, restart_button_rect)

    pygame.display.update()
    clock.tick(current_speed)

cap.release()
cv2.destroyAllWindows()
pygame.quit()