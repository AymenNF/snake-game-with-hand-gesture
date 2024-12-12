import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
from typing import Tuple

class CameraProcessor:
    def __init__(self):
        # Configuration du traitement d'image
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.kernel = np.ones((5,5), np.uint8)

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """Prétraitement de l'image pour améliorer la détection"""
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Amélioration du contraste
        gray = self.clahe.apply(gray)
        
        # Réduction du bruit
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Égalisation d'histogramme
        enhanced = cv2.equalizeHist(denoised)
        
        # Reconversion en couleur
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced_color

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Traitement complet de l'image"""
        # Prétraitement
        processed = self.preprocess_image(frame)
        
        # Création du masque pour la segmentation
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        # Plage de couleur pour la peau
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Amélioration du masque
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Application du masque
        segmented = cv2.bitwise_and(processed, processed, mask=mask)
        
        return processed, segmented

class SnakeGame:
    def __init__(self, width: int = 800, height: int = 600, block_size: int = 20):
        # Initialisation de Pygame
        pygame.init()
        
        # Configuration de la fenêtre
        self.width = width
        self.height = height
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake Game avec Grille")

        # Configuration du serpent
        self.block_size = block_size
        self.grid_rows = height // block_size
        self.grid_columns = width // block_size

        # Couleurs
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.GRID_COLOR = (50, 50, 50)

        # État du jeu
        self.reset_game()

        # Polices
        self.score_font = pygame.font.Font('freesansbold.ttf', 24)
        self.game_over_font = pygame.font.Font('freesansbold.ttf', 64)
        self.game_over_text = self.game_over_font.render('Game Over', True, self.WHITE)
        self.restart_font = pygame.font.Font('freesansbold.ttf', 32)
        self.restart_text = self.restart_font.render('Restart', True, self.WHITE)
        self.restart_rect = self.restart_text.get_rect()
        self.restart_rect.center = (width // 2, height // 2 + 50)

    def reset_game(self):
        """Réinitialise l'état du jeu"""
        self.snake_list = [[self.width // 2, self.height // 2]]
        self.snake_length = 1
        self.direction = 'RIGHT'
        self.score = 0
        self.game_over = False
        self.current_speed = 5
        self.generate_food()

    def generate_food(self):
        """Génère une nouvelle position pour la nourriture"""
        self.food_x = round(random.randrange(0, self.width - self.block_size) / 
                          self.block_size) * self.block_size
        self.food_y = round(random.randrange(0, self.height - self.block_size) / 
                          self.block_size) * self.block_size

    def draw_grid(self):
        """Dessine la grille"""
        for x in range(0, self.width, self.block_size):
            pygame.draw.line(self.window, self.GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, self.block_size):
            pygame.draw.line(self.window, self.GRID_COLOR, (0, y), (self.width, y))

    def draw_snake(self):
        """Dessine le serpent"""
        for block in self.snake_list:
            pygame.draw.rect(self.window, self.GREEN, 
                           [block[0], block[1], self.block_size, self.block_size])
            pygame.draw.rect(self.window, (0, 200, 0), 
                           [block[0], block[1], self.block_size, self.block_size], 1)

    def draw_food(self):
        """Dessine la nourriture"""
        pygame.draw.rect(self.window, self.RED, 
                        [self.food_x, self.food_y, self.block_size, self.block_size])
        pygame.draw.rect(self.window, (255, 150, 150),
                        [self.food_x + 4, self.food_y + 4, 
                         self.block_size - 8, self.block_size - 8])

    def show_info(self):
        """Affiche les informations du jeu"""
        info_texts = [
            f"Score: {self.score}",
            f"Vitesse: {self.current_speed:.1f}",
            "Utilisez votre main",
            f"Grille: {self.grid_rows}x{self.grid_columns}"
        ]
        
        for i, text in enumerate(info_texts):
            info_surface = self.score_font.render(text, True, self.WHITE)
            self.window.blit(info_surface, [10, 10 + (30 * i)])

    def handle_keyboard(self):
        """Gère les entrées clavier"""
        keys = pygame.key.get_pressed()
        
        if not self.game_over:
            if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and self.direction != 'RIGHT':
                self.direction = 'LEFT'
            elif (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and self.direction != 'LEFT':
                self.direction = 'RIGHT'
            elif (keys[pygame.K_UP] or keys[pygame.K_w]) and self.direction != 'DOWN':
                self.direction = 'UP'
            elif (keys[pygame.K_DOWN] or keys[pygame.K_s]) and self.direction != 'UP':
                self.direction = 'DOWN'

    def update(self):
        """Met à jour l'état du jeu"""
        if not self.game_over:
            # Déplacement du serpent
            if self.direction == 'LEFT':
                new_x = self.snake_list[-1][0] - self.block_size
                new_y = self.snake_list[-1][1]
            elif self.direction == 'RIGHT':
                new_x = self.snake_list[-1][0] + self.block_size
                new_y = self.snake_list[-1][1]
            elif self.direction == 'UP':
                new_x = self.snake_list[-1][0]
                new_y = self.snake_list[-1][1] - self.block_size
            elif self.direction == 'DOWN':
                new_x = self.snake_list[-1][0]
                new_y = self.snake_list[-1][1] + self.block_size

            # Vérification des collisions avec les murs
            if (new_x >= self.width or new_x < 0 or 
                new_y >= self.height or new_y < 0):
                self.game_over = True
                return

            # Ajout de la nouvelle position
            self.snake_list.append([new_x, new_y])

            # Suppression de la queue si nécessaire
            if len(self.snake_list) > self.snake_length:
                del self.snake_list[0]

            # Vérification des collisions avec soi-même
            for block in self.snake_list[:-1]:
                if block == [new_x, new_y]:
                    self.game_over = True
                    return

            # Vérification de la collision avec la nourriture
            if new_x == self.food_x and new_y == self.food_y:
                self.generate_food()
                self.snake_length += 1
                self.score += 1
                self.current_speed = min(self.current_speed + 0.5, 15)

    def draw(self):
        """Dessine tous les éléments du jeu"""
        self.window.fill(self.BLACK)
        self.draw_grid()
        
        if not self.game_over:
            self.draw_food()
            self.draw_snake()
            self.show_info()
        else:
            self.window.blit(self.game_over_text, 
                           (self.width // 2 - self.game_over_text.get_width() // 2,
                            self.height // 2 - self.game_over_text.get_height() // 2))
            pygame.draw.rect(self.window, (0, 0, 255), self.restart_rect)
            self.window.blit(self.restart_text, self.restart_rect)

        pygame.display.update()

def main():
    # Initialisation
    game = SnakeGame()
    camera_processor = CameraProcessor()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, 
                          max_num_hands=1, 
                          min_detection_confidence=0.5)
    
    # Capture vidéo
    cap = cv2.VideoCapture(0)
    clock = pygame.time.Clock()
    running = True

    while running:
        # Gestion des événements Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and game.game_over:
                mouse_pos = pygame.mouse.get_pos()
                if game.restart_rect.collidepoint(mouse_pos):
                    game.reset_game()

        # Traitement de la caméra
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        processed_frame, segmented_frame = camera_processor.process_frame(frame)
        
        # Détection des mains
        results = hands.process(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    processed_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), 
                                                         thickness=2, 
                                                         circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), 
                                                         thickness=2)
                )

        # Affichage des flux vidéo
        cv2.imshow('Processed View', processed_frame)
        cv2.imshow('Segmented View', segmented_frame)

        # Mise à jour du jeu
        game.handle_keyboard()
        game.update()
        game.draw()

        # Contrôle de la vitesse
        clock.tick(game.current_speed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()