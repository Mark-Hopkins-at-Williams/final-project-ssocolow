import pygame
import sys
import numpy as np
from torchvision import transforms
from autotools import *

# Constants
WINDOW_SIZE = (500, 500)
BUTTON_WIDTH = 50
BUTTON_HEIGHT = 50
NUM_BUTTONS = 10
FPS = 60
BG_COLOR = (245, 169, 184)
FONT_SIZE = 24
BUTTON_COLOR = (0, 0, 0)
SELECTED_COLOR = (91, 206, 250)
OUTLINE_COLOR = (0, 100, 0)
STEP = 0.4

# Mock functions to avoid import errors


currentImage = getFirstImage()

def preprocess_image():
    process_image = currentImage.reshape(28, 28)
    process_image = process_image.squeeze()
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(process_image)
    grayscale_transform = transforms.Grayscale(num_output_channels=3)
    image_grayscale = grayscale_transform(image_pil)
    return np.array(image_grayscale)

def display_image():
    image_to_display = preprocess_image()
    scaled_surface = pygame.transform.scale(pygame.surfarray.make_surface(np.flip(np.rot90(image_to_display), axis=0)), (200, 200))
    return scaled_surface

def setup_pygame():
    pygame.init()
    return pygame.display.set_mode(WINDOW_SIZE), pygame.font.Font(None, FONT_SIZE)

def create_buttons():
    buttons = []
    num_buttons_per_row = NUM_BUTTONS // 2
    for i in range(NUM_BUTTONS):
        row = i // num_buttons_per_row
        col = i % num_buttons_per_row
        x = col * (BUTTON_WIDTH + 50) + 25
        y = row * (BUTTON_HEIGHT + 50) + 300
        buttons.append(pygame.Rect(x, y, BUTTON_WIDTH, BUTTON_HEIGHT))
    return buttons

def handle_events(buttons):
    global currentImage
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, button in enumerate(buttons):
                if button.collidepoint(event.pos):
                    print(f"Button {i} clicked!")
                    currentImage = takeStepFrom(currentImage, i, STEP)

def draw_buttons(screen, buttons, surfaces, texts, text_rects):
    for i, button in enumerate(buttons):
        if button.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(surfaces[i], SELECTED_COLOR, (1, 1, BUTTON_WIDTH - 2, BUTTON_HEIGHT - 2))
        else:
            pygame.draw.rect(surfaces[i], BUTTON_COLOR, (0, 0, BUTTON_WIDTH, BUTTON_HEIGHT))
            pygame.draw.rect(surfaces[i], (255, 255, 255), (1, 1, BUTTON_WIDTH - 2, BUTTON_HEIGHT - 2))
            pygame.draw.rect(surfaces[i], OUTLINE_COLOR, (1, BUTTON_HEIGHT - 1, BUTTON_WIDTH - 2, 1), 2)
            pygame.draw.rect(surfaces[i], OUTLINE_COLOR, (1, 1, BUTTON_WIDTH - 2, 1), 2)
        surfaces[i].blit(texts[i], text_rects[i])
        screen.blit(surfaces[i], (button.x, button.y))

def run_game():
    scaled_surface = display_image()
    screen, font = setup_pygame()
    buttons = create_buttons()
    surfaces = [pygame.Surface((BUTTON_WIDTH, BUTTON_HEIGHT)) for _ in range(NUM_BUTTONS)]
    texts = [font.render(str(i), True, BUTTON_COLOR) for i in range(NUM_BUTTONS)]
    text_rects = [text.get_rect(center=(BUTTON_WIDTH // 2, BUTTON_HEIGHT // 2)) for text in texts]
    clock = pygame.time.Clock()

    while True:
        clock.tick(FPS)
        screen.fill(BG_COLOR)
        handle_events(buttons)
        draw_buttons(screen, buttons, surfaces, texts, text_rects)
        scaled_surface = display_image()  # Update the displayed image
        screen.blit(scaled_surface, (150, 20))
        pygame.display.update()

def main():
    run_game()

if __name__ == "__main__":
    main()
