import pygame
from voiture import Voiture

pygame.init()

screen = pygame.display.set_mode((1280,720))

clock = pygame.time.Clock()

ma_voiture = Voiture()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
    keys = pygame.key.get_pressed()
    events = []
    if keys[pygame.K_UP]:
        events.append(pygame.K_UP)
    if keys[pygame.K_DOWN]:
        events.append(pygame.K_DOWN)
    if keys[pygame.K_RIGHT]:
        events.append(pygame.K_RIGHT)
    if keys[pygame.K_LEFT]:
        events.append(pygame.K_LEFT)
    ma_voiture.update(events)

    print(ma_voiture.x_position, ma_voiture.y_position)

    screen.fill("purple")

    rect = pygame.Rect(ma_voiture.x_position, ma_voiture.y_position, 5, 5)
    rect = pygame.draw.rect(screen, 'black', rect, width=0)

    pygame.display.flip()
    clock.tick(60)