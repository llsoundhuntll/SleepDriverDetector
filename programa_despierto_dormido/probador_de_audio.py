import pygame

pygame.init()
pygame.mixer.music.load("wakeup.mp3")
pygame.mixer.music.play()

# Esperar hasta que termine la reproducción
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)