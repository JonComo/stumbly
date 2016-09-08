import numpy as np
import pygame

PAD = 12

def normalize(W):
    W = W/np.linalg.norm(W, axis=0)
    return W

def randn(shape):
    return normalize(np.random.randn(shape[0], shape[1]))

def render_matrix(screen, m, p):
    x = p[0]
    y = p[1]

    norm = np.clip(m, -1, 1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = norm[i, j]
            r = int(max(v, 0) * 255)
            b = int(-min(v, 0) * 255)
            pygame.draw.rect(screen, (r, 0, b), (x, y, PAD, PAD), 0)
            x += PAD
        x = p[0]
        y += PAD
        
def render_matrices(screen, ms):
    sx = 10
    sy = 10
    for m in ms:
        render_matrix(screen, m, (sx, sy))
        sx += m.shape[1] * PAD + PAD