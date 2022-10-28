import pygame
from OpenGL.GL import *	

pygame.init()

pix = 5
tamaño = 100 * pix

screen = pygame.display.set_mode(
    (tamaño, tamaño),
    pygame.OPENGL | pygame.DOUBLEBUF
)

tamaño = int(tamaño/pix)

def pixel(x, y, color):
    glEnable(GL_SCISSOR_TEST)
    glScissor(x, y, 10, 10)
    glClearColor(color[0], color[1], color[2], 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_SCISSOR_TEST)

def pixeles(matriz, color):
    glEnable(GL_SCISSOR_TEST)
    glClearColor(color[0], color[1], color[2], 1.0)
    for i in range(tamaño):
        for j in range(tamaño):
            if matriz[i][j] == 1:
                glScissor(i*pix, j*pix, pix, pix)
                glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_SCISSOR_TEST)

x = 0
speed = 1

mat = []
for i in range(tamaño):
    mat.append([])
    for j in range(tamaño):
        mat[i].append(0)

def verificador():
    nueva = []
    for i in range(tamaño):
        nueva.append([])
        for j in range(tamaño):
            vecinos=0
            #1
            if mat[(i+1)% tamaño][j] == 1:
                vecinos+=1
            #2
            if mat[(i+1)% tamaño][j-1] == 1:
                vecinos+=1
            #3
            if mat[i][j-1] == 1:
                vecinos+=1
            #4
            if mat[i-1][j-1] == 1:
                vecinos+=1
            #5
            if mat[i-1][j] == 1:
                vecinos+=1
            #6
            if mat[i-1][(j+1)%tamaño] == 1:
                vecinos+=1
            #7
            if mat[i][(j+1)% tamaño] == 1:
                vecinos+=1
            #8
            if mat[(i+1)% tamaño][(j+1)% tamaño] == 1:
                vecinos+=1
            
            nueva[i].append(0)

            if vecinos < 2 and mat[i][j] == 1:
                nueva[i][j] = 0
            if (vecinos == 2 or vecinos == 3) and mat[i][j] == 1:
                nueva[i][j] = 1
            if vecinos > 3 and mat[i][j] == 1:
                nueva[i][j] = 0
            if vecinos == 3 and mat[i][j] == 0:
                nueva[i][j] = 1

    return nueva




def bloque(x,y):
    mat[x][y] = 1
    mat[x][y+1] = 1
    mat[x+1][y] = 1
    mat[x+1][y+1] = 1

def colmena(x,y):
    mat[x][y] = 1
    mat[x+1][y] = 1
    mat[x-1][y-1] = 1
    mat[x+2][y-1] = 1
    mat[x][y-2] = 1
    mat[x+1][y-2] = 1

def linea(x,y):
    mat[x][y] = 1
    mat[x+1][y] = 1
    mat[x-1][y] = 1

colmena(10,50)
bloque(50,60)
linea(50,10)

verificador()

running = True
while running:
    #clean
    glClearColor(0.0, 0.0, 0.2, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)

    #paint
    pixeles(mat, (1, 1, 1))

    mat = verificador()


    #flip
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False