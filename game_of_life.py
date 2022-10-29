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

def movido(x,y):
    mat[x][y] = 1
    mat[x][y-2] = 1
    mat[x+1][y-2] = 1
    mat[x-1][y-2] = 1
    mat[x+1][y-1] = 1

def medusa(x,y):
    mat[x][y] = 1
    mat[x+1][y] = 1
    mat[x+2][y] = 1
    mat[x+3][y] = 1
    mat[x+4][y] = 1
    mat[x+4][y-1] = 1
    mat[x+4][y-2] = 1
    mat[x+3][y-3] = 1
    mat[x-1][y-1] = 1
    mat[x-1][y-3] = 1
    mat[x+1][y-4] = 1

def casii(x,y):
    mat[x][y]=1
    mat[x][y-1]=1
    mat[x][y-2]=1
    mat[x-1][y-2]=1
    mat[x+1][y-2]=1

    mat[x][y-5]=1
    mat[x+1][y-5]=1
    mat[x-1][y-5]=1
    mat[x][y-6]=1
    mat[x][y-7]=1
    mat[x][y-8]=1
    mat[x][y-9]=1
    mat[x][y-10]=1
    mat[x+1][y-10]=1
    mat[x-1][y-10]=1

    mat[x][y-13]=1
    mat[x-1][y-13]=1
    mat[x+1][y-13]=1
    mat[x][y-14]=1
    mat[x][y-15]=1

def flecha(x,y):
    mat[x][y]=1
    mat[x][y-1]=1
    mat[x-1][y-2]=1
    mat[x-2][y-2]=1
    mat[x-3][y-2]=1
    mat[x-4][y-2]=1
    mat[x+1][y+1]=1
    mat[x+1][y+2]=1
    mat[x+2][y]=1
    mat[x+3][y+1]=1
    mat[x+4][y+1]=1
    mat[x+5][y+1]=1

    mat[x+7][y-1]=1
    mat[x+8][y-1]=1
    mat[x+9][y-1]=1
    mat[x+10][y-1]=1
    mat[x+6][y-2]=1
    mat[x+6][y-3]=1
    mat[x+3][y-3]=1
    mat[x+1][y-4]=1
    mat[x+2][y-4]=1
    mat[x+3][y-4]=1
    mat[x+5][y-4]=1
    mat[x+5][y-5]=1

def corazon(x,y):

    mat[x-2][y-2]=1
    mat[x-2][y-3]=1

    mat[x-1][y-1]=1
    mat[x-1][y-4]=1

    mat[x][y]=1
    mat[x][y-3]=1

    mat[x+1][y]=1
    mat[x+1][y-7]=1

    mat[x+2][y+1]=1
    mat[x+2][y-2]=1
    mat[x+2][y-6]=1
    mat[x+2][y-8]=1

    mat[x+3][y]=1
    mat[x+3][y-1]=1
    mat[x+3][y-2]=1
    mat[x+3][y-6]=1
    mat[x+3][y-7]=1

    mat[x+5][y]=1
    mat[x+5][y-1]=1
    mat[x+5][y-2]=1
    mat[x+5][y-6]=1
    mat[x+5][y-7]=1

    mat[x+6][y+1]=1
    mat[x+6][y-2]=1
    mat[x+6][y-6]=1
    mat[x+6][y-8]=1

    mat[x+7][y]=1
    mat[x+7][y-7]=1

    mat[x+8][y]=1
    mat[x+8][y-2]=1

    mat[x+9][y-1]=1
    mat[x+9][y-4]=1

    mat[x+10][y-2]=1
    mat[x+10][y-3]=1

def generador(x,y):
    mat[x][y]=1
    mat[x][y-1]=1
    mat[x+1][y]=1
    mat[x+1][y-1]=1
    
    mat[x+10][y]=1
    mat[x+10][y-1]=1
    mat[x+10][y-2]=1

    mat[x+11][y+1]=1
    mat[x+11][y-3]=1

    mat[x+12][y+2]=1
    mat[x+12][y-4]=1

    mat[x+13][y+2]=1
    mat[x+13][y-4]=1

    mat[x+14][y-1]=1

    mat[x+15][y+1]=1
    mat[x+15][y-3]=1

    mat[x+16][y]=1
    mat[x+16][y-1]=1
    mat[x+16][y-2]=1

    mat[x+17][y-1]=1

    mat[x+20][y]=1
    mat[x+20][y+1]=1
    mat[x+20][y+2]=1

    mat[x+21][y]=1
    mat[x+21][y+1]=1
    mat[x+21][y+2]=1

    mat[x+22][y+3]=1
    mat[x+22][y-1]=1

    mat[x+24][y-1]=1
    mat[x+24][y-2]=1
    mat[x+24][y+3]=1
    mat[x+24][y+4]=1

    mat[x+34][y+2]=1
    mat[x+34][y+1]=1
    mat[x+35][y+2]=1
    mat[x+35][y+1]=1



generador(10,80)
colmena(10,50)
bloque(50,60)
linea(50,10)
movido(50,80)
medusa(20,20)
casii(70,80)
flecha(20,70)
corazon(60,60)
generador(60,10)

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