from numpy import blackman
import pygame
from math import * 
from OpenGL.GL import *

WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
TRANSPARENTE = (255, 255, 255)
SKY = (97, 85, 127)
GROUND = (185, 204, 214)
SKY2 = (243, 114, 32)
GROUND2 = (83, 153, 176)

colors = [
  (4, 40, 63),
  (0, 91, 82),
  (219, 242, 38),
  (0, 0, 255),
  (255, 255, 255)
]

walls = {
    "1": pygame.image.load('./pared.png'), 
    "2": pygame.image.load('./muro.png')
}

enemis = [
    {
        "name": 0,
        "x" : 120,
        "y" : 120,
        "1":  pygame.image.load('./cora.png')
    }, 
    {
        "name": 1,
        "x" : 300,
        "y" : 300,
        "1":  pygame.image.load('./cora.png')
    }, 
    {
        "name": 2,
        "x" : 300,
        "y" : 110,
        "1":  pygame.image.load('./cora.png')
    }, 
    {
        "name": 3,
        "x" : 90,
        "y" : 200,
        "1":  pygame.image.load('./cora.png')
    }

]

class Raycaster(object):
    def __init__ (self, screen):
        self.screen = screen
        x, y, self.width, self.height = screen.get_rect()
        self.blocksize = 50
        self.map = []
        self.player = {
            'x': int(self.blocksize + self.blocksize / 2),
            'y': int(self.blocksize + self.blocksize / 2),
            'fov': int(pi/3),
            'a': int(0), 
            'vidas': 0
        }
        self.scale = 10
        self.zbuffer = [99999 for z in range(0, int(self.width))]

    def regresar(self):
        if abs(self.player["a"]) == 0:
            if self.mov == "izq":
                self.player["y"] += 10
            elif self.mov == "der":
                self.player["y"] -= 10
            elif self.mov == "arr":
                self.player["x"] -= 10
            elif self.mov == "abaj":
                self.player["x"] += 10
        if abs(self.player["a"]) == pi/4:
            if self.mov == "izq":
                self.player["x"] -= 10
                self.player["y"] += 10
            elif self.mov == "der":
                self.player["x"] += 10
                self.player["y"] -=10
            elif self.mov == "arr":
                self.player["x"] -= 10
                self.player["y"] -= 10
            elif self.mov == "abaj":
                self.player["x"] += 10
                self.player["y"] += 10
        if abs(self.player["a"]) == pi/2:
            if self.mov == "izq":
                self.player["x"] -= 10
            elif self.mov == "der":
                self.player["x"] += 10
            elif self.mov == "arr":
                self.player["y"] -= 10
            elif self.mov == "abaj":
                self.player["y"] += 10
        if abs(self.player["a"]) == 3*pi/4:
            if self.mov == "izq":
                self.player["x"] -= 10
                self.player["y"] -= 10
            elif self.mov == "der":
                self.player["x"] += 10
                self.player["y"] += 10
            elif self.mov == "arr":
                self.player["x"] += 10
                self.player["y"] -= 10
            elif self.mov == "abaj":
                self.player["x"] -= 10
                self.player["y"] += 10
        if abs(self.player["a"]) == pi:
            if self.mov == "izq":
                self.player["y"] -= 10
            elif self.mov == "der":
                self.player["y"] += 10
            elif self.mov == "arr":
                self.player["x"] += 10
            elif self.mov == "abaj":
                self.player["x"] -= 10
        if abs(self.player["a"]) == 5*pi/4:
            if self.mov == "izq":
                self.player["x"] += 10
                self.player["y"] -= 10
            elif self.mov == "der":
                self.player["x"] -= 10
                self.player["y"] += 10
            elif self.mov == "arr":
                self.player["x"] += 10
                self.player["y"] += 10
            elif self.mov == "abaj":
                self.player["x"] -= 10
                self.player["y"] -= 10
        if abs(self.player["a"]) == 3*pi/2:
            if self.mov == "izq":
                self.player["x"] += 10
            elif self.mov == "der":
                self.player["x"] -= 10
            elif self.mov == "arr":
                self.player["y"] += 10
            elif self.mov == "abaj":
                self.player["y"] -= 10
        if abs(self.player["a"]) == 7*pi/4:
            if self.mov == "izq":
                self.player["x"] += 10
                self.player["y"] += 10
            elif self.mov == "der":
                self.player["x"] -= 10
                self.player["y"] -= 10
            elif self.mov == "arr":
                self.player["x"] -= 10
                self.player["y"] += 10
            elif self.mov == "abaj":
                self.player["x"] += 10
                self.player["y"] -= 10

    def clearZ(self):
        self.zbuffer = [99999 for z in range(0, self.width)]

    def point(self, x, y, c = WHITE):
        self.screen.set_at((x, y), c)

    def block(self, x, y, wall):
        for i in range(x, x + self.blocksize):
            for j in range(y, y + self.blocksize):
                tx = int((i - x) * 128 / self.blocksize)
                ty = int((j - y) * 128 / self.blocksize)
                c = wall.get_at((tx, ty))
                self.point(i, j, c)

    def load_map(self, filename):
        with open(filename) as f:
            for line in f.readlines():
                self.map.append(list(line))
    
    def draw_map(self):
        for x in range(0, 500, self.blocksize):
            for y in range(0, 500, self.blocksize):
                i = int(x/self.blocksize)
                j = int(y/self.blocksize)
                if self.map[j][i] != ' ':
                    self.block(x, y, walls[self.map[j][i]])

    def draw_player(self):
        self.point(self.player["x"], self.player["y"])

    def render(self):
        #self.draw_map()
        #self.draw_player()
        
        density = 100

        '''
        #mini mapa
        for i in range(0, density):
            a = self.player["a"] - self.player["fov"]/2 + self.player["fov"]*i/density
            d, c, tx = self.cast_ray(a)

        #separador
        for i in range (0, 500):
            self.point(499, i)
            self.point(500, i)
            self.point(501, i)

        '''


        #3d
        for i in range(0, int(self.width)):
            a = self.player["a"] - self.player["fov"]/2 + self.player["fov"]*i/(self.width)
            d, c, tx = self.cast_ray(a)
            x = i
            try:
                h = self.height/(d * cos(a - self.player['a'])) * self.height/self.scale

                if self.zbuffer[i] >= d:
                    self.draw_stake(x, h, c, tx)
                    self.zbuffer[i] = d
            except:
                self.regresar()


        '''
        for enemy in enemis:
            self.point(enemy["x"], enemy["y"], (255, 0, 0))
        '''

        for enemy in enemis:
            if enemy:
                self.draw_sprite(enemy)

    def draw_sprite(self, sprite):
        sprite_a = atan2(
            sprite["y"] - self.player["y"], 
            sprite["x"] - self.player["x"]
        )

        d = (
            (self.player["x"] - sprite["x"])**2 + 
            (self.player["y"] - sprite["y"])**2
            )** 0.5

        sprite_size = int(((self.width)/d) * self.height/self.scale/6)

        sprite_x = int(
            
            (sprite_a - self.player["a"]) * 
            (self.width) / self.player["fov"] 
            + sprite_size)

        sprite_y = int(self.height/2 - sprite_size/2)
        
        for x in range(sprite_x, sprite_x + sprite_size):
            for y in range(sprite_y, sprite_y + sprite_size):
                tx = int((x - sprite_x) * 128 / sprite_size)
                ty = int((y - sprite_y) * 128 / sprite_size)
                
                c = sprite["1"].get_at((tx, ty))

                if c != TRANSPARENTE:
                    if(x > 0 and x < int(self.width)):
                        if self.zbuffer[x] >= d:
                            self.point(x, y, c)
                            self.zbuffer[x] = d

    def corazon(self, sprite):
        sprite_a = atan2(
            sprite["y"] - self.player["y"], 
            sprite["x"] - self.player["x"]
        )

        d = (
            (self.player["x"] - sprite["x"])**2 + 
            (self.player["y"] - sprite["y"])**2
            )** 0.5

        sprite_size = int(((self.width)/d) * self.height/self.scale/6)

        sprite_x = int(
            
            (sprite_a - self.player["a"]) * 
            (self.width) / self.player["fov"] 
            + sprite_size)

        sprite_y = int(self.height/2 - sprite_size/2)

        for x in range(300,301):
            for y in range(300,301):
                tx = int((x - sprite_x) * 128 / sprite_size)
                ty = int((y - sprite_y) * 128 / sprite_size)

                try:
                    c = sprite["1"].get_at((tx, ty))
                except:
                    c=0

                if  c != WHITE and c !=0 and self.zbuffer[250] >= d:
                    c = 0
                    numero = sprite["name"]
                    enemis.pop(numero)
                    enemis.insert(numero, None)
                    self.player["vidas"] += 1
                    pygame.mixer.Sound.play(pygame.mixer.Sound('./obtener.wav'))
                    break

    def cast_ray(self, a):
        d = 0
        ox = self.player["x"]
        oy = self.player["y"]

        while True:
            x = int(ox + d*cos(a))
            y = int(oy + d*sin(a))

            i = int(x/self.blocksize)
            j = int(y/self.blocksize)


            if self.map[j][i] != ' ':
                hitx = x -  i * self.blocksize
                hity = y -  j * self.blocksize

                if 1 < hitx < self.blocksize-1:
                    maxhit = hitx
                else:
                    maxhit = hity 

                tx = int(maxhit * 128 / self.blocksize)
                return d, self.map[j][i], tx


            #self.point(x, y)
            d += 1

    def draw_stake(self, x, h, c, tx):
        start_y = int(self.height/2 - h/2)
        end_y = int(self.height/2 + h/2)
        heigth= end_y - start_y

        for y in range(start_y, end_y):
            ty = int((y - start_y) *128 / heigth)
            color = walls[c].get_at((tx, ty))
            self.point(x, y, color)
        #Mira
        self.point(300, 300, RED)
        self.point(301, 300, RED)
        self.point(300, 301, RED)
        self.point(299, 300, RED)
        self.point(300, 299, RED)
        self.point(302, 300, RED)
        self.point(300, 302, RED)
        self.point(300, 298, RED)
        self.point(298, 300, RED)




pygame.init()
screen = pygame.display.set_mode((600, 600))
r = Raycaster(screen)


pygame.mixer.init()
pygame.mixer.music.load('./inicio.wav')
pygame.mixer.music.set_volume(0.1)
pygame.mixer.music.play(-1)



ini = pygame.image.load('./Inicio.png')
fini = pygame.image.load('./fin.png')


inicio = True
while inicio:
    
    for x in range(0, 600):
        for y in range(0, 600):
            color = ini.get_at((x, y))
            r.point(x, y, color)

            
    pygame.display.flip()
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            inicio = False

        if (event.type == pygame.KEYDOWN):
            if (event.key == pygame.K_1):
                level = 1
            if (event.key == pygame.K_2):
                level = 2
            if event.key == pygame.K_KP_ENTER:
                try: 
                    level 
                    inicio = False
                except:
                    pass

if level == 1:
    r.load_map("./map.txt")
elif level == 2:
    r.load_map("./map2.txt")

running = True
while running:
    r.clearZ()
    screen.fill(BLACK)
    if level == 1:
        screen.fill(SKY, (0, 0, r.width, r.height/2))
        screen.fill(GROUND, (0, r.height/2, r.width, r.height/2))
    else:
        screen.fill(SKY2, (0, 0, r.width, r.height/2))
        screen.fill(GROUND2, (0, r.height/2, r.width, r.height/2))

    r.render()

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
        if event.type == pygame.MOUSEMOTION:
                r.player["a"] += event.rel[0] * 360 / (r.width/4)
        

        if (event.type == pygame.KEYDOWN):
            if event.key == pygame.K_a:
                r.player["a"] = (r.player["a"] - pi / 4)% (2 * pi)
            if event.key == pygame.K_d:
                r.player["a"] = (r.player["a"] + pi / 4)% (2 * pi)


            if event.key == pygame.K_RIGHT:
                r.mov = "der"
                if r.player["a"] == 0:
                    r.player["y"] += 10
                if abs(r.player["a"]) == pi/4:
                    r.player["x"] -= 10
                    r.player["y"] += 10
                if abs(r.player["a"]) == pi/2:
                    r.player["x"] -= 10
                if abs(r.player["a"]) == 3*pi/4:
                    r.player["x"] -= 10
                    r.player["y"] -= 10
                if abs(r.player["a"]) == pi:
                    r.player["y"] -= 10
                if abs(r.player["a"]) == 5*pi/4:
                    r.player["x"] += 10
                    r.player["y"] -= 10
                if abs(r.player["a"]) == 3*pi/2:
                    r.player["x"] += 10
                if abs(r.player["a"]) == 7*pi/4:
                    r.player["x"] += 10
                    r.player["y"] += 10
            if event.key == pygame.K_LEFT:
                r.mov = "izq"
                if r.player["a"] == 0:
                    r.player["y"] -= 10
                if abs(r.player["a"]) == pi/4:
                    r.player["x"] += 10
                    r.player["y"] -= 10
                if abs(r.player["a"]) == pi/2:
                    r.player["x"] += 10
                if abs(r.player["a"]) == 3*pi/4:
                    r.player["x"] += 10
                    r.player["y"] += 10
                if abs(r.player["a"]) == pi:
                    r.player["y"] += 10
                if abs(r.player["a"]) == 5*pi/4:
                    r.player["x"] -= 10
                    r.player["y"] += 10
                if abs(r.player["a"]) == 3*pi/2:
                    r.player["x"] -= 10
                if abs(r.player["a"]) == 7*pi/4:
                    r.player["x"] -= 10
                    r.player["y"] -= 10
            if event.key == pygame.K_UP:
                r.mov = "arr"
                if r.player["a"] == 0:
                    r.player["x"] += 10
                if abs(r.player["a"]) == pi/4:
                    r.player["x"] += 10
                    r.player["y"] += 10
                if abs(r.player["a"]) == pi/2:
                    r.player["y"] += 10
                if abs(r.player["a"]) == 3*pi/4:
                    r.player["x"] -= 10
                    r.player["y"] += 10
                if abs(r.player["a"]) == pi:
                    r.player["x"] -= 10
                if abs(r.player["a"]) == 5*pi/4:
                    r.player["x"] -= 10
                    r.player["y"] -= 10
                if abs(r.player["a"]) == 3*pi/2:
                    r.player["y"] -= 10
                if abs(r.player["a"]) == 7*pi/4:
                    r.player["x"] += 10
                    r.player["y"] -= 10
            if event.key == pygame.K_DOWN:
                r.mov = "abaj"
                if r.player["a"] == 0:
                    r.player["x"] -= 10
                if abs(r.player["a"]) == pi/4:
                    r.player["x"] -= 10
                    r.player["y"] -= 10
                if abs(r.player["a"]) == pi/2:
                    r.player["y"] -= 10
                if abs(r.player["a"]) == 3*pi/4:
                    r.player["x"] += 10
                    r.player["y"] -= 10
                if abs(r.player["a"]) == pi:
                    r.player["x"] += 10
                if abs(r.player["a"]) == 5*pi/4:
                    r.player["x"] += 10
                    r.player["y"] += 10
                if abs(r.player["a"]) == 3*pi/2:
                    r.player["y"] += 10
                if abs(r.player["a"]) == 7*pi/4:
                    r.player["x"] -= 10
                    r.player["y"] += 10
            if event.key == pygame.K_KP_ENTER:
                pygame.mixer.Sound.play(pygame.mixer.Sound('./disparo.wav'))
                for i in enemis:
                    if i != None:
                        r.corazon(i)

            if r.player["vidas"] == 4:
                running = False
                fin = True


while fin:
    
    for x in range(0, 600):
        for y in range(0, 600):
            color = fini.get_at((x, y))
            r.point(x, y, color)

            
    pygame.display.flip()
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            fin = False

        if (event.type == pygame.KEYDOWN):
            if event.key == pygame.K_KP_ENTER:
                fin = False