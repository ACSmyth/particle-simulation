from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import math
import random
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import colorsys

class Particle:
    def __init__(self, x, y):
        self.mass = 49
        self.vel = random.random() * 5
        self.x = x
        self.y = y
        self.theta = random.random() * 2 * math.pi
        self.r = int(self.mass ** 0.5)

    def collide(self, other):
        # self
        phi = math.atan2(self.y - other.y, self.x - other.x)
        term1 = self.vel * math.cos(self.theta - phi) * (self.mass - other.mass)
        term2 = 2 * other.mass * other.vel * math.cos(other.theta - phi)
        term3 = (term1 + term2) / (self.mass + other.mass)
        term4 = self.vel * math.sin(self.theta - phi) * math.cos(phi + math.pi / 2)
        vp1x = (term3 * math.cos(phi)) + term4

        other_term4 = self.vel * math.sin(self.theta - phi) * math.sin(phi + math.pi / 2)
        vp1y = (term3 * math.sin(phi)) + other_term4

        vp1 = (vp1x ** 2 + vp1y ** 2) ** 0.5
        thetap1 = math.atan2(vp1y, vp1x)
        self.vel = min(vp1, max_vel)
        self.theta = thetap1

        # other
        phi = math.atan2(other.y - self.y, other.x - self.x)
        term1 = other.vel * math.cos(other.theta - phi) * (other.mass - self.mass)
        term2 = 2 * self.mass * self.vel * math.cos(self.theta - phi)
        term3 = (term1 + term2) / (other.mass + self.mass)
        term4 = other.vel * math.sin(other.theta - phi) * math.cos(phi + math.pi / 2)
        vp1x = (term3 * math.cos(phi)) + term4

        other_term4 = other.vel * math.sin(other.theta - phi) * math.sin(phi + math.pi / 2)
        vp1y = (term3 * math.sin(phi)) + other_term4

        vp1 = (vp1x ** 2 + vp1y ** 2) ** 0.5
        thetap1 = math.atan2(vp1y, vp1x)
        other.vel = min(vp1, max_vel)
        other.theta = thetap1

    def move(self):
        new_x = self.x + self.vel * math.cos(self.theta)
        new_y = self.y + self.vel * math.sin(self.theta)
        if new_x < self.r or new_x >= w - self.r:
            self.theta = math.pi - self.theta
            new_x += self.vel * math.cos(self.theta)
        if new_y < self.r or new_y >= h - self.r:
            self.theta = 2*math.pi - self.theta
            new_y += self.vel * math.sin(self.theta)
        self.x = new_x
        self.y = new_y

    def draw(self, screen, pygame, xshift=0, yshift=0):
        hue = lerp(self.vel, 0, max_vel, 0.6, 1)
        col = colorsys.hsv_to_rgb(hue, 1, 1)
        col = [e * 255 for e in col]
        pos = [int(self.x) + xshift, int(self.y) + yshift]
        pygame.draw.circle(screen, col, pos, self.r)

class QuadDivider:
    def __init__(self, x, y, w, h, depth=3):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.depth = depth
        if self.depth > 0:
            sub_w = w/2
            sub_h = h/2
            sq = []
            sq.append(QuadDivider(x, y, sub_w, sub_h, depth-1))
            sq.append(QuadDivider(x + sub_w, y, sub_w, sub_h, depth-1))
            sq.append(QuadDivider(x, y + sub_h, sub_w, sub_h, depth-1))
            sq.append(QuadDivider(x + sub_w, y + sub_h, sub_w, sub_h, depth-1))
            self.sub_quads = sq
            self.is_terminal = False
        else:
            self.is_terminal = True
            self.particles = []

    def process_physics(self):
        to_reset = []
        if not self.is_terminal:
            for quad in self.sub_quads:
                sub_to_reset = quad.process_physics()
                to_reset.extend(sub_to_reset)
        else:
            to_reset = []
            for p in self.particles:
                for p2 in self.particles:
                    if p is p2: continue
                    if dist(p.x, p.y, p2.x, p2.y) < p.r + p2.r:
                        p.collide(p2)

            for p in self.particles:
                p.move()
                # detect if any need to change places
                if not self.in_range(p):
                    to_reset.append(p)
            self.particles = [ele for ele in self.particles if ele not in to_reset]
        return to_reset

    def get_particles(self):
        if self.is_terminal:
            return self.particles
        else:
            ret = []
            for quad in self.sub_quads:
                ret.extend(quad.get_particles())
            return ret

    def in_range(self, p):
        return p.x >= self.x and p.x < self.x + self.w and p.y >= self.y and p.y < self.y + self.h

    def add_particle(self, p):
        if self.is_terminal:
            self.particles.append(p)
        else:
            for quad in self.sub_quads:
                if quad.in_range(p):
                    quad.add_particle(p)
                    return
            raise Exception(str(p.x) + ' ' + str(p.y))

max_vel = 5
total_w = 1600
w, h = 1000, 1000
pygame.init()
pygame.display.set_caption('Simulation')
screen = pygame.display.set_mode([total_w, h])
quad = QuadDivider(0, 0, w, h, depth=5)
num_particles = 500

for q in range(num_particles):
    bfr = 20
    x = random.randint(bfr, w-1-bfr)
    y = random.randint(bfr, h-1-bfr)
    p = Particle(x, y)
    #particles.append(p)
    quad.add_particle(p)

def dist(x1, y1, x2, y2):
    return ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

def process_physics():
    to_reset = quad.process_physics()
    for p in to_reset:
        quad.add_particle(p)

    '''
    for p in q.get_particles():
        p.move()

    for p in particles:
        for p2 in particles:
            if p is p2: continue
            if dist(p.x, p.y, p2.x, p2.y) < p.r + p2.r:
                p.collide(p2)
    '''

def lerp(x, a, b, c, d):
    return ((d - c) / (b - a)) * (x - a) + c

def draw_speed_graph(xs, ys, screen, pygame):
    xshift = 50
    yshift = -50
    bar_w = (total_w - w - 80) / len(xs) - 3
    pygame.draw.line(screen, [0,0,0], [w + xshift - 5, h + yshift], [total_w + xshift - 80, h + yshift], 2)
    pygame.draw.line(screen, [0,0,0], [w + xshift - 5, 2], [w + xshift - 5, h + yshift], 2)

    font = pygame.font.SysFont(None, 24)
    speed_txt = font.render('Speed (pixels / second)', True, [0,0,0])
    screen.blit(speed_txt, (int(w + (total_w - w) / 2 - 90), int(h + yshift + 25)))

    freq_txt = font.render('Frequency', True, [0,0,0])
    freq_txt = pygame.transform.rotate(freq_txt, 90)
    screen.blit(freq_txt, (w + xshift - 30, 30))

    for y in range(10):
        scl = 0.9
        text_y = int(lerp(y, 0, 9, 0, num_particles / 3 * scl))
        ln = len(str(text_y))
        if ln == 1:
            len_shift = 5
        elif ln == 2:
            len_shift = 0
        elif ln == 3:
            len_shift = -5

        actual_y = lerp(y, 0, 9, (h + yshift), h - ((h - (100 + yshift)) * scl))
        img = font.render(str(text_y), True, [0,0,0])
        screen.blit(img, (int(w + bar_w + len_shift), int(actual_y)))


    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        actual_x = lerp(x, 0, max_vel, w + xshift, total_w - bar_w + xshift - 80)
        actual_y = lerp(y, 0, num_particles / 3, h + yshift, 100 + yshift)


        hue = lerp(x, 0, max_vel, 0.6, 1)
        col = colorsys.hsv_to_rgb(hue, 1, 1)
        col = [e * 255 for e in col]

        dimensions = [actual_x, actual_y, bar_w, h - actual_y + yshift]
        dimensions = [int(e) for e in dimensions]
        pygame.draw.rect(screen, col, dimensions)

        if i % 5 == 0:
            img = font.render(str(int(x)), True, [0,0,0])
            screen.blit(img, (int(actual_x + 5), int(h + yshift + 5)))

running = True
fps = 180
spf = 1 / fps
avg = 0
tick_ct = 0

speedss = []

while running:
    start = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()

    screen.fill([255,255,255])
    pygame.draw.rect(screen, (0,0,0), [2, 2, w-4, h-4], 1)

    process_physics()

    xs = [q/5 for q in range(26)]
    speeds = [0 for q in range(26)]
    for p in quad.get_particles():
        p.draw(screen, pygame, xshift=2, yshift=2)
        idx = min(xs, key=lambda e: abs(e-p.vel))
        idx = xs.index(idx)
        speeds[idx] += 1
    speedss.append(deepcopy(speeds))

    # smoothed avg
    smoothing = 200
    avg_speeds = [0 for q in range(len(speeds))]
    n = 0
    for q in range(smoothing):
        if q > len(speedss)-1: break
        for i in range(len(speeds)):
            avg_speeds[i] += speedss[len(speedss)-q-1][i]
        n += 1
    avg_speeds = [e / n for e in avg_speeds]
    draw_speed_graph(xs, avg_speeds, screen, pygame)

    pygame.display.flip()

    end = time.time()
    net = end-start
    if net < spf:
        time.sleep(spf - net)

    tick_ct += 1

    if tick_ct % 5 == 0:
        pygame.image.save(screen, 'images/img' + str(tick_ct//5) + '.png')

pygame.quit()