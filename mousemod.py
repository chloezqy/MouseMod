"""
MouseMod Gene Drive Simulator: Interactive Game-based Visualization

This tool simulates the dynamics of gene drive alleles across two demes (populations)
using recurrence equations based on Greenbaum et al.'s model. Users can manipulate
key parameters such as selection, conversion, dominance, and migration via sliders,
and observe the allele frequencies and population changes through animated mice
and real-time plotting.

Created with:
- Pygame for animation and UI
- Matplotlib for plotting allele frequency dynamics
- Procreate for drawings by Chloe Zhu and Monica Wan
"""

import pygame
import math
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ---------------- Gene Drive Simulation Logic ----------------
# This function updates allele frequencies in both demes for one generation
def step_gene_drive(q1, q2, s, c, h, m, alpha):
    q1_post = ((1 - alpha*m) * q1 + m * q2) / (1 - alpha*m + m)
    q2_post = ((1 - m) * q2 + alpha*m * q1) / (1 - m + alpha*m)
    s_n = 0.5 * (1 - c) * (1 - h * s)
    s_c = c * (1 - s)
    w1 = (q1_post**2 * (1 - s)
          + 2 * q1_post * (1 - q1_post) * (2*s_n + s_c)
          + (1 - q1_post)**2)
    w2 = (q2_post**2 * (1 - s)
          + 2 * q2_post * (1 - q2_post) * (2*s_n + s_c)
          + (1 - q2_post)**2)
    q1_next = ((q1_post**2 * (1 - s)
                + 2 * q1_post * (1 - q1_post) * (s_n + s_c)) / w1)
    q2_next = ((q2_post**2 * (1 - s)
                + 2 * q2_post * (1 - q2_post) * (s_n + s_c)) / w2)
    return q1_next, q2_next

# UI element for controlling simulation parameters interactively
class Slider:
    def __init__(self, x, y, label, min_val, max_val, init_val):
        self.x, self.y = x, y
        self.w, self.h = 150, 6
        self.knob_radius = 10
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = init_val
        self.dragging = False
        self.rect = pygame.Rect(x, y, self.w, self.h)

    def draw(self, surf, font):
        pygame.draw.rect(surf, (100,100,100), self.rect, border_radius=10)
        frac = (self.value - self.min_val) / (self.max_val - self.min_val)
        knob_x = self.x + frac * self.w
        knob_y = self.y + self.h//2
        pygame.draw.circle(surf, (255, 165, 0), (int(knob_x), knob_y), self.knob_radius)
        txt = font.render(f"{self.label}: {self.value:.2f}", True, (20,20,20))
        surf.blit(txt, (self.x, self.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            frac = (self.value - self.min_val) / (self.max_val - self.min_val)
            knob_x = self.x + frac * self.w
            knob_y = self.y + self.h//2
            if (event.pos[0]-knob_x)**2 + (event.pos[1]-knob_y)**2 <= (self.knob_radius+5)**2:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx = max(self.x, min(event.pos[0], self.x + self.w))
            frac = (mx - self.x) / self.w
            self.value = self.min_val + frac * (self.max_val - self.min_val)

# Represents a mouse individual with genotype and visual behavior
class Mouse:
    def __init__(self, pos, genotype, group):
        self.pos = list(pos)
        self.genotype = genotype
        self.group = group
        self.migrating = False
        self.dest = None

    def start_migration(self, dest, new_group):
        self.dest = dest
        self.group = new_group
        # self.migrating = True

    def draw(self, surf):
        # print("drawing")
        img = {(1,'aa'): g1wt_img, (1,'Aa'): g1ht_img, (1,'AA'): g1mt_img,
               (2,'aa'): g2wt_img, (2,'Aa'): g2ht_img, (2,'AA'): g2mt_img}[
               (self.group, self.genotype)]
        rect = img.get_rect(center=self.pos)
        surf.blit(img, rect)

    def move(self, center, radius=100):
        if self.migrating:
            dx = self.dest[0] - self.pos[0]
            dy = self.dest[1] - self.pos[1]
            dist = math.hypot(dx, dy)
            speed = 4
            if dist < speed:
                self.pos = list(self.dest)
                self.migrating = False
            else:
                self.pos[0] += dx/dist * speed
                self.pos[1] += dy/dist * speed
            return
        # normal random walk
        dx, dy = random.choice([-2,-1,0,1,2]), random.choice([-2,-1,0,1,2])
        nx, ny = self.pos[0]+dx, self.pos[1]+dy
        if (nx-center[0])**2 + (ny-center[1])**2 <= (radius-10)**2:
            self.pos = [nx, ny]

# ---------------- Initialization ----------------
# Initialize Pygame and create screen, load background
pygame.init()
SCREEN_W, SCREEN_H = 1000, 600
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

# Load images
# Background image
background_img = pygame.image.load("images/background.png")
background_img = pygame.transform.scale(background_img, (SCREEN_W, SCREEN_H))
# group1 images
g1wt_img = pygame.image.load("images/Rats/group1_wt_resized.png")
g1ht_img = pygame.image.load("images/Rats/group1_het_resized.png")
g1mt_img = pygame.image.load("images/Rats/group1_mutant_resized.png")
# group2 images
g2wt_img = pygame.image.load("images/Rats/group2_wt_resized.png")
g2ht_img = pygame.image.load("images/Rats/group2_het_resized.png")
g2mt_img = pygame.image.load("images/Rats/group2_mutant_resized.png")

# Parameters
s, c, h = 0.5, 0.8, 0.3
m = 0.05
alpha = 1.0
q1, q2 = 0.7, 0.1
GENERATION = 0

CENTER1 = (SCREEN_W//5 - 10, SCREEN_H//2 + 140)
CENTER2 = (2*SCREEN_W//5 + 20, SCREEN_H//2 - 90)
ISLAND_R = 130
NUM_MICE = 20
STEP_DELAY_MS = 100
last_update_time = pygame.time.get_ticks()

hist_q1, hist_q2 = [], []

font = pygame.font.SysFont(None, 26)

# Create interactive sliders for each simulation parameter
sliders = [
    Slider(490, 480, "s",  0.0, 1.0, s),
    Slider(490, 520, "c",  0.0, 1.0, c),
    Slider(490, 560, "h",  0.0, 1.0, h),
    Slider(680, 480, "m",  0.0, 0.5, m),
    Slider(680, 520, "q1", 0.0, 1.0, q1),
    Slider(680, 560, "q2", 0.0, 1.0, q2),
]
run_button = pygame.Rect(860, 470, 100, 40)
reset_button = pygame.Rect(860, 525, 100, 40)
auto_run = False

# Function to randomly generate mice population based on current allele frequency
# Initialize mice populations
def reinit_mice():
    def init(q, center, group_id):
        counts = np.random.multinomial(NUM_MICE, [q**2, 2*q*(1-q), (1-q)**2])
        gens = ['AA']*counts[0] + ['Aa']*counts[1] + ['aa']*counts[2]
        random.shuffle(gens)
        arr = []
        for gt in gens:
            ang = random.uniform(0,2*np.pi)
            rad = random.uniform(0, ISLAND_R-15)
            x = int(center[0] + rad*np.cos(ang))
            y = int(center[1] + rad*np.sin(ang))
            arr.append(Mouse((x,y), gt, group_id))
        print("init mice", arr)
        return arr
    return init(q1, CENTER1, 1), init(q2, CENTER2, 2)

mice1, mice2 = reinit_mice()

# ---------------- Plot Setup ----------------
# Initialize matplotlib plot for allele frequency over generations
fig = plt.figure(figsize=(4, 3.8), dpi=85)
ax = fig.add_subplot(111)
line1, = ax.plot([], [], label='Deme 1', color='red')
line2, = ax.plot([], [], label='Deme 2', color='blue')
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_title("Gene Drive Dynamics")
ax.set_xlabel("Generation")
ax.set_ylabel("Allele Frequency")
ax.legend()
canvas = FigureCanvasAgg(fig)

# Update genotypes of mice based on new allele frequencies
# Update genotypes
def update_genotypes(mice, q, center, group_id):
    counts = np.random.multinomial(NUM_MICE, [q**2, 2*q*(1-q), (1-q)**2])
    gens = ['AA']*counts[0] + ['Aa']*counts[1] + ['aa']*counts[2]
    random.shuffle(gens)
    for mouse, gt in zip(mice, gens):
        mouse.genotype = gt
        mouse.move(center, ISLAND_R)

# ---------------- Main Simulation Loop ----------------
# Handles events, updates simulation, and draws visuals
while True:
    now = pygame.time.get_ticks()
    # Read parameter values from sliders each frame
    s = sliders[0].value
    c = sliders[1].value
    h = sliders[2].value
    m = sliders[3].value

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if run_button.collidepoint(event.pos):
                auto_run = not auto_run
            elif reset_button.collidepoint(event.pos):
                # Reset to default values
                s, c, h, m = 0.5, 0.8, 0.3, 0.05
                q1, q2 = 0.7, 0.1
                # Reset sliders visually
                sliders[0].value = s
                sliders[1].value = c
                sliders[2].value = h
                sliders[3].value = m
                sliders[4].value = q1
                sliders[5].value = q2
                # Reset sim state
                GENERATION = 0
                hist_q1.clear()
                hist_q2.clear()
                mice1, mice2 = reinit_mice()
                auto_run = False

        for sl in sliders:
            sl.handle_event(event)

    if auto_run and now - last_update_time >= STEP_DELAY_MS:
        
        q1_next, q2_next = step_gene_drive(q1, q2, s, c, h, m, alpha)
        if abs(q1 - q1_next) < 1e-6 and abs(q2 - q2_next) < 1e-6:
            auto_run = False
        else:
            q1, q2 = q1_next, q2_next
            GENERATION += 1
            hist_q1.append(q1)
            hist_q2.append(q2)
            if len(hist_q1) > 100:
                hist_q1.pop(0)
                hist_q2.pop(0)
            # migration: mark migrants
            n_mig = int(round(m * NUM_MICE))
            if n_mig > 0:
                print("migration")
                migrants1 = random.sample(mice1, n_mig)
                for mouse in migrants1:
                    mice1.remove(mouse)
                    mice2.append(mouse)
                    mouse.start_migration(CENTER2, 2)
                migrants2 = random.sample(mice2, n_mig)
                for mouse in migrants2:
                    mice2.remove(mouse)
                    mice1.append(mouse)
                    mouse.start_migration(CENTER1, 1)
            # update genotypes after migration
            update_genotypes(mice1, q1, CENTER1, 1)
            update_genotypes(mice2, q2, CENTER2, 2)
            last_update_time = now
    screen.blit(background_img, (0, 0))
    # pygame.draw.circle(screen, (130,200,130), CENTER1, ISLAND_R)
    # pygame.draw.circle(screen, (130,200,130), CENTER2, ISLAND_R)

    # 2) then move & draw mice on top
    for m in mice1 + mice2:
        m.move(CENTER1 if m.group==1 else CENTER2)
        m.draw(screen)

    # Draw Run and Reset buttons
    pygame.draw.rect(screen, (135,169,107), run_button, border_radius=10)
    pygame.draw.rect(screen, (135,169,107), reset_button, border_radius=10)
    run_text = font.render("Run" if not auto_run else "Pause", True, (255,255,255))
    run_text_rect = run_text.get_rect(center=run_button.center)
    screen.blit(run_text, run_text_rect)
    reset_text = font.render("Reset", True, (255,255,255))
    reset_text_rect = reset_text.get_rect(center=reset_button.center)
    screen.blit(reset_text, reset_text_rect)

    for sl in sliders:
        sl.draw(screen, font)

    gen_text = font.render(f"Gen: {GENERATION}   q1={q1:.3f}   q2={q2:.3f}", True, (20,20,20))
    screen.blit(gen_text, (SCREEN_W - gen_text.get_width() - 20, 20))

    ax.set_xlim(0, max(100, GENERATION))
    line1.set_data(range(len(hist_q1)), hist_q1)
    line2.set_data(range(len(hist_q2)), hist_q2)
    canvas.draw()
    raw = canvas.get_renderer().tostring_rgb()
    surf = pygame.image.fromstring(raw, canvas.get_width_height(), "RGB")
    
    # Draw the plot
    screen.blit(surf, (SCREEN_W - surf.get_width(), 60))

    pygame.display.flip()
    clock.tick(60)