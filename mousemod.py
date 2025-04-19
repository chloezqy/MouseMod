import pygame
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ---------------- Gene Drive Simulation Logic ----------------

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
                + 2 * q1_post * (1 - q1_post) * (s_n + s_c))
               / w1)
    q2_next = ((q2_post**2 * (1 - s)
                + 2 * q2_post * (1 - q2_post) * (s_n + s_c))
               / w2)
    return q1_next, q2_next

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
        pygame.draw.rect(surf, (100,100,100), self.rect)
        frac = (self.value - self.min_val) / (self.max_val - self.min_val)
        knob_x = self.x + frac * self.w
        knob_y = self.y + self.h//2
        pygame.draw.circle(surf, (200,50,50), (int(knob_x), knob_y), self.knob_radius)
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

class Mouse:
    def __init__(self, pos, genotype):
        self.pos = list(pos)
        self.genotype = genotype

    def draw(self, surf):
        cmap = {'aa':(150,150,150), 'Aa':(255,180,50), 'AA':(200,30,30)}
        pygame.draw.circle(surf, cmap[self.genotype], self.pos, 8)

    def move(self, center, radius=100):
        dx, dy = random.choice([-2,-1,0,1,2]), random.choice([-2,-1,0,1,2])
        nx, ny = self.pos[0]+dx, self.pos[1]+dy
        if (nx-center[0])**2 + (ny-center[1])**2 <= (radius-10)**2:
            self.pos = [nx, ny]

# ---------------- Initialization ----------------
pygame.init()
SCREEN_W, SCREEN_H = 1000, 600
ISLAND_AREA_W = 685
PLOT_W = SCREEN_W - ISLAND_AREA_W
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

# Parameters
s, c, h = 0.5, 0.8, 0.3
m = 0.05
alpha = 1.0
q1, q2 = 0.7, 0.1
GENERATION = 0

# Layout
CENTER1 = (ISLAND_AREA_W//3, SCREEN_H//2)
CENTER2 = (2*ISLAND_AREA_W//3, SCREEN_H//2)
ISLAND_R = 100
NUM_MICE = 20
STEP_DELAY_MS = 100
last_update_time = pygame.time.get_ticks()

# History for plot
hist_q1, hist_q2 = [], []

# UI Controls
sliders = [
    Slider(30, 480, "s",  0.0, 1.0, s),
    Slider(240, 480, "c",  0.0, 1.0, c),
    Slider(450, 480, "h",  0.0, 1.0, h),
    Slider(30, 530, "m",  0.0, 0.5, m),
    Slider(240, 530, "q1", 0.0, 1.0, q1),
    Slider(450, 530, "q2", 0.0, 1.0, q2),
]
font = pygame.font.SysFont(None, 26)
run_button = pygame.Rect(290, 560, 100, 30)
reset_button = pygame.Rect(420, 560, 100, 30)
auto_run = False

# Initialize mice populations

def reinit_mice():
    def init(q, center):
        counts = np.random.multinomial(NUM_MICE, [q**2, 2*q*(1-q), (1-q)**2])
        gens = ['AA']*counts[0] + ['Aa']*counts[1] + ['aa']*counts[2]
        random.shuffle(gens)
        arr = []
        for gt in gens:
            ang = random.uniform(0,2*np.pi)
            rad = random.uniform(0, ISLAND_R-15)
            x = int(center[0] + rad*np.cos(ang))
            y = int(center[1] + rad*np.sin(ang))
            arr.append(Mouse((x,y), gt))
        return arr
    return init(q1, CENTER1), init(q2, CENTER2)

mice1, mice2 = reinit_mice()

# Plot setup
fig = plt.figure(figsize=(PLOT_W/100, PLOT_W/100), dpi=100)
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

# Update-genotype helper
def update_genotypes(mice, q, center):
    counts = np.random.multinomial(NUM_MICE, [q**2, 2*q*(1-q), (1-q)**2])
    gen_list = ['AA']*counts[0] + ['Aa']*counts[1] + ['aa']*counts[2]
    random.shuffle(gen_list)
    for mouse, gt in zip(mice, gen_list):
        mouse.genotype = gt
        mouse.move(center, ISLAND_R)

# ---------------- Main Loop ----------------
while True:
    now = pygame.time.get_ticks()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if run_button.collidepoint(event.pos):
                auto_run = not auto_run
            elif reset_button.collidepoint(event.pos):
                s = sliders[0].value
                c = sliders[1].value
                h = sliders[2].value
                m = sliders[3].value
                q1 = sliders[4].value
                q2 = sliders[5].value
                GENERATION = 0
                hist_q1.clear()
                hist_q2.clear()
                mice1, mice2 = reinit_mice()
                auto_run = True
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
            update_genotypes(mice1, q1, CENTER1)
            update_genotypes(mice2, q2, CENTER2)
            last_update_time = now

    # Draw
    screen.fill((245,245,220))
    pygame.draw.circle(screen, (130,200,130), CENTER1, ISLAND_R)
    pygame.draw.circle(screen, (130,200,130), CENTER2, ISLAND_R)
    for m_obj in mice1 + mice2:
        m_obj.draw(screen)
    pygame.draw.rect(screen, (70,130,180), run_button)
    pygame.draw.rect(screen, (70,130,180), reset_button)
    screen.blit(font.render("Run" if not auto_run else "Pause", True, (255,255,255)), (run_button.x+10, run_button.y+5))
    screen.blit(font.render("Reset", True, (255,255,255)), (reset_button.x+20, reset_button.y+5))
    for sl in sliders: sl.draw(screen, font)
    screen.blit(font.render(f"Gen: {GENERATION}   q1={q1:.3f}   q2={q2:.3f}", True, (20,20,20)), (20,20))

    # Plot
    ax.set_xlim(0, max(100, GENERATION))
    line1.set_data(range(len(hist_q1)), hist_q1)
    line2.set_data(range(len(hist_q2)), hist_q2)
    canvas.draw()
    raw = canvas.get_renderer().tostring_rgb()
    surf = pygame.image.fromstring(raw, canvas.get_width_height(), "RGB")
    screen.blit(surf, (ISLAND_AREA_W, (SCREEN_H - PLOT_W)//2))

    pygame.display.flip()
    clock.tick(60)
