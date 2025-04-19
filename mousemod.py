import pygame
import sys
import random
import numpy as np

# ---------------- Gene Drive Simulation Logic ----------------

def step_gene_drive(q1, q2, s, c, h, m, alpha):
    """
    Advance one generation of the two-deme gene drive model with asymmetric migration.
    Returns updated allele frequencies (q1_next, q2_next).
    """
    # Migration
    q1_post = ((1 - alpha*m) * q1 + m * q2) / (1 - alpha*m + m)
    q2_post = ((1 - m) * q2 + alpha*m * q1) / (1 - m + alpha*m)

    # Selection components
    s_n = 0.5 * (1 - c) * (1 - h * s)
    s_c = c * (1 - s)

    # Mean fitness
    w1 = (q1_post**2 * (1 - s)
          + 2 * q1_post * (1 - q1_post) * (2*s_n + s_c)
          + (1 - q1_post)**2)
    w2 = (q2_post**2 * (1 - s)
          + 2 * q2_post * (1 - q2_post) * (2*s_n + s_c)
          + (1 - q2_post)**2)

    # Next-generation frequencies
    q1_next = ((q1_post**2 * (1 - s)
                + 2 * q1_post * (1 - q1_post) * (s_n + s_c))
               / w1)
    q2_next = ((q2_post**2 * (1 - s)
                + 2 * q2_post * (1 - q2_post) * (s_n + s_c))
               / w2)

    return q1_next, q2_next

# ---------------- Pygame Visualization ----------------
pygame.init()

# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Gene Drive Simulator")
clock = pygame.time.Clock()

# Colors
BG_COLOR = (245, 245, 220)
ISLAND_COLOR = (130, 200, 130)
TEXT_COLOR = (20, 20, 20)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (90, 150, 200)
BUTTON_TEXT = (255, 255, 255)

# Simulation parameters (tweak as desired)
s = 0.5       # selection cost
c = 0.8       # conversion rate
h = 0.3       # dominance
m = 0.05      # migration (deme2 -> deme1)
alpha = 1.0   # migration ratio (deme1->deme2 = alpha * m)

# Time control: milliseconds per generation step
STEP_DELAY_MS = 500  # 0.5 seconds between updates
last_update_time = pygame.time.get_ticks()

# Initial allele frequencies
q1, q2 = 0.7, 0.1
GENERATION = 0

# Island positions
ISLAND_RADIUS = 140
CENTER1 = (SCREEN_WIDTH//4, SCREEN_HEIGHT//2)
CENTER2 = (3*SCREEN_WIDTH//4, SCREEN_HEIGHT//2)
NUM_MICE = 20

# Button settings
button_rect = pygame.Rect(SCREEN_WIDTH//2 - 75, SCREEN_HEIGHT - 80, 150, 50)

# Font
defont = pygame.font.SysFont(None, 30)

class Mouse:
    def __init__(self, pos, genotype):
        self.pos = list(pos)
        self.genotype = genotype  # 'aa', 'Aa', 'AA'

    def draw(self, surface):
        color_map = {
            'aa': (150, 150, 150),  # gray
            'Aa': (255, 180, 50),   # orange
            'AA': (200,  30,  30),  # red
        }
        pygame.draw.circle(surface, color_map[self.genotype], self.pos, 8)

    def move(self, center):
        # small random step of 1-2 pixels
        dx = random.choice([-2, -1, 0, 1, 2])
        dy = random.choice([-2, -1, 0, 1, 2])
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        # ensure within island boundary
        if (new_x - center[0])**2 + (new_y - center[1])**2 <= (ISLAND_RADIUS - 10)**2:
            self.pos[0], self.pos[1] = new_x, new_y

# Slider class
class Slider:
    def __init__(self, x, y, label, min_val, max_val, init_val):
        self.x = x
        self.y = y
        self.w = 150
        self.h = 6
        self.knob_radius = 10
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = init_val
        self.dragging = False
        self.slider_rect = pygame.Rect(x, y, self.w, self.h)

    def draw(self, surface):
        # Draw line
        pygame.draw.rect(surface, (100, 100, 100), self.slider_rect)
        # Position of knob
        knob_x = self.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.w
        pygame.draw.circle(surface, (200, 50, 50), (int(knob_x), self.y + self.h // 2), self.knob_radius)
        # Label
        label_surf = defont.render(f"{self.label}: {self.value:.2f}", True, (0, 0, 0))
        surface.blit(label_surf, (self.x, self.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            knob_x = self.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.w
            if abs(event.pos[0] - knob_x) < self.knob_radius + 5 and abs(event.pos[1] - (self.y + self.h // 2)) < self.knob_radius + 5:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = max(self.x, min(event.pos[0], self.x + self.w))
            percent = (rel_x - self.x) / self.w
            self.value = self.min_val + percent * (self.max_val - self.min_val)

# Create sliders
sliders = [
    Slider(50, 480, "s", 0.0, 1.0, s),
    Slider(50, 530, "c", 0.0, 1.0, c),
    Slider(250, 480, "h", 0.0, 1.0, h),
    Slider(250, 530, "q1", 0.0, 1.0, q1),
    Slider(450, 480, "q2", 0.0, 1.0, q2),
]

# Reset button
reset_button = pygame.Rect(700, 500, 150, 40)


def initialize_mice(q, center):
    """
    Create initial Mouse objects with positions and genotypes based on q.
    """
    mice = []
    # Generate genotype counts
    p_AA = q**2
    p_Aa = 2*q*(1 - q)
    p_aa = (1 - q)**2
    counts = np.random.multinomial(NUM_MICE, [p_AA, p_Aa, p_aa])
    genotypes = ['AA']*counts[0] + ['Aa']*counts[1] + ['aa']*counts[2]
    random.shuffle(genotypes)
    for genotype in genotypes:
        angle = random.uniform(0, 2*np.pi)
        radius = random.uniform(0, ISLAND_RADIUS - 15)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        mice.append(Mouse((x, y), genotype))
    return mice

# Initialize persistent mouse populations
mice1 = initialize_mice(q1, CENTER1)
mice2 = initialize_mice(q2, CENTER2)

# Main loop
auto_run = False
running = True
while running:
    current_time = pygame.time.get_ticks()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                auto_run = not auto_run
        for slider in sliders:
            slider.handle_event(event)
        if reset_button.collidepoint(pygame.mouse.get_pos()) and event.type == pygame.MOUSEBUTTONDOWN:
            s = sliders[0].value
            c = sliders[1].value
            h = sliders[2].value
            q1 = sliders[3].value
            q2 = sliders[4].value
            GENERATION = 0
            mice1 = initialize_mice(q1, CENTER1)
            mice2 = initialize_mice(q2, CENTER2)


    # Update simulation at fixed time intervals
    if auto_run and current_time - last_update_time >= STEP_DELAY_MS:
        # update allele frequencies
        q1, q2 = step_gene_drive(q1, q2, s, c, h, m, alpha)
        GENERATION += 1
        last_update_time = current_time
        # update genotypes based on new q1, q2
        def update_genotypes(mice, q):
            counts = np.random.multinomial(NUM_MICE, [q**2, 2*q*(1-q), (1-q)**2])
            gen_list = ['AA']*counts[0] + ['Aa']*counts[1] + ['aa']*counts[2]
            random.shuffle(gen_list)
            for mouse, g in zip(mice, gen_list):
                mouse.genotype = g
        update_genotypes(mice1, q1)
        update_genotypes(mice2, q2)
        # move mice slowly
        for m_obj in mice1:
            m_obj.move(CENTER1)
        for m_obj in mice2:
            m_obj.move(CENTER2)

    # Draw background
    screen.fill(BG_COLOR)

    # Draw islands
    pygame.draw.circle(screen, ISLAND_COLOR, CENTER1, ISLAND_RADIUS)
    pygame.draw.circle(screen, ISLAND_COLOR, CENTER2, ISLAND_RADIUS)

    # Draw mice
    for mouse in mice1 + mice2:
        mouse.draw(screen)

    # Draw button
    mx, my = pygame.mouse.get_pos()
    color = BUTTON_HOVER if button_rect.collidepoint((mx, my)) else BUTTON_COLOR
    pygame.draw.rect(screen, color, button_rect)
    txt = "Run" if not auto_run else "Pause"
    txt_surf = defont.render(txt, True, BUTTON_TEXT)
    screen.blit(txt_surf, (button_rect.x + 40, button_rect.y + 12))

    # Draw text info
    info = f"Gen: {GENERATION}   q1={q1:.2f}   q2={q2:.2f}"
    info_surf = defont.render(info, True, TEXT_COLOR)
    screen.blit(info_surf, (20, 20))

    # Draw sliders
    for slider in sliders:
        slider.draw(screen)

    # Draw reset button
    pygame.draw.rect(screen, BUTTON_COLOR, reset_button)
    reset_txt = defont.render("Reset", True, BUTTON_TEXT)
    screen.blit(reset_txt, (reset_button.x + 40, reset_button.y + 10))


    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
