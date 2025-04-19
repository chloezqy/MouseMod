# 🧬 Gene Drive Simulator 🐁✨  
**An Interactive, Hand-Drawn Exploration of Gene Drive Dynamics**

Welcome to our **Gene Drive Simulator** — a playful yet powerful tool for visualizing how gene drives spread across two populations.

🖌️ Hand-drawn islands.  
🐁 Adorable mice illustrations.  
🎛️ Real-time interactivity.  
📈 Scientifically accurate dynamics.

This project blends biology, game design, and art into an **interactive educational experience**. Based on real population genetics research, it helps you explore what happens when a gene drive is released — and what it takes to keep it contained.

👩‍🎨 Made With Love By **Chloe Zhu** and **Monica Wan** -- designers, coders, and artists behind the vision and implementation.

🙏 Acknowledgements to **Savannah Cheng** and **Camellia Jiang** for their creative input, thoughtful feedback, and support throughout the project.

---

## 🌿 What You Can Do

### 🐁 Interact With Two Populations
- Two island habitats, lovingly illustrated
- Mice show up in three cute styles:
  - **Red** = wild-type (aa)  
  - **Pink** = heterozygote (Aa)  
  - **Blue** = gene drive homozygote (AA)

### 🎛️ Adjust Simulation Parameters
Use sliders to change:
- **Selection** `s` — how costly the drive is to fitness
- **Conversion** `c` — how aggressively the drive converts heterozygotes
- **Dominance** `h` — how visible its effects are
- **Migration** `m` — how often mice cross islands
- **Initial frequencies** `q1`, `q2` — set the scene

### 📈 Watch the Story Unfold
- Mice populations animate and evolve
- A real-time **matplotlib plot** shows how allele frequencies change over generations
- Toggle between **Run**, **Pause**, and **Reset** for complete control

---

## 📚 Scientific Background

This simulation is inspired by the model in:

> Greenbaum, G. et al. (2021). *Genetic engineering for conservation: Bioethical and population-genetic considerations*. PLoS Genetics.

We use the **two-deme recurrence model** described in the paper, incorporating asymmetric migration and allele conversion dynamics.

It demonstrates critical outcomes like:
- **Differential targeting** (gene drive spreads only in one population)  
- **Containment thresholds** (how migration can break barriers)  
- **Stability and convergence** of allele frequencies

---

## 🎨 Hand-Drawn Aesthetic

We believe science education can be beautiful.

- 🖼️ **Custom illustrated islands** — soft, organic shapes instead of rigid grids  
- 🐭 **Cute mouse sprites** — express genotype visually and joyfully  
- 💡 **Friendly UI** — smooth sliders and rounded buttons

Every visual was carefully crafted to keep the experience accessible, playful, and conceptually clear.

---

## 🧰 Tech Stack

- 🐍 **Python**
- 🎮 **Pygame** for animations and controls
- 📊 **Matplotlib** for real-time graphs
- ⚙️ **NumPy** for fast simulations

---

## 🚀 How to Run

```bash
pip install pygame matplotlib numpy
python mousemod.py
