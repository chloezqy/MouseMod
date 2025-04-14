import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def run_gene_drive_model(s: float, c: float, h: float, m: float, alpha: float,
                         initial_q1: float, initial_q2: float, 
                         max_generations: int = 10000, 
                         convergence_threshold: float = 1e-10) -> Tuple[float, float, List, List]:
    """
    Simulate the two-deme gene drive model from Greenbaum et al. (2021) with asymmetric migration.
    
    Parameters:
    - s: selection coefficient (fitness cost)
    - c: conversion rate 
    - h: dominance coefficient
    - m: migration rate from deme 2 to deme 1
    - alpha: ratio of migration rates (α); migration from deme 1 to deme 2 is α*m
    - initial_q1, initial_q2: initial frequencies in deme 1 and deme 2
    - max_generations: maximum number of generations to simulate
    - convergence_threshold: threshold for determining convergence
    
    Returns:
    - equilibrium frequencies in both demes and history of frequencies
    """
    q1, q2 = initial_q1, initial_q2
    q1_history = [q1]
    q2_history = [q2]
    
    for generation in range(max_generations):
        # Migration phase with asymmetric rates
        # From the paper: q̃₁ = [(1-αm)q₁ + mq₂] / (1-αm+m)
        # and q̃₂ = [(1-m)q₂ + αmq₁] / (1-m+αm)
        q1_post_migration = ((1 - alpha*m) * q1 + m * q2) / (1 - alpha*m + m)
        q2_post_migration = ((1 - m) * q2 + alpha*m * q1) / (1 - m + alpha*m)
        
        # Calculate selection components
        s_n = 0.5 * (1 - c) * (1 - h * s)  # non-converted heterozygotes
        s_c = c * (1 - s)  # converted heterozygotes (conversion before selection)
        
        # Calculate mean fitness in each deme
        mean_fitness_1 = (q1_post_migration**2 * (1-s) + 
                          2 * q1_post_migration * (1-q1_post_migration) * (2*s_n+s_c) + 
                          (1-q1_post_migration)**2)
        
        mean_fitness_2 = (q2_post_migration**2 * (1-s) + 
                          2 * q2_post_migration * (1-q2_post_migration) * (2*s_n+s_c) + 
                          (1-q2_post_migration)**2)
        
        # Selection phase (calculate frequencies for next generation)
        q1_next = (q1_post_migration**2 * (1-s) + 
                   2 * q1_post_migration * (1-q1_post_migration) * (s_n + s_c)) / mean_fitness_1
        
        q2_next = (q2_post_migration**2 * (1-s) + 
                   2 * q2_post_migration * (1-q2_post_migration) * (s_n + s_c)) / mean_fitness_2
        
        # Check for convergence
        if (abs(q1_next - q1) < convergence_threshold and 
            abs(q2_next - q2) < convergence_threshold):
            q1, q2 = q1_next, q2_next
            q1_history.append(q1)
            q2_history.append(q2)
            break
        
        q1, q2 = q1_next, q2_next
        q1_history.append(q1)
        q2_history.append(q2)
    
    return q1, q2, q1_history, q2_history

def find_critical_migration(s: float, c: float, h: float, alpha: float = 1.0, 
                           initial_q1: float = 0.7, initial_q2: float = 0.1,
                           precision: float = 0.001) -> float:
    """
    Find the critical migration threshold (m*) for a given gene drive configuration.
    This is an approximation - it finds the highest m where differential targeting occurs.
    """
    m_low, m_high = 0.0, 0.5
    
    while m_high - m_low > precision:
        m_mid = (m_low + m_high) / 2
        final_q1, final_q2, _, _ = run_gene_drive_model(s, c, h, m_mid, alpha, initial_q1, initial_q2)
        
        # Check if we have differential targeting (DTE)
        has_dte = (0 < final_q2 < final_q1 < 1)
        
        if has_dte:
            m_low = m_mid  # Try a higher migration rate
        else:
            m_high = m_mid  # Try a lower migration rate
    
    return m_low

def test_parameter_set(s: float, c: float, h: float, m: float, alpha: float,
                       initial_values: List[Tuple[float, float]]) -> None:
    """Test a set of parameters with different initial conditions."""
    print(f"Parameters: s={s}, c={c}, h={h}, m={m}, alpha={alpha}")
    
    results = []
    for i, (init_q1, init_q2) in enumerate(initial_values):
        final_q1, final_q2, _, _ = run_gene_drive_model(s, c, h, m, alpha, init_q1, init_q2)
        results.append((init_q1, init_q2, final_q1, final_q2))
        print(f"Initial: ({init_q1:.2f}, {init_q2:.2f}) → Final: ({final_q1:.6f}, {final_q2:.6f})")
    
    return results

def plot_dynamics(s: float, c: float, h: float, m: float, alpha: float,
                  initial_q1: float, initial_q2: float) -> None:
    """Plot the dynamics of gene drive frequencies over time."""
    _, _, q1_history, q2_history = run_gene_drive_model(s, c, h, m, alpha, initial_q1, initial_q2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(q1_history, label='Deme 1 (Target)')
    plt.plot(q2_history, label='Deme 2 (Non-target)')
    plt.xlabel('Generation')
    plt.ylabel('Gene Drive Allele Frequency')
    plt.title(f'Gene Drive Dynamics (s={s}, c={c}, h={h}, m={m}, α={alpha})')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_alpha_comparison(s: float, c: float, h: float, m: float, 
                         initial_q1: float = 0.7, initial_q2: float = 0.001) -> None:
    """Plot a comparison of different alpha values."""
    alpha_values = [0.1, 0.5, 1.0, 2.0, 10.0]
    
    plt.figure(figsize=(12, 8))
    
    for alpha in alpha_values:
        final_q1, final_q2, q1_history, q2_history = run_gene_drive_model(
            s, c, h, m, alpha, initial_q1, initial_q2)
        plt.plot(q1_history[:100], label=f'α={alpha}, Deme 1', linestyle='-')
        plt.plot(q2_history[:100], label=f'α={alpha}, Deme 2', linestyle='--')
    
    plt.xlabel('Generation')
    plt.ylabel('Gene Drive Allele Frequency')
    plt.title(f'Effect of Asymmetric Migration (s={s}, c={c}, h={h}, m={m})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Example 1: B2 configuration from the paper (Fig. 2)
    s, c, h = 0.5, 0.6, 0.3
    
    # Test differential targeting with symmetric migration (alpha = 1)
    print("Testing B2 configuration with symmetric migration:")
    m = 0.02
    alpha = 1.0
    initial_values = [(0.001, 0.01), (0.5, 0.3), (0.9, 0.1)]
    results_symmetric = test_parameter_set(s, c, h, m, alpha, initial_values)
    
    # Test with asymmetric migration
    print("\nTesting B2 configuration with asymmetric migration (α = 0.5):")
    alpha = 0.5
    results_asymmetric = test_parameter_set(s, c, h, m, alpha, initial_values)
    
    # Plot dynamics for a specific case with symmetric migration
    plot_dynamics(s, c, h, m, 1.0, 0.01, 0.1)
    
    # Plot comparison of different alpha values
    plot_alpha_comparison(s, c, h, m)
    
    # Find critical migration threshold for this configuration
    m_star = find_critical_migration(s, c, h)
    print(f"\nEstimated critical migration threshold (m*) for s={s}, c={c}, h={h}: {m_star:.4f}")
    
    # Example 2: Parameters from malaria vector example (full conversion)
    s_malaria, c_malaria, h_malaria = 0.73, 1.0, 0.5  # h doesn't matter when c=1
    m_malaria = 0.09  # Just below the critical threshold mentioned for mosquitoes
    
    print("\nTesting malaria vector example:")
    initial_values_malaria = [(0.65, 0.1), (0.8, 0.2)]
    results_malaria = test_parameter_set(s_malaria, c_malaria, h_malaria, m_malaria, 1.0, initial_values_malaria)
    
    # Example 3: Parameters from rodent example (lower conversion)
    s_rodent, c_rodent, h_rodent = 0.6, 0.72, 1.0  # Dominant gene drive
    m_rodent = 0.07  # Below the critical threshold mentioned for dominant rodent drives
    
    print("\nTesting rodent example (dominant gene drive):")
    initial_values_rodent = [(0.7, 0.1), (0.8, 0.2)]
    results_rodent = test_parameter_set(s_rodent, c_rodent, h_rodent, m_rodent, 1.0, initial_values_rodent)
    
    # Example 4: Testing asymmetric migration for invasive species management
    # The paper mentions α < 1 (reduced migration from target to non-target) is beneficial
    print("\nTesting asymmetric migration for invasive species management (α = 0.1):")
    results_asymm_invasive = test_parameter_set(s_rodent, c_rodent, h_rodent, 0.08, 0.1, initial_values_rodent)