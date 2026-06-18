import numpy as np
import pandas as pd
from scipy.stats import linregress

class ReefMetricsCalculator:
    """
    Computes Community-Level and Population-Level indicators for coral reef health
    based on stereovision 3D size and species classification.
    """
    def __init__(self):
        # Ecological Length-Weight constants: W = a * L^b
        # (Replace these placeholders with actual values from FishBase for your 5 targets)
        self.lw_constants = {
            "Target_Species_1": {"a": 0.015, "b": 3.0},
            "Target_Species_2": {"a": 0.020, "b": 2.9},
            # Default values for unrecognized species
            "Unknown": {"a": 0.018, "b": 3.0} 
        }

    def compute_weight(self, length_cm, species):
        """Calculates weight (g) from 3D length (cm) using W = aL^b"""
        constants = self.lw_constants.get(species, self.lw_constants["Unknown"])
        return constants["a"] * (length_cm ** constants["b"])

    def community_mean_metrics(self, df):
        """Mean Length and Weight across the entire community."""
        return {
            "Mean_Community_Length_cm": df['Length_3D_cm'].mean(),
            "Mean_Community_Weight_g": df['Weight_g'].mean()
        }

    def community_max_length(self, df):
        """Lmax: Quantifies changes in species composition."""
        max_per_species = df.groupby('Species')['Length_3D_cm'].max()
        return {"Community_Mean_Lmax_cm": max_per_species.mean()}

    def size_spectra(self, df):
        """
        Calculates Slope, Intercept, and Size Diversity Spectra (DS)
        using standard logarithmic (log2) size class bins.
        """
        # Create standard log2 bins (e.g., 2-4cm, 4-8cm, 8-16cm...)
        lengths = df['Length_3D_cm'].dropna()
        if len(lengths) < 3:
            return {"Spectra_Slope": None, "Spectra_Intercept": None}

        min_val, max_val = lengths.min(), lengths.max()
        if min_val <= 0: return {"Spectra_Slope": None, "Spectra_Intercept": None}
        
        bins = np.logspace(np.log2(min_val), np.log2(max_val), num=10, base=2)
        counts, bin_edges = np.histogram(lengths, bins=bins)
        
        # Filter empty bins to avoid log(0)
        valid = counts > 0
        log_counts = np.log10(counts[valid])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        log_bins = np.log10(bin_centers[valid])
        
        # Linear regression to find step-down slope
        if len(log_bins) > 1:
            slope, intercept, r_value, _, _ = linregress(log_bins, log_counts)
            return {
                "Spectra_Slope": slope,
                "Spectra_Intercept": intercept,
                "Size_Diversity_Spectra_DS": r_value**2 # Often correlated with variance
            }
        return {"Spectra_Slope": None, "Spectra_Intercept": None}

    def population_metrics(self, df):
        """
        Calculates Li, Lmax,i, and Fulton's K for each specific target species.
        """
        metrics = {}
        for species, group in df.groupby('Species'):
            l_95 = np.percentile(group['Length_3D_cm'], 95) if len(group) > 1 else group['Length_3D_cm'].max()
            
            # Fulton's Condition Index (K = 100 * W / L^3)
            # Measures morphological well-being
            k_indices = 100 * (group['Weight_g'] / (group['Length_3D_cm']**3))
            
            metrics[species] = {
                "Mean_Length_Pop_Li": group['Length_3D_cm'].mean(),
                "Mean_Weight_Pop_Wi": group['Weight_g'].mean(),
                "Max_Length_Pop_Lmax_i": group['Length_3D_cm'].max(),
                "Robust_Max_Pop_L95": l_95,
                "Mean_Fultons_K_Ki": k_indices.mean()
            }
        return metrics