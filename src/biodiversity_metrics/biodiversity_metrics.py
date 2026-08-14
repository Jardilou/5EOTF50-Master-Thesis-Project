import numpy as np
import pandas as pd
from scipy.stats import linregress
import os

class ReefMetricsCalculator:
    """
    Computes Community-Level and Population-Level indicators for coral reef health
    based on stereovision 3D size and species classification.
    """
    def __init__(self):
        # Ecological Length-Weight constants: W = a * L^b
        self.lw_constants = {
            "Dascyllus reticulatus": {"a": 0.02951, "b": 2.99},
            "Scaridae": {"a": 0.02138, "b": 2.99},
            "Pomacentrus moluccensis": {"a": 0.02951, "b": 2.96},
            "Unknown": {"a": 0.018, "b": 3.0} 
        }

    def compute_weight(self, length_cm, species):
        """Calculates weight (g) from 3D length (cm) using W = aL^b"""
        constants = self.lw_constants.get(species, self.lw_constants["Unknown"])
        return constants["a"] * (length_cm ** constants["b"])

    def community_mean_metrics(self, df):
        return {
            "Total_Community_Abundance": len(df),
            "Mean_Community_Length_cm": df['Length_3D_cm'].mean(),
            "Mean_Community_Weight_g": df['Weight_g'].mean()
        }

    def community_max_length(self, df):
        max_per_species = df.groupby('Species')['Length_3D_cm'].max()
        return {"Community_Mean_Lmax_cm": max_per_species.mean()}

    def size_spectra(self, df):
        lengths = df['Length_3D_cm'].dropna()
        if len(lengths) < 3:
            return {"Spectra_Slope": None, "Spectra_Intercept": None}

        min_val, max_val = lengths.min(), lengths.max()
        if min_val <= 0: return {"Spectra_Slope": None, "Spectra_Intercept": None}
        
        bins = np.logspace(np.log2(min_val), np.log2(max_val), num=10, base=2)
        counts, bin_edges = np.histogram(lengths, bins=bins)
        
        valid = counts > 0
        log_counts = np.log10(counts[valid])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        log_bins = np.log10(bin_centers[valid])
        
        if len(log_bins) > 1:
            slope, intercept, r_value, _, _ = linregress(log_bins, log_counts)
            return {
                "Spectra_Slope": slope,
                "Spectra_Intercept": intercept,
                "Size_Diversity_Spectra_DS": r_value**2
            }
        return {"Spectra_Slope": None, "Spectra_Intercept": None}

    def population_metrics(self, df):
        """
        Calculates Population metrics including MaxN, MeanCount, and Fulton's K.
        """
        metrics = {}
        has_frames = 'Frame_ID' in df.columns

        for species, group in df.groupby('Species'):
            l_95 = np.percentile(group['Length_3D_cm'], 95) if len(group) > 1 else group['Length_3D_cm'].max()
            
            # Fulton's Condition Index (K = 100 * W / L^3)
            k_indices = 100 * (group['Weight_g'] / (group['Length_3D_cm']**3))
            
            # Calculate MaxN and MeanCount
            if has_frames:
                counts_per_frame = group.groupby('Frame_ID').size()
                maxn = counts_per_frame.max()
                meancount = counts_per_frame.mean()
            else:
                maxn = len(group)
                meancount = len(group)

            metrics[species] = {
                "Total_Observations": len(group),
                "MaxN": maxn,
                "MeanCount": meancount,
                "Mean_Length_Pop_Li": group['Length_3D_cm'].mean(),
                "Mean_Weight_Pop_Wi": group['Weight_g'].mean(),
                "Max_Length_Pop_Lmax_i": group['Length_3D_cm'].max(),
                "Robust_Max_Pop_L95": l_95,
                "Mean_Fultons_K_Ki": k_indices.mean()
            }
        return metrics

    def export_to_csv(self, df, pop_metrics, comm_means, comm_lmax, spectra, output_dir="outputs/metrics"):
        """Saves all calculated dataframes and dictionaries into CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save the raw cleaned dataframe
        raw_path = os.path.join(output_dir, "raw_fish_data.csv")
        df.to_csv(raw_path, index=False)
        
        # 2. Save Population Metrics
        pop_df = pd.DataFrame.from_dict(pop_metrics, orient='index')
        pop_df.index.name = 'Species'
        pop_df.to_csv(os.path.join(output_dir, "population_metrics.csv"))
        
        # 3. Save Community Metrics (combined into one row)
        comm_dict = {**comm_means, **comm_lmax, **spectra}
        comm_df = pd.DataFrame([comm_dict])
        comm_df.to_csv(os.path.join(output_dir, "community_metrics.csv"), index=False)
        
        print(f"\n[+] Success: Ecological metrics saved to CSV in '{output_dir}'")