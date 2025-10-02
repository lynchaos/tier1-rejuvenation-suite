# Enhanced Biomarker Panel - Integration of Developer Proposals
# ==========================================================
# Merging current YAML implementation with proposed CSV enhancements

import pandas as pd
import yaml
from pathlib import Path

def integrate_biomarker_enhancements():
    """
    Integrate developer proposals into current biomarker panel
    """
    
    # Load current YAML implementation
    current_panel_path = Path("/home/pi/projects/data/validated_biomarkers.yaml")
    with open(current_panel_path, 'r') as f:
        current_panel = yaml.safe_load(f)
    
    # Load proposed CSV panel
    proposed_panel_path = Path("/home/pi/projects/proposals/newfilesforomics/curated_aging_biomarker_panel.csv")
    proposed_df = pd.read_csv(proposed_panel_path)
    
    print("🔬 Current Panel Analysis:")
    current_genes = set()
    for pathway, data in current_panel['biomarkers'].items():
        pathway_genes = [marker['gene'] for marker in data['markers']]
        current_genes.update(pathway_genes)
        print(f"  {pathway}: {len(pathway_genes)} genes")
    
    print(f"\nTotal current genes: {len(current_genes)}")
    
    print("\n🆕 Proposed Panel Analysis:")
    proposed_genes = set(proposed_df['gene_symbol'].tolist())
    proposed_categories = proposed_df['category'].unique()
    
    for category in proposed_categories:
        category_genes = proposed_df[proposed_df['category'] == category]['gene_symbol'].tolist()
        print(f"  {category}: {len(category_genes)} genes")
    
    print(f"\nTotal proposed genes: {len(proposed_genes)}")
    
    # Find missing genes
    missing_genes = proposed_genes - current_genes
    print(f"\n📋 Missing genes in current implementation: {len(missing_genes)}")
    
    # New categories in proposal
    current_categories = set(current_panel['biomarkers'].keys())
    proposed_cat_mapped = {
        'cellular_senescence': 'cellular_senescence',
        'dna_damage_repair': 'dna_damage_repair', 
        'mitochondrial_function': 'mitochondrial_function',
        'telomere_maintenance': 'telomere_maintenance',
        'autophagy_proteostasis': 'autophagy_proteostasis',
        'inflammation_immunity': 'inflammation_SASP',
        'metabolic_pathways': 'metabolism',
        'epigenetic_regulators': 'epigenetics',
        'ecm_remodeling': 'ecm_remodeling',  # NEW
        'stem_cell_regeneration': 'stem_cell_regeneration'  # NEW
    }
    
    new_categories = set(proposed_cat_mapped.values()) - current_categories
    print(f"\n🆕 New categories to add: {new_categories}")
    
    return {
        'current_genes': current_genes,
        'proposed_genes': proposed_genes,
        'missing_genes': missing_genes,
        'new_categories': new_categories,
        'proposed_df': proposed_df,
        'current_panel': current_panel
    }

if __name__ == "__main__":
    analysis = integrate_biomarker_enhancements()