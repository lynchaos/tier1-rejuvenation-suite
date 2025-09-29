# Single-Cell Rejuvenation Atlas: Seurat-based Analysis
# Complementary R analysis pipeline using Seurat

library(Seurat)
library(dplyr)
library(ggplot2)
library(patchwork)
library(monocle3)

#' Rejuvenation Analysis with Seurat
#' @param seurat_obj Seurat object containing single-cell data
#' @return Processed Seurat object with rejuvenation analysis
rejuvenation_analysis <- function(seurat_obj) {
  
  # Standard preprocessing
  cat("Starting Seurat preprocessing...\n")
  
  # QC and filtering
  seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
  seurat_obj <- subset(seurat_obj, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 20)
  
  # Normalization
  seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
  
  # Find variable features
  seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
  
  # Scaling
  all.genes <- rownames(seurat_obj)
  seurat_obj <- ScaleData(seurat_obj, features = all.genes)
  
  # PCA
  seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj))
  
  # Clustering
  seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
  seurat_obj <- FindClusters(seurat_obj, resolution = 0.5)
  
  # UMAP
  seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)
  
  cat("Basic preprocessing complete\n")
  return(seurat_obj)
}

#' Trajectory Analysis for Aging/Rejuvenation
trajectory_analysis <- function(seurat_obj) {
  cat("Performing trajectory analysis...\n")
  
  # Convert to monocle3 format for trajectory analysis
  cds <- as.cell_data_set(seurat_obj)
  cds <- preprocess_cds(cds, num_dim = 100)
  cds <- align_cds(cds, alignment_group = "batch", residual_model_formula_str = "~ bg.300.loading + bg.400.loading + bg.500.1.loading + bg.500.2.loading")
  cds <- reduce_dimension(cds)
  cds <- cluster_cells(cds)
  cds <- learn_graph(cds)
  
  # Order cells (would need to specify root cells based on biology)
  # cds <- order_cells(cds)
  
  cat("Trajectory analysis complete\n")
  return(list(seurat = seurat_obj, monocle = cds))
}

#' Senescence Marker Analysis
senescence_analysis <- function(seurat_obj) {
  cat("Analyzing senescence markers...\n")
  
  senescence_markers <- c("CDKN1A", "CDKN2A", "TP53", "RB1", "GLB1")
  available_markers <- intersect(senescence_markers, rownames(seurat_obj))
  
  if(length(available_markers) > 0) {
    # Add senescence score
    seurat_obj <- AddModuleScore(seurat_obj, 
                                features = list(available_markers),
                                name = "Senescence_Score")
    
    # Plot senescence markers
    p1 <- FeaturePlot(seurat_obj, features = "Senescence_Score1")
    p2 <- VlnPlot(seurat_obj, features = "Senescence_Score1", group.by = "seurat_clusters")
    
    print(p1 | p2)
  }
  
  cat(paste("Senescence analysis complete for", length(available_markers), "markers\n"))
  return(seurat_obj)
}

#' Pluripotency Scoring
pluripotency_analysis <- function(seurat_obj) {
  cat("Scoring pluripotency markers...\n")
  
  pluripotency_markers <- c("POU5F1", "SOX2", "KLF4", "MYC", "NANOG")
  available_markers <- intersect(pluripotency_markers, rownames(seurat_obj))
  
  if(length(available_markers) > 0) {
    # Add pluripotency score
    seurat_obj <- AddModuleScore(seurat_obj, 
                                features = list(available_markers),
                                name = "Pluripotency_Score")
    
    # Plot pluripotency
    p1 <- FeaturePlot(seurat_obj, features = "Pluripotency_Score1")
    p2 <- VlnPlot(seurat_obj, features = "Pluripotency_Score1", group.by = "seurat_clusters")
    
    print(p1 | p2)
  }
  
  cat(paste("Pluripotency analysis complete for", length(available_markers), "markers\n"))
  return(seurat_obj)
}

#' Cell-Cell Communication Analysis
communication_analysis <- function(seurat_obj) {
  cat("Analyzing cell-cell communication...\n")
  
  # This would integrate with CellChat or similar package
  # For now, just identify potential ligand-receptor pairs
  
  # Example: find DEGs for each cluster
  markers <- FindAllMarkers(seurat_obj, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
  
  cat("Communication analysis complete\n")
  return(markers)
}

#' Main Analysis Pipeline
run_rejuvenation_seurat_pipeline <- function(data_path) {
  cat("Starting Seurat-based Rejuvenation Analysis Pipeline...\n")
  
  # Load data
  seurat_obj <- Read10X(data.dir = data_path)
  seurat_obj <- CreateSeuratObject(counts = seurat_obj, project = "rejuvenation", min.cells = 3, min.features = 200)
  
  # Run analysis steps
  seurat_obj <- rejuvenation_analysis(seurat_obj)
  
  trajectory_result <- trajectory_analysis(seurat_obj)
  seurat_obj <- trajectory_result$seurat
  
  seurat_obj <- senescence_analysis(seurat_obj)
  seurat_obj <- pluripotency_analysis(seurat_obj)
  
  communication_markers <- communication_analysis(seurat_obj)
  
  # Generate summary plots
  p1 <- DimPlot(seurat_obj, reduction = "umap", group.by = "seurat_clusters")
  p2 <- DimPlot(seurat_obj, reduction = "umap", group.by = "orig.ident")
  print(p1 | p2)
  
  # Save results
  saveRDS(seurat_obj, file = "rejuvenation_seurat_analysis.rds")
  write.csv(communication_markers, file = "communication_markers.csv")
  
  cat("Seurat pipeline complete!\n")
  return(seurat_obj)
}

# Example usage
if (FALSE) {
  # Run with actual data
  seurat_result <- run_rejuvenation_seurat_pipeline("../data/10x_data/")
}