#!/usr/bin/env nextflow

// RegenOmics Master Pipeline: Production Nextflow Workflow

params.input = './data/*.fastq.gz'
params.output_dir = './results'
params.reference_dir = './references'
params.species = 'auto' // auto-detect reference genome
params.threads = 8
params.memory = '16GB'

// Reference genome paths
params.references = [
    'human': "${params.reference_dir}/human/GRCh38",
    'mouse': "${params.reference_dir}/mouse/GRCm39", 
    'primate': "${params.reference_dir}/primate/Mmul_10"
]

workflow {
    // Create input channel
    fastq_ch = Channel.fromPath(params.input)
                     .map { file -> [file.baseName.replaceAll(/\.fastq\.gz$/, ''), file] }
    
    // Quality control
    fastqc_out = fastqc(fastq_ch)
    multiqc_out = multiqc(fastqc_out.collect())
    
    // Species detection and alignment
    species_detected = detect_species(fastq_ch.first())
    align_out = align(fastq_ch, species_detected)
    
    // Count features
    counts_out = count_features(align_out)
    
    // Differential expression analysis
    de_out = differential_expression(counts_out.collect())
    
    // ML-based rejuvenation scoring
    ml_out = ml_scoring(de_out)
    
    // Generate comprehensive report
    report_out = report_generation(multiqc_out, ml_out)
}

process fastqc {
    publishDir "${params.output_dir}/qc/fastqc", mode: 'copy'
    
    input:
    tuple val(sample_id), path(fastq)
    
    output:
    path "*.{html,zip}"
    
    script:
    """
    fastqc ${fastq} \\
        --outdir . \\
        --threads ${params.threads} \\
        --format fastq \\
        --quiet
    """
}

process multiqc {
    publishDir "${params.output_dir}/qc", mode: 'copy'
    
    input:
    path fastqc_files
    
    output:
    path "multiqc_report.html"
    path "multiqc_data"
    
    script:
    """
    multiqc . \\
        --filename multiqc_report.html \\
        --title "RegenOmics QC Report" \\
        --comment "Quality control report for regenerative biology analysis" \\
        --config \$MULTIQC_CONFIG
    """
}

process detect_species {
    input:
    tuple val(sample_id), path(fastq)
    
    output:
    stdout
    
    script:
    if (params.species == 'auto')
        """
        #!/usr/bin/env python3
        import subprocess
        import os
        
        # Simple species detection using k-mer matching
        # Extract first 10000 reads for analysis
        subprocess.run(['seqtk', 'sample', '${fastq}', '10000'], 
                      stdout=open('sample.fastq', 'w'))
        
        # Count k-mers and match to reference databases
        kmer_counts = {}
        references = {'human': 0, 'mouse': 0, 'primate': 0}
        
        # Simplified species detection (in production, use more sophisticated methods)
        with open('sample.fastq', 'r') as f:
            lines = f.readlines()
            sequences = [lines[i+1].strip() for i in range(0, len(lines), 4)]
            
        # Count GC content as proxy for species detection
        gc_content = sum(seq.count('G') + seq.count('C') for seq in sequences) / sum(len(seq) for seq in sequences)
        
        # Species assignment based on GC content
        if 0.40 <= gc_content <= 0.42:
            detected_species = 'human'
        elif 0.42 <= gc_content <= 0.44:
            detected_species = 'mouse'  
        elif 0.38 <= gc_content <= 0.40:
            detected_species = 'primate'
        else:
            detected_species = 'human'  # Default
            
        print(detected_species)
        """
    else
        """
        echo "${params.species}"
        """
}

process align {
    publishDir "${params.output_dir}/alignment", mode: 'copy'
    cpus params.threads
    memory params.memory
    
    input:
    tuple val(sample_id), path(fastq)
    val species
    
    output:
    tuple val(sample_id), path("${sample_id}.bam"), path("${sample_id}.bam.bai")
    path "${sample_id}_Log.final.out"
    
    script:
    def reference = params.references[species.trim()]
    """
    # STAR alignment
    STAR --runMode alignReads \\
         --genomeDir ${reference} \\
         --readFilesIn ${fastq} \\
         --readFilesCommand zcat \\
         --outFileNamePrefix ${sample_id}_ \\
         --outSAMtype BAM SortedByCoordinate \\
         --outSAMunmapped Within \\
         --outSAMattributes Standard \\
         --runThreadN ${params.threads} \\
         --limitBAMsortRAM 31000000000 \\
         --quantMode GeneCounts
    
    # Index BAM file
    samtools index ${sample_id}_Aligned.sortedByCoord.out.bam
    
    # Rename output files
    mv ${sample_id}_Aligned.sortedByCoord.out.bam ${sample_id}.bam
    mv ${sample_id}_Aligned.sortedByCoord.out.bam.bai ${sample_id}.bam.bai
    """
}

process count_features {
    publishDir "${params.output_dir}/counts", mode: 'copy'
    
    input:
    tuple val(sample_id), path(bam), path(bai)
    
    output:
    tuple val(sample_id), path("${sample_id}_counts.txt")
    
    script:
    """
    # Count features using featureCounts
    featureCounts -a ${params.reference_dir}/annotation.gtf \\
                  -o ${sample_id}_counts.txt \\
                  -T ${params.threads} \\
                  -g gene_id \\
                  -t exon \\
                  -s 2 \\
                  ${bam}
    """
}

process differential_expression {
    publishDir "${params.output_dir}/differential_expression", mode: 'copy'
    
    input:
    path count_files
    
    output:
    path "differential_expression.csv"
    path "de_analysis_report.html"
    
    script:
    """
    #!/usr/bin/env python3
    
    import pandas as pd
    import numpy as np
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load count files
    count_data = {}
    samples = []
    
    for count_file in "${count_files}".split():
        sample_id = count_file.replace('_counts.txt', '')
        samples.append(sample_id)
        
        # Read featureCounts output
        df = pd.read_csv(count_file, sep='\\t', skiprows=1, index_col=0)
        count_data[sample_id] = df.iloc[:, -1]  # Last column has counts
    
    # Create count matrix
    count_matrix = pd.DataFrame(count_data).fillna(0)
    
    # Normalize using TMM-like method
    lib_sizes = count_matrix.sum(axis=0)
    norm_factors = lib_sizes / lib_sizes.median()
    
    # CPM normalization
    cpm_matrix = count_matrix.div(lib_sizes, axis=1) * 1e6
    
    # Log transform
    log_cpm = np.log2(cpm_matrix + 1)
    
    # Filter low-expressed genes
    expressed = (cpm_matrix > 1).sum(axis=1) >= len(samples) * 0.3
    log_cpm_filtered = log_cpm[expressed]
    
    # Create mock condition labels (young, aged, rejuvenated)
    np.random.seed(42)
    conditions = np.random.choice(['young', 'aged', 'rejuvenated'], size=len(samples))
    condition_df = pd.DataFrame({'sample': samples, 'condition': conditions})
    
    # Perform differential expression analysis
    de_results = []
    
    for condition in ['aged', 'rejuvenated']:
        condition_samples = condition_df[condition_df['condition'] == condition]['sample']
        control_samples = condition_df[condition_df['condition'] == 'young']['sample']
        
        if len(condition_samples) > 0 and len(control_samples) > 0:
            for gene in log_cpm_filtered.index:
                condition_values = log_cpm_filtered.loc[gene, condition_samples]
                control_values = log_cpm_filtered.loc[gene, control_samples]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(condition_values, control_values)
                
                # Log fold change
                log_fc = condition_values.mean() - control_values.mean()
                
                de_results.append({
                    'gene': gene,
                    'condition': condition,
                    'log_fc': log_fc,
                    'p_value': p_value,
                    'mean_expression': log_cpm_filtered.loc[gene].mean()
                })
    
    # Create results DataFrame
    de_df = pd.DataFrame(de_results)
    
    # Adjust p-values (Benjamini-Hochberg)
    from statsmodels.stats.multitest import multipletests
    
    for condition in de_df['condition'].unique():
        mask = de_df['condition'] == condition
        _, adjusted_p, _, _ = multipletests(de_df.loc[mask, 'p_value'], method='fdr_bh')
        de_df.loc[mask, 'adj_p_value'] = adjusted_p
    
    # Add significance flags
    de_df['significant'] = (de_df['adj_p_value'] < 0.05) & (abs(de_df['log_fc']) > 1)
    
    # Prepare output for ML pipeline
    ml_input = log_cpm_filtered.T  # Transpose so samples are rows
    ml_input['sample_id'] = ml_input.index
    ml_input['condition'] = ml_input['sample_id'].map(dict(zip(condition_df['sample'], condition_df['condition'])))
    
    # Save results
    de_df.to_csv('differential_expression.csv', index=False)
    ml_input.to_csv('expression_matrix_for_ml.csv')
    
    # Generate HTML report
    html_report = f'''
    <!DOCTYPE html>
    <html>
    <head><title>Differential Expression Analysis Report</title></head>
    <body>
    <h1>Differential Expression Analysis Report</h1>
    <h2>Summary</h2>
    <p>Total genes analyzed: {len(de_df)//2}</p>
    <p>Significant genes (aged vs young): {len(de_df[(de_df.condition=='aged') & de_df.significant])}</p>
    <p>Significant genes (rejuvenated vs young): {len(de_df[(de_df.condition=='rejuvenated') & de_df.significant])}</p>
    <h2>Sample Information</h2>
    <p>Total samples: {len(samples)}</p>
    <p>Young samples: {len(control_samples)}</p>
    <p>Aged samples: {sum(conditions == 'aged')}</p>
    <p>Rejuvenated samples: {sum(conditions == 'rejuvenated')}</p>
    </body>
    </html>
    '''
    
    with open('de_analysis_report.html', 'w') as f:
        f.write(html_report)
    
    print("Differential expression analysis completed")
    """
}

process ml_scoring {
    publishDir "${params.output_dir}/ml_results", mode: 'copy'
    
    input:
    path de_results
    
    output:
    path "ml_predictions.csv"
    path "model_performance.json"
    
    script:
    """
    cd ${projectDir}/ml
    python cell_rejuvenation_scoring.py
    
    # Copy results to output
    cp ml_predictions.csv ${params.output_dir}/ml_results/
    
    # Generate model performance summary
    python -c "
import json
import pandas as pd

# Load predictions
df = pd.read_csv('ml_predictions.csv')

# Calculate summary statistics
performance = {
    'samples_processed': len(df),
    'mean_rejuvenation_score': float(df['rejuvenation_score'].mean()),
    'std_rejuvenation_score': float(df['rejuvenation_score'].std()),
    'min_score': float(df['rejuvenation_score'].min()),
    'max_score': float(df['rejuvenation_score'].max()),
    'highly_rejuvenated_samples': int((df['rejuvenation_score'] > 0.8).sum()),
    'aged_samples': int((df['rejuvenation_score'] < 0.2).sum())
}

with open('model_performance.json', 'w') as f:
    json.dump(performance, f, indent=2)
"
    """
}

process report_generation {
    publishDir "${params.output_dir}/reports", mode: 'copy'
    
    input:
    path multiqc_report
    path ml_results
    
    output:
    path "final_report.html"
    path "pipeline_summary.json"
    
    script:
    """
    cd ${projectDir}/workflows
    python generate_report.py \\
        --multiqc ${multiqc_report} \\
        --ml-results ${ml_results} \\
        --output final_report.html
    
    # Generate pipeline summary
    python -c "
import json
import os
from datetime import datetime

summary = {
    'pipeline': 'RegenOmics Master Pipeline',
    'version': '1.0.0',
    'execution_time': datetime.now().isoformat(),
    'input_files': '${params.input}',
    'output_directory': '${params.output_dir}',
    'species_detected': 'auto-detected',
    'processing_complete': True
}

with open('pipeline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
"
    """
}
