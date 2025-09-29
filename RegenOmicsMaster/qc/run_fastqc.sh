#!/bin/bash
# Run FastQC on all FASTQ files in data/

mkdir -p fastqc
for f in ../data/*.fastq.gz; do
    fastqc "$f" -o fastqc/
done
