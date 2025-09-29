#!/bin/bash
# Run MultiQC on FastQC results

mkdir -p multiqc
multiqc fastqc/ -o multiqc/
