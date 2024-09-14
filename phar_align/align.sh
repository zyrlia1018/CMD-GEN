#!/bin/bash

# Path to Python interpreter (adjust as needed)
PYTHON=python3

# Paths and parameters for the Python script
SMILES_PATH="../gen_result/phar_WRN_result.txt"
INPUT_PATH="../data/phar_WRN.posp"
OUTPUT_DIR="./WRN/"
PHAR_TOLERANCE=0

# Check if output directory exists, create if not
mkdir -p "$OUTPUT_DIR"

# Run the Python script
$PYTHON align_test_wrn.py \
    --smiles_path "$SMILES_PATH" \
    --input_path "$INPUT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --phar_tolerance $PHAR_TOLERANCE

