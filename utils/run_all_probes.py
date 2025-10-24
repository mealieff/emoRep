#!/bin/bash
# =====================================================
# run_all_probes.sh
# Runs all five emotion reasoning probe training scripts
# and stores outputs + evaluation summaries
# =====================================================

# Exit on error
set -e

# Timestamp for this batch run
DATE=$(date +"%Y%m%d_%H%M%S")

# Output directory
OUTPUT_DIR="../output/probe_run_$DATE"
mkdir -p "$OUTPUT_DIR"

# Log file for overall run
LOG_FILE="$OUTPUT_DIR/full_run.log"

echo "===========================================" | tee -a "$LOG_FILE"
echo " Running All Probe Training Scripts         " | tee -a "$LOG_FILE"
echo " Timestamp: $DATE                           " | tee -a "$LOG_FILE"
echo " Output Directory: $OUTPUT_DIR              " | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# --- Probe 1: Acoustic Understanding ---
echo "[1/5] Running Acoustic Understanding Probe..."
python3 train_acoustic.py --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# --- Probe 2: Semantic Understanding ---
echo "[2/5] Running Semantic Understanding Probe..."
python3 train_semantic.py --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# --- Probe 3: Temporal Coherence ---
echo "[3/5] Running Temporal Coherence Probe..."
python3 train_temporal.py --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# --- Probe 4: Affective Reasoning ---
echo "[4/5] Running Affective Reasoning Probe..."
python3 train_affective.py --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

# --- Probe 5: Social Context Grounding ---
echo "[5/5] Running Social Context Grounding Probe..."
python3 train_social.py --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
echo " ✅ All Probes Completed Successfully!" | tee -a "$LOG_FILE"
echo " Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"

