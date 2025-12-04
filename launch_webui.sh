#!/bin/bash

# ... (Keep your existing CONDA configuration here) ...
CONDA_SH_PATH="/home/jimi/.pyenv/versions/miniconda3-3.12-24.11.1-0/etc/profile.d/conda.sh" 

if [ -z "$CONDA_PREFIX" ]; then
    if [ -f "$CONDA_SH_PATH" ]; then
        source "$CONDA_SH_PATH"
    else
        echo "Error: conda.sh not found." >&2
        exit 1
    fi
fi

conda activate flashvsr || { echo "Error: Failed to activate conda environment."; exit 1; }
cd /home/jimi/Documents/FlashVSR || { echo "Error: Failed to change directory."; exit 1; }

echo "Launching Gradio web UI..."

# --- NEW CODE START ---
# Run a background process that sleeps for 5 seconds, then opens the browser
# We use xdg-open for Linux. 
(sleep 5 && xdg-open "http://127.0.0.1:7860") &
# --- NEW CODE END ---

python examples/WanVSR/webui.py

read -p "Press Enter to continue..."
