#!/bin/bash
set -e 

MODEL_DIR="/workspace/Wan2.2/Wan2.2-TI2V-5B"
PROMPTS_DIR="/workspace/outputs_dir"
WAN_DIR="/workspace/Wan2.2/outputs"


# Ensure the destination directory for the videos exists
mkdir -p "$WAN_DIR"

# 2. Safe conditional download directly into the persistent models volume
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "Downloading the Wan2.2 Model to $MODEL_DIR..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$MODEL_DIR"
else
    echo "Model already exists in $MODEL_DIR. Skipping download."
fi

echo "Processing Prompts."
if [ ! -f "$PROMPTS_DIR/prompts.txt" ]; then
    echo "[!] Error: prompts.txt not found in $PROMPTS_DIR!"
    exit 1
fi

set +e 

while IFS= read -r prompt || [ -n "$prompt" ]; do
    prompt=$(echo "$prompt" | xargs)

    if [[ -z "$prompt" || "$prompt" == \#* ]]; then
        continue
    fi

    # Create a clean, safe filename from the prompt to check for existence
    SAFE_NAME=$(echo "$prompt" | tail -c 50 | tr -dc '[:alnum:]_ ' | tr ' ' '_').mp4

    # Check if the file already exists in the designated wan directory
    if [ -f "$WAN_DIR/$SAFE_NAME" ]; then
        echo "[*] Video for '$prompt' already exists in $WAN_DIR. Skipping."
        continue
    fi

    echo ""
    echo "Generating video for: '$prompt'"
    
    # Run the generation using Distributed PyTorch (Multi-GPU) FSDP + Ulysses
    torchrun --nproc_per_node=2 generate.py \
        --task ti2v-5B \
        --size 1280*704 \
        --ckpt_dir "$MODEL_DIR" \
        --dit_fsdp \
        --t5_fsdp \
        --ulysses_size 2 \
        --prompt "$prompt"
        
    # Find the newest .mp4 file in the current working folder
    NEWEST_MP4=$(ls -t *.mp4 2>/dev/null | head -n 1)
    
    if [ -n "$NEWEST_MP4" ]; then
        # Move the file into the designated 'wan' directory using the safe name
        mv "$NEWEST_MP4" "$WAN_DIR/$SAFE_NAME"
        echo "[*] Successfully saved to $WAN_DIR/$SAFE_NAME"
    else
        echo "[!] Warning: Generation finished, but no .mp4 file was found."
    fi

done < "$PROMPTS_DIR/prompts.txt"

echo "Pipeline Complete!"