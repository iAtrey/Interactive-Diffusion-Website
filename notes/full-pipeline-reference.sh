#!/bin/bash

# This file is old. Refer to tutorial.MD. 

# ── STEP 1: Log into the server ─────────────────────────────────────────────
ssh [NAME]@[IP]
git clone https://github.com/JinhuaLyu/DiffATS.git
cd DiffATS


# ── STEP 2: Set up your main environment ────────────────────────────────────
python -m venv [Main_Env_Name]

source /home/[NAME]/[Main_Env_Name]/bin/activate


pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt   # installs the rest of DiffATS's dependencies

python3 -c "import torch; print(torch.cuda.is_available())"   # must print True


# ── STEP 3: Check the GPU ───────────────────────────────────────────────────

nvidia-smi

# ── STEP 4: Handle the Taichi / Python-version problem ──────────────────────
# Only needed if your server's default Python is too new for Taichi (Taichi generally needs Python 3.10–3.12).
python3 --version 

pip install uv
uv python install 3.11
uv venv --python 3.11 [TAICHI_ENV_NAME]

# stay in your MAIN env (the one with uv) and install into the Taichi env:
uv pip install --python [TAICHI_ENV_NAME]/bin/python3.11 \
    taichi torch numpy tqdm scipy


# ── STEP 5: Fix the GPU index in the data generator ─────────────────────────
DATA_GEN_FILE="exps/tensor_physics/exp_karman_vortex/data_generation/generate_data.py"

cp "$DATA_GEN_FILE" "$DATA_GEN_FILE.bak"   # always back up before editing

# Check what index is currently hardcoded:
grep CUDA_VISIBLE_DEVICES "$DATA_GEN_FILE"

# If it doesn't match your real GPU index from Step 3, fix it:
sed -i "s/CUDA_VISIBLE_DEVICES'\] = '[OLD_GPU_INDEX]'/CUDA_VISIBLE_DEVICES'] = '[YOUR_GPU_INDEX]'/" "$DATA_GEN_FILE"


# ── STEP 6: Generate the FULL raw dataset (all 10,000 clips) ───────────────

cd ~/DiffATS/exps/tensor_physics/exp_karman_vortex/data_generation

[TAICHI_ENV_NAME]/bin/python3.11 generate_data.py \
    --out-dir data

# ── STEP 7: Extract and align Tucker factors from all 10,000 clips ─────────
cd ../data_tucker

[TAICHI_ENV_NAME]/bin/python3.11 save_tucker_karman.py \
    --data_dir ../data_generation/data \
    --out_dir [TUCKER_OUTPUT_DIR_NAME] \
    --seed 42 \
    --n_workers [NUMBER_OF_CPU_CORES]

# ── STEP 8: Confirm the pretrained checkpoint is present ───────────────────
ls [PATH_TO_CHECKPOINT_DIRECTORY]


# ── STEP 9: Run inference ──────────────────────────────────────────────────

cd ../generate
source /home/[NAME]/[Main_Env_Name]/bin/activate
mkdir -p output

python3 gen_karman_2d.py \
  --ckpt [PATH_TO_CHECKPOINT_DIRECTORY]/karman_vortex_2d_epoch00500_step0156000.pt \
  --output_dir output \
  --train_data_dir ../data_tucker/[TUCKER_OUTPUT_DIR_NAME] \
  --test_data_dir ../data_tucker/[TUCKER_OUTPUT_DIR_NAME]/test_data \
  --batch_size 50 \
  --seeds 0 1 2 3 4 \
  --sample_steps 250 \
  --device cuda:[YOUR_GPU_INDEX]


# ── STEP 10: Reconstruct the video ──────────────────────────────────────────
pip install imageio imageio-ffmpeg matplotlib

python3 make_video.py \
  --gen_path output/[GENERATED_FACTOR_FILENAME].pt \
  --sample_idx [WHICH_SAMPLE_TO_RENDER] \
  --out_dir output/[VIDEO_OUTPUT_FOLDER_NAME]


# ── Download the finished video to your own computer ───────────────────────

scp [NAME]@[IP]:~/DiffATS/exps/tensor_physics/exp_karman_vortex/generate/output/[VIDEO_OUTPUT_FOLDER_NAME]/video.mp4 ./ 
