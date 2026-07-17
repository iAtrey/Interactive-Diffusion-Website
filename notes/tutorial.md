# Running DiffATS Kármán Vortex from scratch

This goes from a fresh login to a finished video. Everything here uses scripts that already exist in the repo.

Replace anything in brackets with your own value: `[USERNAME]`, `[IP]`, `[CHECKPOINT_DIR]`, `[MAIN_ENV]`, `[TAICHI_ENV]`, `[GPU_INDEX]`, `[CPU_CORES]`.

The pipeline is: raw data → Tucker factors → inference → reconstruct → video.

## 1. Log in and clone

```bash
ssh [USERNAME]@[IP]
cd ~
git clone https://github.com/JinhuaLyu/DiffATS.git
cd DiffATS
```

The checkpoint isn't in the repo, it's given to you separately. Find where it lives:

```bash
find / -iname "karman_vortex_2d_*.pt" 2>/dev/null
```

Whatever folder that prints is your `[CHECKPOINT_DIR]`. You need it in step 8. If it prints nothing, you don't have the checkpoint yet and there's no point continuing, since step 8 is the only thing that uses it and it can't be generated locally.

## 2. Main environment

```bash
python3 -m venv ~/[MAIN_ENV]
source ~/[MAIN_ENV]/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib pillow tqdm pyyaml imageio imageio-ffmpeg
python3 -c "import torch; print(torch.cuda.is_available())"
```

That last line has to print True. If it doesn't, fix it before going further.

## 3. Check the GPU

```bash
nvidia-smi
nvidia-smi -L
```

Write down your GPU number, that's your `[GPU_INDEX]`. On a single-GPU machine it's 0. Also make sure nobody else is running a job, since two at once can run the GPU out of memory.

## 4. Taichi environment

The data generator needs Taichi, and Taichi doesn't support new Python versions. Check yours:

```bash
python3 --version
```

If it's 3.13 or newer, pip will say "No matching distribution found" for taichi. That's a version problem, not a network problem. Use uv to get an older Python without sudo:

```bash
pip install uv
uv python install 3.11
uv venv --python 3.11 ~/DiffATS/[TAICHI_ENV]
uv pip install --python ~/DiffATS/[TAICHI_ENV]/bin/python3.11 taichi torch numpy tqdm scipy
```

Stay in your main environment while running that last command, since that's where uv lives. If you activate the Taichi env, uv disappears.

Check it works:

```bash
~/DiffATS/[TAICHI_ENV]/bin/python3.11 -c "import taichi as ti; ti.init(arch=ti.gpu); print('OK')"
```

From here, use `[TAICHI_ENV]` for steps 6 and 7, and `[MAIN_ENV]` for steps 8, 9, and 10. Wrong environment gives you ModuleNotFoundError.

If your Python is already 3.10–3.12 you can skip all this and install taichi normally.

## 5. Fix the GPU number in the data generator

The generator has a GPU number written into the file:

```bash
cd ~/DiffATS/exps/tensor_physics/exp_karman_vortex/data_generation
grep CUDA_VISIBLE_DEVICES generate_data.py
```

If it doesn't match your `[GPU_INDEX]`, back up and change it:

```bash
cp generate_data.py generate_data.py.bak
sed -i "s/CUDA_VISIBLE_DEVICES'\] = '.*'/CUDA_VISIBLE_DEVICES'] = '[GPU_INDEX]'/" generate_data.py
grep CUDA_VISIBLE_DEVICES generate_data.py
```

Note that the script still prints "Using GPU (device 4)" when it runs. That text is hardcoded in a print statement and isn't the real device. Ignore it.

## 6. Generate the raw data

```bash
cd ~/DiffATS/exps/tensor_physics/exp_karman_vortex/data_generation
~/DiffATS/[TAICHI_ENV]/bin/python3.11 generate_data.py --out-dir data
```

With no flags this runs the full 200 parameter sets × 50 clips = 10,000 clips. Expect several hours. If you get disconnected, rerun the same command; it skips shards that already exist.

Output is `data/shard_000.pt` through `data/shard_199.pt`.

To change the clip count, two flags:

- `--param-range START END` — which parameter sets, inclusive. Default is all 200.
- `--clips N` — clips per set. Default is 50.

Total = (parameter sets) × (clips per set).

```bash
# full: 10,000 clips, hours
python3 generate_data.py --out-dir data

# quick test: 15 clips, about a minute
python3 generate_data.py --param-range 0 2 --clips 5 --out-dir data_test
```

A small run is fine to test that the pipeline works, but the quality will be worse. Step 7 builds the reference anchor and the normalization numbers from whatever you give it, so a narrow slice of the data means those numbers won't match what the checkpoint was trained on.

You can also split the work across terminals by giving each a different `--param-range`.

## 7. Tucker decomposition

```bash
cd ~/DiffATS/exps/tensor_physics/exp_karman_vortex/data_tucker
~/DiffATS/[TAICHI_ENV]/bin/python3.11 save_tucker_karman.py \
    --data_dir ../data_generation/data \
    --out_dir tucker_karman_rT10_rX128_rY30 \
    --seed 42 \
    --n_workers [CPU_CORES]
```

This compresses every clip and rotates them all to line up with one reference clip. It produces `ref_anchor.pt`, `norm_stats.pt`, `manifest.txt`, and the shard files automatically.

Don't change `--seed 42`. The seed picks the reference anchor, and changing it changes the coordinate system the checkpoint expects.

At the end it prints a sanity check. Tucker recon RelErr around 0.05 is normal, IC SVD RelErr should be under 0.1.

For the test set, run the same script with `--anchor_path` pointing at the `ref_anchor.pt` from above, so it reuses the same anchor instead of picking a new one, and `--shard_pattern` to select the test shards. If you were given a prebuilt test set, just copy it into `tucker_karman_rT10_rX128_rY30/test_data/` and write a manifest:

```bash
cd tucker_karman_rT10_rX128_rY30/test_data
ls *.pt > manifest.txt
```

## 8. Inference

Back to the main environment for this.

```bash
cd ~/DiffATS/exps/tensor_physics/exp_karman_vortex/generate
source ~/[MAIN_ENV]/bin/activate
mkdir -p output

python3 gen_karman_2d.py \
  --ckpt [CHECKPOINT_DIR]/karman_vortex_2d_epoch00500_step0156000.pt \
  --output_dir output \
  --train_data_dir ../data_tucker/tucker_karman_rT10_rX128_rY30 \
  --test_data_dir ../data_tucker/tucker_karman_rT10_rX128_rY30/test_data \
  --batch_size 50 \
  --seeds 0 \
  --sample_steps 250 \
  --epoch_tag epoch00500 \
  --device cuda:[GPU_INDEX]
```

Output is `output/epoch00500_seed0.pt`, which holds the generated Tucker factors, not pixels.

`--epoch_tag` only sets the filename, but set it to match your checkpoint's epoch because steps 9 and 10 look for that exact name. Leaving it at the default gives you a file labelled epoch00200 even though the checkpoint is epoch 500.

`--seeds 0` does one set; `--seeds 0 1 2 3 4` does five and takes five times as long. Lower `--batch_size` if you run out of GPU memory. Roughly 10 minutes per 50 samples.

The script asserts the test set is exactly 500 samples, around line 190. On the full run this passes on its own and you don't touch it. If you used a smaller `--param-range` in step 6, your test set won't be 500 and the assert stops the run before it starts.

To get past it, check the line numbers first, since they can shift:

```bash
grep -n "assert len(test_dataset)" gen_karman_2d.py
sed -n '189,191p' gen_karman_2d.py
```

That should show the print line, then the two assert lines:

```
print(f'Test samples: {len(test_dataset)}', flush=True)
    assert len(test_dataset) == 500, \
        f'Expected 500 test samples, got {len(test_dataset)}'
```

The assert is two lines, not one, because of the line continuation. Back up, delete both, and check that the print survived and nothing else got cut:

```bash
cp gen_karman_2d.py gen_karman_2d.py.bak
sed -i "190,191d" gen_karman_2d.py
sed -n '185,192p' gen_karman_2d.py
```

Only do this if you know why your test set isn't 500. The assert is there to catch a broken test set, so removing it silences a real check. Write down that you removed it, since it means you're off the intended setup and the results aren't comparable to a full run.

## 9. Reconstruct the videos

```bash
cd ~/DiffATS/exps/tensor_physics/tools
python3 reconstruct_gen.py \
  --exp karman \
  --seed 0 \
  --epoch 500 \
  --dir ~/DiffATS/exps/tensor_physics/exp_karman_vortex/generate/output
```

This turns the factors into actual pixels and writes `epoch00500_seed0_videos.pt` next to the input. The epoch and seed have to match the filename from step 8, since the script builds the input path from those numbers.

The output is several GB for 500 samples, so check your space.

## 10. Render the video

The renderer is `exps/tensor_physics/visualization/making_videos.py`. It puts the real simulation on the left and the generated one on the right, holds the conditioning frame for a few seconds, then plays both. It writes two color styles in one run.

Open it and set the three paths at the top:

```bash
cd ~/DiffATS/exps/tensor_physics/visualization
nano making_videos.py
```

That opens the file in the terminal. Arrow down to the `GEN_PATH`, `GT_DIR`, and `OUT_DIR` lines near the top and change them to:

```python
GEN_PATH = "/home/[USERNAME]/DiffATS/exps/tensor_physics/exp_karman_vortex/generate/output/epoch00500_seed0_videos.pt"
GT_DIR   = "/home/[USERNAME]/DiffATS/exps/tensor_physics/exp_karman_vortex/data_generation/data"
OUT_DIR  = "/home/[USERNAME]/DiffATS/exps/tensor_physics/exp_karman_vortex/generate/output/videos"
```

Save with Ctrl+O, then Enter. Exit with Ctrl+X.

Check the three paths actually took, and that the file still parses, since pasting into nano can mangle things:

```bash
grep -n "^GEN_PATH\|^GT_DIR\|^OUT_DIR" making_videos.py
python3 -c "import ast; ast.parse(open('making_videos.py').read()); print('syntax ok')"
```

If you'd rather not use nano, scp the file to your laptop, edit it in whatever editor you like, and scp it back. Or use the VS Code Remote SSH extension and edit it in place.

`GT_DIR` has to point at the raw shards with a `vor` key, not the Tucker ones. Note that the script looks for files named `test_shard_NNN.pt` while `generate_data.py` writes `shard_NNN.pt`, so you may need to rename or edit the pattern.

Check ffmpeg is reachable:

```bash
which ffmpeg
```

If that prints nothing, the script exits with an error. On a cluster you'd run `module load ffmpeg/4.2.2`. The fallback path inside the script points at a specific cluster and won't exist elsewhere.

Then run it:

```bash
source ~/[MAIN_ENV]/bin/activate
python3 making_videos.py
```

You get four files in `OUT_DIR`: a gif and mp4 for each of the two color styles. `SEED` inside the file picks which sample gets rendered, `FPS` is the speed, and `CONDITION_SECONDS` is how long the first frame is held.

If you get "param mismatch on niu", the generated sample and the ground truth clip aren't the same case. Check `SAMPLES_PER_SHARD` against how many samples are actually in each of your ground truth shards.

## 11. Copy it to your laptop

Run this on your laptop, not the server. Open a second terminal so your server session stays open.

```bash
scp [USERNAME]@[IP]:~/DiffATS/exps/tensor_physics/exp_karman_vortex/generate/output/videos/karman_vortex_gt_vs_gen_redblue.mp4 ./
```

The `./` means the current folder on your laptop. Run `pwd` first if you're not sure where that is.

## Things worth knowing

The anchor seed is 42 and everything downstream depends on it.

Tucker compression alone loses about 5% of the detail before the model is involved, so some softness is expected.

The condition frame is the first frame of the clip, and the model generates the 200 frames after it.

If you use fewer clips in step 6, say so when you report results. The anchor and normalization come from that data, and a small sample makes them a poor match for the checkpoint.
