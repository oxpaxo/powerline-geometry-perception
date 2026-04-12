## Repository Name  
  
`powerline-segformer-aux`  
  
## Short English Description  
  
SegFormer-based powerline perception with centerline, orientation, and distance-field auxiliary supervision.  
  
---  
  
## Task Goal  
  
Start a **small-scale debug training run** for the current project to verify that:  
  
1. the SegFormer backbone builds correctly,  
2. the centerline + orientation + distance auxiliary branch all run together,  
3. the training loop is stable for a short run,  
4. logs and checkpoints are written correctly.  
  
This is **not** a full training run.  
This is a **500-iteration debug run** on a small subset (about 10–20 images).  
  
---  
  
## Which Config to Use  
  
Use:  
  
`configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py`  
  
Do **not** use the full config first.  
  
Reason:  
- this debug config already inherits the distance-aux version,  
- uses `splits/debug_train.txt` ,  
- uses small worker settings,  
- and is already configured for **500 iterations**.  
  
---  
  
## Before Training  
  
### 1. Check the workspace root  
  
Assume the workspace root is:  
  
`mmsegmentation-main`  
  
All commands below should be run from this root directory.  
  
### 2. Check the debug split  
  
Open:  
  
`projects/powerline_v1/datasets/TTPLA/splits/debug_train.txt`  
  
Make sure it contains only the **10–20 training samples** intended for this debug run.  
  
If it contains too many entries, reduce it to the desired subset first.  
  
### 3. Confirm the target config exists  
  
Confirm the file exists:  
  
`configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py`  
  
---  
  
## Training Command  
  
Run this command from the workspace root:  
  
```bash  
python tools/train.py configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py

### Windows PowerShell version

cd E:\DeepLearning\MyProject\mmsegmentation-main  
python tools\train.py configs\powerline_v1\powerline_v1_segformer_b1_aux_distance_debug.py

---

## What This Run Should Do

This run should:

- build the SegFormer + distance auxiliary model,
- load the debug subset from `debug_train.txt`,
- train for **500 iterations**,
- write logs to the debug work directory,
- save checkpoints if configured.

---

## Expected Output Directory

The debug run should write outputs under:

`work_dirs/powerline_v1_segformer_b1_aux_distance_debug`

Check this directory for:

- logs,
- checkpoints,
- training progress files.

---

## What to Watch in the Logs

During training, check whether the logs contain:

- centerline-related loss
- orientation-related loss
- `loss_distance`

Focus on these questions:

1. Does training start without build/import errors?
2. Does `loss_distance` appear in the log?
3. Are all losses finite (not NaN / Inf)?
4. Does the run complete 500 iterations successfully?

---

## After Training

After the 500-iteration run finishes, report back with:

1. whether the run finished successfully,
2. the last 20–30 log lines,
3. whether `loss_distance` appeared,
4. whether any warnings/errors occurred,
5. the output path used by the run.

---

## Optional Next Step After Successful Training

If the 500-iteration run succeeds, run the visualization script:

python tools/debug/inspect_aux_targets_and_preds.py --config configs/powerline_v1/powerline_v1_segformer_b1_aux_distance_debug.py --num-samples 4 --out-dir work_dirs/debug_vis

Then report whether:

- GT center looks correct,
- GT distance map looks reasonable,
- predicted distance map is non-empty and stable.

---

## Important Constraints

- Do **not** switch to the full training config yet.
- Do **not** enable tower training.
- Do **not** modify the config unless a blocking runtime error occurs.
- If an error occurs, stop and report the exact traceback.

---

## Codex Execution Instruction

Please do the following in order:

1. Verify the workspace root and target config file.
2. Verify `debug_train.txt` exists and is being used.
3. Start training with the debug config.
4. Let the run continue until it completes 500 iterations or fails.
5. Return:
    - the exact command used,
    - whether the run succeeded,
    - key log excerpts,
    - any error traceback if failed,
    - the work directory path,
    - whether `loss_distance` appeared in the logs.
    
```
