# GPU Inference Troubleshooting - Issues Fixed

## Problems Identified

1. **No GPU processes showing in nvidia-smi**
   - Script wasn't loading data to GPU properly
   - Missing device verification logs

2. **No output in SLURM logs**
   - Inference script had incomplete code (missing segments_info formatting)
   - No checkpoint validation
   - Missing error handling

3. **Uncertainty about GPU usage**
   - No explicit logging of device selection
   - No CUDA availability checks

## Fixes Applied

### 1. Enhanced GPU Logging in `m2f_inference.py`
- Added comprehensive GPU detection at inference start
- Shows GPU name, memory, and CUDA availability
- Warns if running on CPU

### 2. Completed Missing Code in `m2f_inference.py`
- Fixed incomplete PNG saving code for COCO predictions
- Added proper segments_info formatting
- Added error handling and progress logging

### 3. Enhanced SLURM Job Script (`train_job.sh`)
- **Added PyTorch CUDA check** - Verifies PyTorch can see GPUs
- **Added checkpoint validation** - Auto-selects latest checkpoint if model_final.pth missing
- **Added python3 -u flag** - Unbuffered output for real-time logging
- **Reduced test size** - `--max-images 10` for quick testing
- **Added working directory check** - Ensures we're in correct path
- **Enhanced logging** - Clear section markers for debugging

### 4. Created GPU Test Script (`test_gpu.py`)
- Quick test to verify GPU availability
- Run before submitting jobs to catch environment issues

## How to Use

### Step 1: Test GPU Availability (Optional but Recommended)
```bash
cd /work/sramam2s/Mask2Former
conda activate robot_lab
python3 test_gpu.py
```

This will verify:
- ✓ PyTorch can see CUDA
- ✓ GPU operations work
- ✓ Correct environment activated

### Step 2: Submit Job
```bash
sbatch train_job.sh
```

### Step 3: Monitor Output
```bash
# Watch the job queue
squeue -u $USER

# Tail the output log (replace JOBID)
tail -f slurm_JOBID.out

# Check errors
tail -f slurm_JOBID.err
```

## What to Look For in Output

### ✓ Good Signs:
```
PYTORCH CUDA CHECK
========================================
PyTorch version: 2.x.x
CUDA available: True
GPU count: 1
Device name: NVIDIA A100-SXM4-80GB
```

```
MASK2FORMER + MAPANYTHING UNIFIED INFERENCE
Device: cuda
✓ CUDA Available: 1 GPU(s)
  Current GPU: 0 - NVIDIA A100-SXM4-80GB
  Memory: 80.0 GB
```

### ✗ Bad Signs:
```
CUDA available: False
```
or
```
⚠ WARNING: CUDA not available! Running on CPU
```

**If you see these:**
1. Check conda environment: `conda activate robot_lab`
2. Check PyTorch installation: `python3 -c "import torch; print(torch.__version__)"`
3. Verify GPU allocation in SLURM: `#SBATCH --gres=gpu:1`

## Expected Workflow

1. **Job starts** → Shows SLURM info, nvidia-smi output
2. **PyTorch CUDA check** → Confirms GPU visible to PyTorch
3. **Model loading** → Shows GPU device being used
4. **Inference loop** → Processes 10 images (for testing)
5. **Prediction saving** → Creates JSON + PNG files
6. **Evaluation** → Runs COCO Panoptic API (if predictions exist)

## Common Issues & Solutions

### Issue: "No running processes" in nvidia-smi
- **Cause**: Inference hasn't started yet, or model on CPU
- **Solution**: Check for "CUDA available: True" in logs

### Issue: Checkpoint not found
- **Solution**: Script now auto-selects latest checkpoint (model_*.pth)
- Check `./output_cluster/` has .pth files

### Issue: Job completes instantly with no output
- **Cause**: Python crash before printing
- **Solution**: Check slurm_JOBID.err file
- Run `python3 test_gpu.py` to verify environment

### Issue: Very slow inference
- **Cause**: Running on CPU instead of GPU
- **Solution**: Look for "⚠ WARNING: CUDA not available!" in logs
- Reinstall PyTorch with CUDA: `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`

## Files Modified

1. ✅ `m2f_inference.py` - Added GPU logging, fixed incomplete code
2. ✅ `train_job.sh` - Enhanced debugging, checkpoint validation, unbuffered output
3. ✅ `test_gpu.py` - New GPU test script

## Next Steps

1. Run `python3 test_gpu.py` to verify GPU setup
2. Submit job: `sbatch train_job.sh`
3. Monitor: `tail -f slurm_*.out`
4. If successful with 10 images, increase to full dataset (remove `--max-images 10`)
