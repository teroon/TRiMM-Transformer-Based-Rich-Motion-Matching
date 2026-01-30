# copy the ./emage_evaltools folder into your folder
from emage_evaltools.mertic import BC
import torch
import numpy as np
import glob
from emage_utils.motion_io import beat_format_load
from emage_utils.motion_rep_transfer import get_motion_rep_numpy
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_dataset(dataset_path, gt_path, audio_path="examples/audio"):
    """Evaluate a single dataset against ground truth - only BC metric"""
    # init BC evaluator for this dataset
    # BC metric implies 30 FPS alignment
    bc_evaluator = BC(download_path="./emage_evaltools/", sigma=0.3, order=7)
    
    all_motion_pred = glob.glob(os.path.join(dataset_path, "*.npz"))
    motion_gt_paths = glob.glob(os.path.join(gt_path, "*.npz"))
    
    if len(all_motion_pred) == 0:
        print(f"No prediction files found in {dataset_path}")
        return None
    
    if len(motion_gt_paths) == 0:
        print(f"No ground truth files found in {gt_path}")
        return None
        
    # Create a mapping between prediction files and ground truth files
    # Extract filenames without extensions for matching
    pred_files_map = {os.path.splitext(os.path.basename(path))[0]: path for path in all_motion_pred}
    gt_files_map = {os.path.splitext(os.path.basename(path))[0]: path for path in motion_gt_paths}
    
    # Find common files between predictions and ground truth
    common_files = set(pred_files_map.keys()).intersection(set(gt_files_map.keys()))
    
    if len(common_files) == 0:
        print(f"No matching prediction-groundtruth pairs found in {dataset_path}")
        return None
    
    print(f"Found {len(common_files)} matching files for evaluation in {os.path.basename(dataset_path)}")
    
    processed_count = 0
    # Process matched files
    for filename in sorted(common_files):
        try:
            motion_pred_path = pred_files_map[filename]
            motion_gt_path = gt_files_map[filename]
            
            motion_pred_data = beat_format_load(motion_pred_path)['poses']
            motion_gt_data = beat_format_load(motion_gt_path)['poses']
            
            # ==========================================================
            #  FPS Downsampling (120 FPS -> 30 FPS)
            #  Reason: BC metric and audio alignment expect 30 FPS.
            #  Original data is 120 FPS, so we slice with step 4 (120/30=4).
            # ==========================================================
            source_fps = 120
            target_fps = 30
            step = int(source_fps / target_fps) # step = 4
            
            # Downsample Prediction
            if motion_pred_data.shape[0] > 0:
                # Basic check: only downsample if it looks like high FPS (optional, but safe to force if you are sure)
                # Here we force it as requested.
                motion_pred_data = motion_pred_data[::step]

            # Downsample GT (if GT is also 120 FPS)
            # Usually GT is used for finding the file, but if you compare lengths later, it's good to sync.
            # We assume GT might also be 120 FPS if it came from the same source.
            if motion_gt_data.shape[0] > 0:
                 # Heuristic: if GT length is close to original Pred length (before slice), slice it too.
                 # Or just slice it if you know your GT folder is 120 FPS. 
                 # Assuming safe to slice if length allows:
                 if motion_gt_data.shape[0] > motion_pred_data.shape[0] * 2: 
                     motion_gt_data = motion_gt_data[::step]
            # ==========================================================

            # Check sequence length to avoid convolution errors
            if motion_pred_data.shape[0] < 10 or motion_gt_data.shape[0] < 10:
                print(f"Skipping {filename}: sequence too short ({motion_pred_data.shape[0]} frames)")
                continue
                
            t = motion_pred_data.shape[0]
            
            # BC requires position representation and audio
            audio_file_path = os.path.join(audio_path, f"{filename}.wav")
            if os.path.exists(audio_file_path):
                # Get position representation for BC calculation
                motion_position_pred = get_motion_rep_numpy(motion_pred_data, device=device)["position"]  # t*55*3
                motion_position_pred = motion_position_pred.reshape(t, -1)
                
                # ignore the start and end 2s (60 frames at 30fps), this is for beat dataset
                # Now that 't' is based on 30 FPS data, this logic holds true.
                
                # Safety check: ensure t is long enough for the crop
                if t <= 120:
                     print(f"Skipping {filename}: too short for BC crop ({t} frames)")
                     continue

                audio_beat = bc_evaluator.load_audio(audio_file_path, t_start=2 * 16000, t_end=int((t-60)/30*16000))
                motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=60, t_end=t-60, pose_fps=30, without_file=True)
                bc_evaluator.compute(audio_beat, motion_beat, length=t-120, pose_fps=30)
            else:
                print(f"Warning: Audio file {audio_file_path} not found, skipping BC calculation for {filename}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Successfully processed {processed_count}/{len(common_files)} files in {os.path.basename(dataset_path)}")
    
    if processed_count == 0:
        print(f"No files could be processed in {dataset_path}")
        return None
    
    metrics = {}
    
    # Add BC metric if audio files were processed
    if bc_evaluator.counter > 0:
        metrics["bc"] = bc_evaluator.avg()
    else:
        metrics["bc"] = "N/A (no audio files found)"
    
    return metrics

# Define the parent directory containing multiple dataset folders
datasets_parent_dir = "examples/test"  # Parent directory containing dataset folders
gt_dir = "examples/beatnpz"
audio_dir = "examples/audio"  # Audio directory containing all audio files

# Find all subdirectories in the parent directory that contain .npz files (potential datasets)
if not os.path.exists(datasets_parent_dir):
    print(f"Error: Dataset directory {datasets_parent_dir} does not exist.")
    dataset_dirs = []
else:
    all_items = os.listdir(datasets_parent_dir)
    dataset_dirs = []
    for item in all_items:
        item_path = os.path.join(datasets_parent_dir, item)
        if os.path.isdir(item_path):
            # Check if this directory contains .npz files
            npz_files = glob.glob(os.path.join(item_path, "*.npz"))
            if len(npz_files) > 0:
                dataset_dirs.append(item)

# Evaluate each dataset
for dataset_name in dataset_dirs:
    dataset_path = os.path.join(datasets_parent_dir, dataset_name)
    print(f"\nEvaluating dataset: {dataset_name}")
    metrics = evaluate_dataset(dataset_path, gt_dir, audio_dir)
    if metrics:
        print(f"{dataset_name} - BC: {metrics['bc']}")
    else:
        print(f"Failed to evaluate dataset: {dataset_name}")