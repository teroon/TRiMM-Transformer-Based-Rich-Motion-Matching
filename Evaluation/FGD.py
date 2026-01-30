# copy the ./emage_evaltools folder into your folder
from emage_evaltools.mertic import FGD,L1div,BC
import torch
import numpy as np
import glob
from emage_utils.motion_io import beat_format_load
from emage_utils import rotation_conversions as rc
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_dataset(dataset_path, gt_path):
    """Evaluate a single dataset against ground truth"""
    # init evaluators for this dataset
    fgd_evaluator = FGD(download_path="./emage_evaltools/")
    l1div_evaluator = L1div()
    
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
            
            # Check sequence length to avoid convolution errors
            if motion_pred_data.shape[0] < 10 or motion_gt_data.shape[0] < 10:
                print(f"Skipping {filename}: sequence too short ({motion_pred_data.shape[0]} frames)")
                continue
                
            # fgd requires rotation 6d representation
            motion_gt_a = torch.from_numpy(motion_gt_data).to(device).unsqueeze(0)
            motion_pred_a = torch.from_numpy(motion_pred_data).to(device).unsqueeze(0)
            t = motion_gt_a.shape[1]
            motion_gt = rc.axis_angle_to_rotation_6d(motion_gt_a.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
            t = motion_pred_a.shape[1]
            motion_pred = rc.axis_angle_to_rotation_6d(motion_pred_a.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
            
            # Additional check for minimum sequence length after reshaping
            if motion_gt.shape[1] < 10 or motion_pred.shape[1] < 10:
                print(f"Skipping {filename}: reshaped sequence too short")
                continue
            
            fgd_evaluator.update(motion_pred.float(), motion_gt.float())
            l1div_evaluator.compute(motion_pred_data)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Successfully processed {processed_count}/{len(common_files)} files in {os.path.basename(dataset_path)}")
    
    if processed_count == 0:
        print(f"No files could be processed in {dataset_path}")
        return None
    
    metrics = {}
    metrics["fgd"] = fgd_evaluator.compute()
    metrics["l1"] = l1div_evaluator.avg()
    return metrics

# Define the parent directory containing multiple dataset folders
datasets_parent_dir = "examples/test"  # Parent directory containing dataset folders
gt_dir = "examples/zeggsnpz"

# Find all subdirectories in the parent directory that contain .npz files (potential datasets)
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
    metrics = evaluate_dataset(dataset_path, gt_dir)
    if metrics:
        print(f"{dataset_name} - FGD: {metrics['fgd']:.4f}, L1: {metrics['l1']:.4f}")
    else:
        print(f"Failed to evaluate dataset: {dataset_name}")