#!/usr/bin/env python3
"""
Plot training loss curves from Detectron2 training logs.

Extracts iteration and loss values from SLURM output logs and creates
visualization plots for total loss and component losses over training.

Usage:
    python plot_training_loss.py slurm_882072.out
    python plot_training_loss.py slurm_882072.out --output loss_curves.png
"""

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_training_log(log_file):
    """
    Parse Detectron2 training log file to extract iteration and loss values.
    
    Args:
        log_file: Path to log file
        
    Returns:
        dict: Dictionary containing lists of iterations and various losses
    """
    data = {
        'iteration': [],
        'total_loss': [],
        'loss_ce': [],
        'loss_mask': [],
        'loss_dice': [],
        'lr': [],
        'eta': [],
    }
    
    # Pattern to match Detectron2 training logs
    # Example: [12/14 00:39:16 d2.utils.events]:  eta: 15:39:29  iter: 19  total_loss: 290.6  loss_ce: 24 ...
    pattern = r'iter:\s*(\d+)\s+total_loss:\s*([\d.]+)\s+loss_ce:\s*([\d.]+)\s+loss_mask:\s*([\d.]+)\s+loss_dice:\s*([\d.]+)'
    lr_pattern = r'lr:\s*([\d.e+-]+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match iteration and losses
            match = re.search(pattern, line)
            if match:
                iteration = int(match.group(1))
                total_loss = float(match.group(2))
                loss_ce = float(match.group(3))
                loss_mask = float(match.group(4))
                loss_dice = float(match.group(5))
                
                data['iteration'].append(iteration)
                data['total_loss'].append(total_loss)
                data['loss_ce'].append(loss_ce)
                data['loss_mask'].append(loss_mask)
                data['loss_dice'].append(loss_dice)
                
                # Extract learning rate
                lr_match = re.search(lr_pattern, line)
                if lr_match:
                    lr = float(lr_match.group(1))
                    data['lr'].append(lr)
                else:
                    data['lr'].append(np.nan)
    
    return data


def plot_loss_curves(data, output_path='loss_curves.png', show_components=True):
    """
    Create comprehensive loss curve plots.
    
    Args:
        data: Dictionary with iteration and loss data
        output_path: Path to save output figure
        show_components: Whether to plot component losses in addition to total loss
    """
    if not data['iteration']:
        print("No training data found in log file!")
        return
    
    iterations = np.array(data['iteration'])
    
    # Create figure with subplots
    if show_components:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Loss
        ax1 = axes[0, 0]
        ax1.plot(iterations, data['total_loss'], linewidth=2, color='#2E86AB', alpha=0.8)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Total Loss', fontsize=12)
        ax1.set_title('Total Loss over Training', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # Add smoothed trend line
        if len(iterations) > 10:
            window_size = max(len(iterations) // 20, 5)
            smoothed = np.convolve(data['total_loss'], np.ones(window_size)/window_size, mode='valid')
            smooth_iters = iterations[window_size-1:]
            ax1.plot(smooth_iters, smoothed, linewidth=2, color='#A23B72', 
                    label=f'Smoothed (window={window_size})', linestyle='--')
            ax1.legend()
        
        # Plot 2: Component Losses
        ax2 = axes[0, 1]
        ax2.plot(iterations, data['loss_ce'], linewidth=1.5, label='Classification (CE)', alpha=0.8)
        ax2.plot(iterations, data['loss_mask'], linewidth=1.5, label='Mask', alpha=0.8)
        ax2.plot(iterations, data['loss_dice'], linewidth=1.5, label='Dice', alpha=0.8)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss Value', fontsize=12)
        ax2.set_title('Component Losses', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        
        # Plot 3: Learning Rate Schedule
        ax3 = axes[1, 0]
        valid_lr_mask = ~np.isnan(data['lr'])
        if np.any(valid_lr_mask):
            valid_iters = iterations[valid_lr_mask]
            valid_lrs = np.array(data['lr'])[valid_lr_mask]
            ax3.plot(valid_iters, valid_lrs, linewidth=2, color='#F18F01', alpha=0.8)
            ax3.set_xlabel('Iteration', fontsize=12)
            ax3.set_ylabel('Learning Rate', fontsize=12)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(left=0)
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, 'No LR data available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_xlabel('Iteration', fontsize=12)
            ax3.set_ylabel('Learning Rate', fontsize=12)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        # Plot 4: Loss Statistics
        ax4 = axes[1, 1]
        recent_iterations = 100
        if len(iterations) > recent_iterations:
            recent_loss = data['total_loss'][-recent_iterations:]
            recent_iters = iterations[-recent_iterations:]
            
            ax4.plot(recent_iters, recent_loss, linewidth=2, color='#2E86AB', alpha=0.8)
            ax4.axhline(y=np.mean(recent_loss), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(recent_loss):.2f}')
            ax4.set_xlabel('Iteration', fontsize=12)
            ax4.set_ylabel('Total Loss', fontsize=12)
            ax4.set_title(f'Recent Loss (Last {recent_iterations} iterations)', 
                         fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, f'Need >{recent_iterations} iterations for recent view', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_xlabel('Iteration', fontsize=12)
            ax4.set_ylabel('Total Loss', fontsize=12)
            ax4.set_title(f'Recent Loss (Last {recent_iterations} iterations)', 
                         fontsize=14, fontweight='bold')
    else:
        # Simple plot: just total loss
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(iterations, data['total_loss'], linewidth=2, color='#2E86AB', alpha=0.8)
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Total Loss', fontsize=14)
        ax.set_title('Training Loss Curve', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        # Add smoothed trend line
        if len(iterations) > 10:
            window_size = max(len(iterations) // 20, 5)
            smoothed = np.convolve(data['total_loss'], np.ones(window_size)/window_size, mode='valid')
            smooth_iters = iterations[window_size-1:]
            ax.plot(smooth_iters, smoothed, linewidth=2, color='#A23B72', 
                   label=f'Smoothed (window={window_size})', linestyle='--')
            ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Loss curves saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total iterations logged: {len(iterations)}")
    print(f"Iteration range: {iterations[0]} - {iterations[-1]}")
    print(f"\nTotal Loss:")
    print(f"  Initial: {data['total_loss'][0]:.2f}")
    print(f"  Final:   {data['total_loss'][-1]:.2f}")
    print(f"  Min:     {min(data['total_loss']):.2f}")
    print(f"  Mean:    {np.mean(data['total_loss']):.2f}")
    print(f"  Reduction: {((data['total_loss'][0] - data['total_loss'][-1]) / data['total_loss'][0] * 100):.1f}%")
    
    if show_components and data['loss_ce']:
        print(f"\nComponent Losses (final):")
        print(f"  Classification: {data['loss_ce'][-1]:.3f}")
        print(f"  Mask:          {data['loss_mask'][-1]:.3f}")
        print(f"  Dice:          {data['loss_dice'][-1]:.3f}")
    
    if data['lr'] and not all(np.isnan(data['lr'])):
        valid_lrs = [lr for lr in data['lr'] if not np.isnan(lr)]
        if valid_lrs:
            print(f"\nLearning Rate:")
            print(f"  Initial: {valid_lrs[0]:.2e}")
            print(f"  Final:   {valid_lrs[-1]:.2e}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training loss curves from Detectron2 logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_training_loss.py slurm_882072.out
  python plot_training_loss.py slurm_882072.out --output my_loss_plot.png
  python plot_training_loss.py slurm_882072.out --simple
        """
    )
    parser.add_argument('log_file', type=str, help='Path to training log file')
    parser.add_argument('--output', '-o', type=str, default='loss_curves.png',
                       help='Output image path (default: loss_curves.png)')
    parser.add_argument('--simple', action='store_true',
                       help='Create simple plot with only total loss')
    
    args = parser.parse_args()
    
    # Check if log file exists
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}")
        return
    
    print(f"Reading training log: {args.log_file}")
    data = parse_training_log(args.log_file)
    
    if not data['iteration']:
        print("No training iterations found in log file!")
        print("Make sure the log file contains Detectron2 training output.")
        return
    
    print(f"Found {len(data['iteration'])} training iterations")
    
    # Create plots
    plot_loss_curves(data, output_path=args.output, show_components=not args.simple)
    
    print(f"Done! View the plot: {args.output}")


if __name__ == '__main__':
    main()
