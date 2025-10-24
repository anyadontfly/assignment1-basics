import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt


def read_loss_csv(csv_path):
    """
    Read CSV file and return data as dictionary of lists.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        Dictionary with column names as keys and lists of values
    """
    data = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Initialize lists for each column
        for row in reader:
            for key in row.keys():
                if key not in data:
                    data[key] = []
            break
        
        # Reset file pointer and skip header
        f.seek(0)
        next(reader)
        
        # Read all rows
        for row in reader:
            for key, value in row.items():
                if value == '':  # Handle empty values (e.g., validation loss on non-eval steps)
                    data[key].append(None)
                else:
                    try:
                        data[key].append(float(value))
                    except ValueError:
                        data[key].append(value)
    
    return data

def main(args):
    """
    Main function to plot learning curves.
    
    Args:
        args: Parsed command-line arguments
    """
    csv_path = args.csv
    output_path = args.output
    title = args.title
    show = not args.no_show
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    data = read_loss_csv(csv_path)
    
    # Validate required columns
    required_cols = ['step', 'train_loss']
    missing_cols = [col for col in required_cols if col not in data]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if we have validation loss and learning rate
    has_valid_loss = 'valid_loss' in data
    has_learning_rate = 'learning_rate' in data
    
    # Filter out rows with valid loss (not None)
    if has_valid_loss:
        valid_steps = [s for s, v in zip(data['step'], data['valid_loss']) if v is not None]
        valid_losses = [v for v in data['valid_loss'] if v is not None]
    
    # Create figure with subplots
    num_plots = 1 + (1 if has_learning_rate else 0)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))
    
    # Make axes iterable even if there's only one subplot
    if num_plots == 1:
        axes = [axes]
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0]
    ax1.plot(data['step'], data['train_loss'], label='Training Loss', linewidth=1.5, alpha=0.8)
    
    if has_valid_loss and len(valid_losses) > 0:
        ax1.plot(valid_steps, valid_losses, 
                label='Validation Loss', linewidth=2, marker='o', markersize=4, alpha=0.8)
    
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(title if title else 'Learning Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    final_train_loss = data['train_loss'][-1]
    min_train_loss = min(data['train_loss'])
    stats_text = f'Final Train Loss: {final_train_loss:.4f}\nMin Train Loss: {min_train_loss:.4f}'
    
    if has_valid_loss and len(valid_losses) > 0:
        final_valid_loss = valid_losses[-1]
        min_valid_loss = min(valid_losses)
        stats_text += f'\nFinal Valid Loss: {final_valid_loss:.4f}\nMin Valid Loss: {min_valid_loss:.4f}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Learning Rate (if available)
    if has_learning_rate:
        ax2 = axes[1]
        ax2.plot(data['step'], data['learning_rate'], linewidth=1.5, color='green', alpha=0.8)
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot learning curves from training loss CSV logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot and display learning curve
  python scripts/plot_learning_curve.py --csv log/training_loss.csv
  
  # Save plot to file without displaying
  python scripts/plot_learning_curve.py --csv log/training_loss.csv --output plots/learning_curve.png --no-show
  
  # Plot with custom title
  python scripts/plot_learning_curve.py --csv log/training_loss.csv --title "Transformer Training - 512 dim"
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to the CSV file containing training logs'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the plot image (e.g., plots/learning_curve.png)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom title for the plot'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot interactively (useful for saving only)'
    )
    
    args = parser.parse_args()
    main(args)
