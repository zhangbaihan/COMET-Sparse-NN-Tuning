import sys
sys.dont_write_bytecode = True
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis and Plotting")
    
    parser.add_argument("--train_log", type=str, help="Path to training_logs.pkl")
    parser.add_argument("--eval_res", type=str, help="Path to evaluation predictions .pkl")
    parser.add_argument("--output_dir", type=str, default="analysis_plots")
    
    return parser.parse_args()

def plot_training_metrics(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'acc_curve.png'))
    plt.close()
    
    # Plot TopK
    if 'top_k' in history:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['top_k'], label='Top K (Sparsity)')
        plt.title('Top K Sparsity Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Top K (Fraction Kept)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'topk_curve.png'))
        plt.close()

def analyze_predictions(results, output_dir):
    y_true = results['y_true']
    y_pred = results['y_pred']
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Per-class Metrics (Precision, Recall, F1)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract F1 scores per class
    classes = [k for k in report.keys() if k.isdigit()] # Assuming numeric class labels
    if not classes:
        # Try finding '0', '1' etc strings or just take first N keys if they are class names
        classes = [str(i) for i in range(cm.shape[0])]
        
    f1_scores = [report[c]['f1-score'] for c in classes if c in report]
    
    plt.figure(figsize=(12, 6))
    plt.bar(classes, f1_scores)
    plt.title('Per-Class F1 Score')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, 'per_class_f1.png'))
    plt.close()
    
    print("Analysis completed. Metrics summary:")
    print(classification_report(y_true, y_pred))

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.train_log:
        print(f"Loading training log from {args.train_log}...")
        with open(args.train_log, 'rb') as f:
            history = pickle.load(f)
        plot_training_metrics(history, args.output_dir)
        
    if args.eval_res:
        print(f"Loading evaluation results from {args.eval_res}...")
        with open(args.eval_res, 'rb') as f:
            results = pickle.load(f)
        analyze_predictions(results, args.output_dir)

if __name__ == "__main__":
    main()

