import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data extracted from your classification report
data = {
    'Class': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'Precision': [0.494, 0.464, 0.323, 0.324, 0.375, 0.391, 0.403, 0.438, 0.533, 0.412],
    'Recall':    [0.470, 0.477, 0.273, 0.267, 0.355, 0.362, 0.485, 0.402, 0.589, 0.521],
    'F1-Score':  [0.481, 0.471, 0.296, 0.293, 0.365, 0.376, 0.441, 0.419, 0.560, 0.460]
}

df = pd.DataFrame(data)

# Sort by F1 Score for better readability
df = df.sort_values('F1-Score', ascending=True)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(df['Class']))
width = 0.25

rects1 = ax.bar(x - width, df['Precision'], width, label='Precision', color='#88CCEE')
rects2 = ax.bar(x, df['Recall'], width, label='Recall', color='#DDCC77')
rects3 = ax.bar(x + width, df['F1-Score'], width, label='F1-Score', color='#CC6677')

# Add labels and title
ax.set_ylabel('Score')
ax.set_title('Per-Class Performance Metrics (Sorted by F1)')
ax.set_xticks(x)
ax.set_xticklabels(df['Class'])
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_ylim(0, 0.7)

# Add F1 text on top
for i, v in enumerate(df['F1-Score']):
    ax.text(i + width, v + 0.01, f"{v:.2f}", ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('class_metrics.png')
plt.show()