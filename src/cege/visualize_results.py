"""
Visualize CEGE training results
"""

import matplotlib.pyplot as plt
import numpy as np

# Training data from the results
epochs = list(range(1, 11))

train_loss = [1.4158, 1.1097, 1.0695, 1.0386, 1.0319, 0.9944, 1.0092, 0.9610, 0.9656, 0.9761]
train_acc = [45.77, 55.51, 57.28, 56.97, 57.90, 58.77, 57.92, 59.35, 58.32, 58.63]
train_f1 = [46.06, 55.35, 56.86, 56.72, 57.50, 58.34, 57.48, 58.89, 57.74, 58.29]

valid_loss = [1.0254, 1.0334, 1.0069, 0.9793, 0.9768, 1.0214, 0.9792, 0.9222, 0.9528, 0.9367]
valid_acc = [60.07, 60.07, 60.07, 61.77, 60.24, 59.90, 60.07, 62.80, 60.41, 60.58]
valid_f1 = [56.91, 56.91, 56.91, 60.64, 57.12, 56.67, 56.91, 59.36, 58.63, 57.19]

test_loss = [1.2602, 1.1854, 1.1795, 1.1737, 1.0978, 1.1598, 1.0994, 1.0922, 1.0898, 1.0666]
test_acc = [45.81, 51.11, 47.16, 48.46, 55.43, 47.66, 52.59, 46.86, 50.86, 52.22]
test_f1 = [43.31, 50.80, 45.88, 47.97, 56.08, 47.50, 50.78, 45.42, 51.09, 53.40]

time_per_epoch = [141.41, 150.60, 140.53, 153.27, 148.05, 167.81, 164.34, 146.36, 130.13, 136.26]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('CEGE Training Results on IEMOCAP', fontsize=16, fontweight='bold')

# Plot 1: Loss
axes[0, 0].plot(epochs, train_loss, 'o-', label='Train', linewidth=2, markersize=6)
axes[0, 0].plot(epochs, valid_loss, 's-', label='Validation', linewidth=2, markersize=6)
axes[0, 0].plot(epochs, test_loss, '^-', label='Test', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(epochs)

# Plot 2: Accuracy
axes[0, 1].plot(epochs, train_acc, 'o-', label='Train', linewidth=2, markersize=6)
axes[0, 1].plot(epochs, valid_acc, 's-', label='Validation', linewidth=2, markersize=6)
axes[0, 1].plot(epochs, test_acc, '^-', label='Test', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 1].set_title('Accuracy Over Time', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(epochs)
axes[0, 1].axhline(y=max(test_acc), color='r', linestyle='--', alpha=0.5, label=f'Best Test: {max(test_acc):.2f}%')

# Plot 3: F1-Score
axes[1, 0].plot(epochs, train_f1, 'o-', label='Train', linewidth=2, markersize=6)
axes[1, 0].plot(epochs, valid_f1, 's-', label='Validation', linewidth=2, markersize=6)
axes[1, 0].plot(epochs, test_f1, '^-', label='Test', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('F1-Score (%)', fontsize=12)
axes[1, 0].set_title('F1-Score Over Time', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(epochs)
axes[1, 0].axhline(y=max(valid_f1), color='g', linestyle='--', alpha=0.5)
axes[1, 0].axhline(y=max(test_f1), color='r', linestyle='--', alpha=0.5)
axes[1, 0].text(1, max(valid_f1) + 1, f'Best Valid: {max(valid_f1):.2f}%', fontsize=10, color='g')
axes[1, 0].text(1, max(test_f1) - 2, f'Best Test: {max(test_f1):.2f}%', fontsize=10, color='r')

# Plot 4: Training Time
axes[1, 1].bar(epochs, time_per_epoch, color='steelblue', alpha=0.7)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Time (seconds)', fontsize=12)
axes[1, 1].set_title('Training Time per Epoch', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_xticks(epochs)
axes[1, 1].axhline(y=np.mean(time_per_epoch), color='r', linestyle='--', alpha=0.7, 
                   label=f'Avg: {np.mean(time_per_epoch):.1f}s')
axes[1, 1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('cege_training_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cege_training_results.png")

# Create performance comparison plot
fig2, ax = plt.subplots(figsize=(10, 6))

categories = ['Train', 'Validation', 'Test']
final_f1 = [train_f1[-1], valid_f1[-1], test_f1[-1]]
best_f1 = [max(train_f1), max(valid_f1), max(test_f1)]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, final_f1, width, label='Final (Epoch 10)', alpha=0.8)
bars2 = ax.bar(x + width/2, best_f1, width, label='Best Overall', alpha=0.8)

ax.set_ylabel('F1-Score (%)', fontsize=12)
ax.set_title('CEGE Performance Summary', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('cege_performance_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cege_performance_summary.png")

# Print summary statistics
print("\n" + "="*60)
print("CEGE TRAINING SUMMARY")
print("="*60)
print(f"\nBest Performance:")
print(f"  Validation F1: {max(valid_f1):.2f}% (Epoch {valid_f1.index(max(valid_f1)) + 1})")
print(f"  Test F1:       {max(test_f1):.2f}% (Epoch {test_f1.index(max(test_f1)) + 1})")
print(f"  Test Accuracy: {max(test_acc):.2f}% (Epoch {test_acc.index(max(test_acc)) + 1})")

print(f"\nFinal Performance (Epoch 10):")
print(f"  Train F1:      {train_f1[-1]:.2f}%")
print(f"  Validation F1: {valid_f1[-1]:.2f}%")
print(f"  Test F1:       {test_f1[-1]:.2f}%")

print(f"\nImprovement:")
print(f"  Train F1:      {train_f1[0]:.2f}% → {train_f1[-1]:.2f}% (+{train_f1[-1] - train_f1[0]:.2f}%)")
print(f"  Test F1:       {test_f1[0]:.2f}% → {max(test_f1):.2f}% (+{max(test_f1) - test_f1[0]:.2f}%)")

print(f"\nTraining Efficiency:")
print(f"  Total time:    {sum(time_per_epoch):.1f}s ({sum(time_per_epoch)/60:.1f} min)")
print(f"  Avg per epoch: {np.mean(time_per_epoch):.1f}s")
print(f"  Fastest epoch: {min(time_per_epoch):.1f}s (Epoch {time_per_epoch.index(min(time_per_epoch)) + 1})")
print(f"  Slowest epoch: {max(time_per_epoch):.1f}s (Epoch {time_per_epoch.index(max(time_per_epoch)) + 1})")

print("\n" + "="*60)
print("✓ Visualization complete!")
print("="*60)

plt.show()
