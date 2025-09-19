# MNIST_CNN_below_25k
A neural network trained for classifying MNIST data with less than 25k parameters

## 🛠️ Modifications & Improvements

Several experiments and changes were made during the training process:

* **Learning Rate Fix**
  Changed learning rate from `10.01` → `0.01`. The previous high value caused divergence.

* **Loss Reduction**
  Switched `nn.CrossEntropyLoss(reduction='sum')` → `nn.CrossEntropyLoss(reduction='mean')`.
  The `'sum'` option caused exploding gradients, while `'mean'` stabilized training.

* **Neural Network Architecture**

  * Initially used max pooling and a `1x1` conv at the end with 2 layers.
  * Revised design: moved max pooling between layers instead of only at the end (since pooling at the very end is less useful).
  * Accuracy stayed more or less the same, but the structure became more logical.

* **Batch Size Adjustment**
  Reduced batch size from `512` → `32`, which improved convergence stability and generalization. Test accuracy improved to \~95%.

* **Seeding for Reproducibility**
  Added seeds (`torch`, `numpy`, `random`) at the start to ensure identical loss/accuracy for each run. Especially important since we limited runs to **1 epoch**.

---

## 📊 Results

* Stable training with `lr=0.01` and `reduction='mean'`.
* Final test accuracy: **\~95%** (after architecture + batch size tweaks).
* Training and evaluation are deterministic due to fixed seeds.

## 📌 Key Learnings

* **Loss scaling matters**: Always normalize per batch (use `'mean'`).
* **LR tuning**: Too high → divergence, too low → slow convergence.
* **Pooling strategy**: Pooling at intermediate layers is more effective than only at the end.
* **Batch size trade-off**: Smaller batches add noise to gradients, which helped generalization on MNIST.
* **Reproducibility**: Fixing seeds is essential when running very few epochs.

---

✨ With these tweaks, the notebook demonstrates how **hyperparameter tuning, architectural adjustments, and reproducibility practices** can drastically improve model performance and training stability.
