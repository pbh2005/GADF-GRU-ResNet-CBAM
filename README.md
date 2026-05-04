# GADF-GRU-ResNet-CBAM
GADF-GRU-ResNet-CBAM model
# Dataset
In the dataset folder, there are four different working conditions (A, B, C, D) with their respective datasets. The sub-files contain the training sets, validation sets and test sets for each condition.
# Run 
Run dataset.py to convert raw 1D vibration signals into 2D Gramian Angular Difference Field (GADF) images, which will be saved for subsequent model training.

Run train.py to train the GADF-GRU-ResNet-CBAM model on the generated GADF images, and evaluate its performance on the test set.
