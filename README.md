# Detection_CTA
# YOLOv8 with Data Migration and Catastrophic Forgetting Reduction

This repository contains a modified version of YOLOv8, focusing on enhancing parameter updates with data migration techniques and mitigating catastrophic forgetting. The key modifications are made in the training process and in the adaptation layers of the model. This project is designed to improve the model's ability to retain learned information across different tasks while maintaining high accuracy.

## Key Modifications

### 1. **Data Migration-Based Parameter Updates**
   - Implemented in the training function located in `train.py`.
   - Real-time parameter updates are applied to the model during training to facilitate more efficient learning without forgetting previously learned tasks.
   
   **Location**: `train.py`

### 2. **Reduction of Catastrophic Forgetting**
   - Introduced strategies to mitigate catastrophic forgetting during incremental learning, ensuring the model maintains performance on old tasks while learning new ones.

### 3. **Adapter Module Modifications**
   - Changes made in `nn.modules.py` (lines 232-390).
   - Added new parallel adapters to the network architecture to enhance the model's flexibility in handling different tasks.
   
   **Location**: `nn.modules.py`, lines 232-390

### 4. **Model Initialization Modifications**
   - Adjustments in model initialization are made in `nn.task.py` (lines 20-233).
   - KL divergence is calculated in the forward pass to measure the shift in data distribution across different tasks.

   **Location**: `nn.task.py`, lines 20-233

### 5. **KL Divergence for Data Shift Measurement**
   - The forward pass in `nn.task.py` includes the calculation of KL divergence, which helps in determining the extent of data distribution shift between tasks.
   - This feature allows the model to adjust its parameters more effectively by measuring how much the data has changed over time.

   **KL Divergence Code**:
   ```python
   def kl_divergence(self, current_features, past_features):
       current_features = current_features.view(current_features.size(0), -1)  # Flatten
       past_features = past_features.view(past_features.size(0), -1)  # Flatten
       p = torch.nn.functional.softmax(current_features, dim=-1)
       q = torch.nn.functional.softmax(past_features, dim=-1)
       kl = torch.sum(p * torch.log(p / (q + 1e-8)), dim=-1).mean()  # Add small epsilon to prevent NaN
       return kl
