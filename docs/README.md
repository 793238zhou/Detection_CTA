# Detection_CTA
# YOLOv8 with Data Migration and Catastrophic Forgetting Reduction

代码在master分支里面



该仓库包含了修改版的 YOLOv8，重点提升了通过数据迁移技术进行的参数更新，并减轻了灾难性遗忘问题。主要的修改体现在训练过程和模型的适配器层。这个项目旨在提高模型在不同任务间保持高精度的同时，也能够保留已学习的信息。

## 主要修改

### 1. **基于数据迁移的参数冻结与更新**
   - 在 `train.py` 文件中的训练函数中实现。
   - 在训练过程中应用实时的参数更新，以便在不忘记先前学习任务的情况下提高学习效率。
   - 在特定情况下，对于适配器以外的模块进行冻结，在保持原有模型参数的条件下，更新适配器模块的参数。
   **位置**: `train.py`

### 2. **灾难性遗忘的减少**
   - 引入策略以缓解增量学习中的灾难性遗忘，确保模型在学习新任务时保持对旧任务的良好表现。

### 3. **适配器模块的修改**
   - 修改内容位于 `nn.modules.py`。
   - 在网络架构中增加了新的并行适配器，以增强模型在处理不同任务时的灵活性。

   **位置**: `nn.modules.py`，第232-390行

### 4. **模型初始化的修改**
   - 在 `nn.task.py`中进行了模型初始化的调整。
   - 在前向传播过程中增加了KL散度计算，用于衡量不同数据集间的数据分布偏移（有待更新，关于如何进行特征值的提取，能更好的实现模型的环境适应）。

   **位置**: `nn.task.py`，第20-233行

### 5. **用于衡量数据分布偏移的KL散度**
   - `nn.task.py`中的前向传播过程包括KL散度计算，这有助于确定任务间数据分布的变化程度。
   - 这个功能使得模型能够通过测量数据随时间的变化来更有效地调整其参数。

### 6. **前向传播的修改**
   - 在 `yolo.engine.trainer`中增加了前向传播的参数。
   - 在前向传播过程中增加了KL散度计算，用于衡量不同任务间的数据分布偏移。

   **位置**: `yolo.engine.trainer`，第320-327行
### 7. **模型初始化的修改**
   - 在 `yolo.v8.detect.train.py`中进行了模型初始化的调整。
   - 修改了get_model函数的使用方法。

   **位置**: `yolo.v8.detect.train.py`，第61-66行


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
