# LIBERO任务过滤训练指南

## 概述
现在你可以针对LIBERO数据集中的特定任务进行训练，而不是使用整个数据集。

## LIBERO数据集任务列表

LIBERO数据集包含40个不同的任务（task_index 0-39）：

```
task_index 0: put the white mug on the left plate and put the yellow and white mug on the right plate
task_index 1: put the white mug on the plate and put the chocolate pudding to the right of the plate
task_index 2: put the yellow and white mug in the microwave and close it
task_index 3: turn on the stove and put the moka pot on it
task_index 4: put both the alphabet soup and the cream cheese box in the basket
task_index 5: put both the alphabet soup and the tomato sauce in the basket
task_index 6: put both moka pots on the stove
task_index 7: put both the cream cheese box and the butter in the basket
task_index 8: put the black bowl in the bottom drawer of the cabinet and close it
task_index 9: pick up the book and place it in the back compartment of the caddy
task_index 10: put the bowl on the plate
task_index 11: put the wine bottle on the rack
task_index 12: open the top drawer and put the bowl inside
task_index 13: put the cream cheese in the bowl
task_index 14: put the wine bottle on top of the cabinet
task_index 15: push the plate to the front of the stove
task_index 16: turn on the stove
task_index 17: put the bowl on the stove
task_index 18: put the bowl on top of the cabinet
task_index 19: open the middle drawer of the cabinet
task_index 20: pick up the orange juice and place it in the basket
task_index 21: pick up the ketchup and place it in the basket
task_index 22: pick up the cream cheese and place it in the basket
task_index 23: pick up the bbq sauce and place it in the basket
task_index 24: pick up the alphabet soup and place it in the basket
task_index 25: pick up the milk and place it in the basket
task_index 26: pick up the salad dressing and place it in the basket
task_index 27: pick up the butter and place it in the basket
task_index 28: pick up the tomato sauce and place it in the basket
task_index 29: pick up the chocolate pudding and place it in the basket
task_index 30: pick up the black bowl next to the cookie box and place it on the plate
task_index 31: pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
task_index 32: pick up the black bowl on the ramekin and place it on the plate
task_index 33: pick up the black bowl on the stove and place it on the plate
task_index 34: pick up the black bowl between the plate and the ramekin and place it on the plate
task_index 35: pick up the black bowl on the cookie box and place it on the plate
task_index 36: pick up the black bowl next to the plate and place it on the plate
task_index 37: pick up the black bowl next to the ramekin and place it on the plate
task_index 38: pick up the black bowl from table center and place it on the plate
task_index 39: pick up the black bowl on the wooden cabinet and place it on the plate
```

完整列表可以查看：`~/.cache/huggingface/lerobot/physical-intelligence/libero/meta/tasks.jsonl`

## 使用方法

### 方法1：使用预定义的配置

我已经创建了一个新的配置 `pi05_libero_single_task`，你可以直接使用：

1. **修改配置文件**，编辑 `src/openpi/training/config.py`，找到 `pi05_libero_single_task` 配置，修改 `task_indices_filter` 参数：

```python
task_indices_filter=[0],  # 只训练task 0
# 或者
task_indices_filter=[0, 1, 2],  # 训练task 0, 1, 2
# 或者
task_indices_filter=[10],  # 只训练task 10 (put the bowl on the plate)
```

2. **计算归一化统计数据**（如果还没做）：
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero_single_task
```

3. **开始训练**：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_single_task --exp-name=my_single_task_experiment --overwrite
```

### 方法2：创建自定义配置

你也可以在 `config.py` 中创建自己的配置：

```python
TrainConfig(
    name="my_custom_libero_config",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(
            prompt_from_task=True,
            task_indices_filter=[10, 11, 12],  # 选择你想要的任务
        ),
        extra_delta_transform=False,
    ),
    batch_size=256,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=10_000,
        peak_lr=5e-5,
        decay_steps=1_000_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    pytorch_weight_path="/path/to/your/pytorch_weight_path",
    num_train_steps=30_000,
)
```

## 注意事项

1. **数据量**：过滤后的数据集会比原始数据集小很多，训练时间会相应减少
2. **归一化统计**：每次修改 `task_indices_filter` 后，建议重新计算归一化统计数据
3. **批次大小**：如果数据量很小，可能需要减小 `batch_size`
4. **训练步数**：数据量减少后，可能需要调整 `num_train_steps`

## 示例

训练单个任务（task 10: "put the bowl on the plate"）：

```bash
# 1. 修改config.py中的task_indices_filter=[10]
# 2. 计算归一化统计
uv run scripts/compute_norm_stats.py --config-name pi05_libero_single_task

# 3. 开始训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_single_task --exp-name=bowl_on_plate --overwrite
```

训练多个相关任务（tasks 20-29: 所有"pick up X and place it in the basket"任务）：

```bash
# 修改task_indices_filter=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# 然后运行上述命令
```
