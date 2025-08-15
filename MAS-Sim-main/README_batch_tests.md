# 批量测试脚本使用说明

## 脚本文件

1. **`batch_test.sh`** - 完整批量测试脚本
   - 测试参数组合：agents(3,4,5) × graphs(2,3,5) × turns(1,2,3)
   - 总共 27 个测试组合
   - 适合完整测试

2. **`quick_test.sh`** - 快速测试脚本
   - 测试参数组合：agents(3,4) × graphs(2,3) × turns(1,2)
   - 总共 8 个测试组合
   - 适合快速验证

## 数据采样方式

### 顺序采样（新功能）
- **默认行为**：按顺序取前n个MSMarco样本
- **参数**：`--msmarco_samples_per_config 3`（取前3个样本）
- **特殊值**：`--msmarco_samples_per_config -1`（取所有样本）
- **优势**：确保测试结果的可重现性和一致性

### 随机采样（旧功能）
- **参数**：`--msmarco_samples_per_config 3`（随机取3个样本）
- **问题**：每次运行结果可能不同

## 使用方法

### 运行完整批量测试
```bash
cd /work/G-safeguard
./NewMA/batch_test.sh
```

### 运行快速测试
```bash
cd /work/G-safeguard
./NewMA/quick_test.sh
```

### 使用顺序采样的示例
```bash
# 取前3个样本
python -m NewMA.graph_generator \
    --num_agents 3 \
    --num_graphs 2 \
    --num_dialogue_turns 1 \
    --model_type ollama_llama3.2:3b \
    --attacker_strategy misinformation \
    --msmarco_path /work/G-safeguard/MA/datasets/msmarco.json \
    --use_msmarco \
    --msmarco_samples_per_config 3 \
    --verbose

# 取所有样本
python -m NewMA.graph_generator \
    --num_agents 3 \
    --num_graphs 2 \
    --num_dialogue_turns 1 \
    --model_type ollama_llama3.2:3b \
    --attacker_strategy misinformation \
    --msmarco_path /work/G-safeguard/MA/datasets/msmarco.json \
    --use_msmarco \
    --msmarco_samples_per_config -1 \
    --verbose
```

## 配置说明

### 环境变量
- `DEEPSEEK_BASE_URL`: DeepSeek API 基础URL
- `DEEPSEEK_API_KEY`: DeepSeek API 密钥

### 测试参数
- `MODEL_TYPE`: 模型类型 (ollama_llama3.2:3b)
- `ATTACKER_STRATEGY`: 攻击策略 (misinformation)
- `MSMARCO_PATH`: MSMarco 数据集路径
- `--use_msmarco`: 启用MSMarco数据集
- `--msmarco_samples_per_config`: 每个配置使用的样本数量

### 输出目录
- 所有测试结果保存在 `/work/G-safeguard/NewMA/output/`
- 每个测试会创建独立的文件夹
- 文件夹命名格式：`{graph名称}_{model_type}_{attack_strategy}_{query内容}_{timestamp}_num_agents_{num_agents}_num_dialogue_turns_{num_dialogue_turns}`

## 测试结果

脚本会显示：
- 每个测试的执行状态
- 成功/失败统计
- 总体成功率
- 开始和结束时间
- 进度条显示处理进度

## 进度条功能

- 显示总体配置处理进度
- 实时更新处理状态
- 显示剩余时间估计

## 注意事项

1. 确保有足够的磁盘空间存储测试结果
2. 测试可能需要较长时间，建议在后台运行
3. 如果某个测试失败，脚本会继续执行下一个测试
4. 所有测试完成后会显示总体统计信息
5. **顺序采样确保测试结果的可重现性**

## 后台运行

如果需要后台运行，可以使用：
```bash
nohup ./NewMA/quick_test.sh > quick_test.log 2>&1 &
```

这样可以在后台运行并将输出保存到 `quick_test.log` 文件中。

## 数据一致性

使用顺序采样可以确保：
- 每次运行使用相同的数据样本
- 测试结果可以准确比较
- 避免随机性对结果的影响
- 便于调试和复现问题 