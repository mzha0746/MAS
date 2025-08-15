#!/bin/bash

# 快速测试脚本
# 简化版本，用于快速验证
# 使用顺序采样（前n个数据）而不是随机采样

echo "开始快速测试（使用顺序采样）..."

# 设置环境变量
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_API_KEY=""

# 测试配置
MODEL_TYPE="ollama_llama3.2:3b"
ATTACKER_STRATEGY="misinformation"
MSMARCO_PATH="/work/G-safeguard/MA/datasets/msmarco.json"

# 快速测试参数（减少组合数量）
declare -a AGENT_COUNTS=(3 4)
declare -a GRAPH_COUNTS=(2 3)
declare -a DIALOGUE_TURNS=(1 2)

# 创建输出目录
OUTPUT_DIR="/work/G-safeguard/NewMA/output"
mkdir -p $OUTPUT_DIR

# 记录测试开始时间
START_TIME=$(date)
echo "测试开始时间: $START_TIME"
echo "使用顺序采样：每次取前3个MSMarco样本"
echo ""

# 计数器
TOTAL_TESTS=0
SUCCESS_TESTS=0
FAILED_TESTS=0

# 快速测试循环
for agents in "${AGENT_COUNTS[@]}"; do
    for graphs in "${GRAPH_COUNTS[@]}"; do
        for turns in "${DIALOGUE_TURNS[@]}"; do
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            
            echo "=========================================="
            echo "快速测试 $TOTAL_TESTS: agents=$agents, graphs=$graphs, turns=$turns"
            echo "=========================================="
            
            # 运行测试（使用顺序采样）
            python -m NewMA.graph_generator \
                --num_agents $agents \
                --num_graphs $graphs \
                --num_dialogue_turns $turns \
                --model_type $MODEL_TYPE \
                --attacker_strategy $ATTACKER_STRATEGY \
                --msmarco_path $MSMARCO_PATH \
                --use_msmarco \
                --msmarco_samples_per_config 3 \
                --verbose
            
            # 检查退出状态
            if [ $? -eq 0 ]; then
                echo "✓ 测试成功: agents=$agents, graphs=$graphs, turns=$turns"
                SUCCESS_TESTS=$((SUCCESS_TESTS + 1))
            else
                echo "✗ 测试失败: agents=$agents, graphs=$graphs, turns=$turns"
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi
            
            echo ""
        done
    done
done

# 记录测试结束时间
END_TIME=$(date)
echo "=========================================="
echo "快速测试完成"
echo "=========================================="
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "总测试数: $TOTAL_TESTS"
echo "成功测试: $SUCCESS_TESTS"
echo "失败测试: $FAILED_TESTS"
echo "成功率: $((SUCCESS_TESTS * 100 / TOTAL_TESTS))%"
echo "=========================================="
echo ""
echo "注意：所有测试都使用了顺序采样（前3个MSMarco样本），"
echo "确保测试结果的可重现性和一致性。" 