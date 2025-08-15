#!/bin/bash

# 批量测试脚本
# 基于 launch.json 配置

echo "开始批量测试..."

# 设置环境变量
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_API_KEY=""

# 测试配置
MODEL_TYPE="ollama_smollm2:135m"
ATTACKER_STRATEGY="misinformation"
MSMARCO_PATH="/work/G-safeguard/MA/datasets/msmarco.json"

# 测试参数组合
declare -a AGENT_COUNTS=(3 4 5 6 7 8 9 10)
declare -a GRAPH_COUNTS=(5)
declare -a DIALOGUE_TURNS=(3 4 5 6 7 8 9 10)

# 创建输出目录
OUTPUT_DIR="/work/G-safeguard/NewMA/output"
mkdir -p $OUTPUT_DIR

# 记录测试开始时间
START_TIME=$(date)
echo "测试开始时间: $START_TIME"

# 计数器
TOTAL_TESTS=0
SUCCESS_TESTS=0
FAILED_TESTS=0

# 计算总测试数
TOTAL_COMBINATIONS=0
for agents in "${AGENT_COUNTS[@]}"; do
    for graphs in "${GRAPH_COUNTS[@]}"; do
        for turns in "${DIALOGUE_TURNS[@]}"; do
            TOTAL_COMBINATIONS=$((TOTAL_COMBINATIONS + 1))
        done
    done
done

# 进度条函数
print_progress_bar() {
    local progress=$1
    local total=$2
    local width=40
    local percent=$((progress * 100 / total))
    local filled=$((progress * width / total))
    local empty=$((width - filled))
    local bar=""
    for ((i=0; i<filled; i++)); do
        bar="${bar}#"
    done
    for ((i=0; i<empty; i++)); do
        bar="${bar}-"
    done
    printf "\r进度: [%s] %d/%d (%d%%)" "$bar" "$progress" "$total" "$percent"
}

CURRENT_TEST=0

# 批量测试循环
for agents in "${AGENT_COUNTS[@]}"; do
    for graphs in "${GRAPH_COUNTS[@]}"; do
        for turns in "${DIALOGUE_TURNS[@]}"; do
            TOTAL_TESTS=$((TOTAL_TESTS + 1))
            CURRENT_TEST=$((CURRENT_TEST + 1))
            
            # 打印进度条
            print_progress_bar $CURRENT_TEST $TOTAL_COMBINATIONS

            echo ""
            echo "=========================================="
            echo "测试 $TOTAL_TESTS: agents=$agents, graphs=$graphs, turns=$turns"
            echo "=========================================="
            
            # 运行测试
            python -m NewMA.graph_generator \
                --num_agents $agents \
                --num_graphs $graphs \
                --num_dialogue_turns $turns \
                --model_type $MODEL_TYPE \
                --attacker_strategy $ATTACKER_STRATEGY \
                --msmarco_path $MSMARCO_PATH \
                #--verbose
            
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

# 打印100%进度条
print_progress_bar $TOTAL_COMBINATIONS $TOTAL_COMBINATIONS
echo ""

# 记录测试结束时间
END_TIME=$(date)
echo "=========================================="
echo "批量测试完成"
echo "=========================================="
echo "开始时间: $START_TIME"
echo "结束时间: $END_TIME"
echo "总测试数: $TOTAL_TESTS"
echo "成功测试: $SUCCESS_TESTS"
echo "失败测试: $FAILED_TESTS"
echo "成功率: $((SUCCESS_TESTS * 100 / TOTAL_TESTS))%"
echo "==========================================" 