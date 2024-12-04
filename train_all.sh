#!/bin/bash

# 定义配置文件数组
configs=("chest" "foot" "head" "abdomen" "pancreas")
train_nums=(3 6 9)

# 创建新的tmux会话名称，使用时间戳避免冲突
session_name="training_$(date +%Y%m%d_%H%M%S)"

# 创建新的tmux会话
tmux new-session -d -s "$session_name"

# 创建一个窗口用于显示训练状态
tmux new-window -t "$session_name:1" -n "status"
tmux send-keys -t "$session_name:1" "echo '训练状态监控窗口'" C-m

# 用于跟踪窗口索引
window_index=2

# 遍历所有组合
for config in "${configs[@]}"; do
    for train_num in "${train_nums[@]}"; do
        # 根据配置文件选择GPU
        if [[ "$config" == "abdomen" || "$config" == "pancreas" ]]; then
            gpu="1"
        else
            gpu="0"
        fi
        
        # 为每个训练任务创建新的窗口
        window_name="${config}_${train_num}"
        tmux new-window -t "$session_name:$window_index" -n "$window_name"
        
        # 构建训练命令
        cmd="CUDA_VISIBLE_DEVICES=$gpu python train.py --train_num $train_num --config config/${config}.yaml"
        
        # 在新窗口中执行训练命令
        tmux send-keys -t "$session_name:$window_index" "$cmd" C-m
        
        echo "Started training in window $window_index: config=${config}, train_num=${train_num}, GPU=${gpu}"
        
        ((window_index++))
    done
done

# 切换到第一个窗口
tmux select-window -t "$session_name:1"

echo "所有训练任务已在tmux会话 '$session_name' 中启动"
echo "使用以下命令连接到tmux会话："
echo "tmux attach-session -t $session_name"

# 自动连接到tmux会话
tmux attach-session -t "$session_name"
