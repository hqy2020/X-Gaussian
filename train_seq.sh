#!/bin/bash

# 定义配置文件数组
configs=("chest" "foot" "head" "abdomen" "pancreas")
train_nums=(3 6 9)

# 创建新的tmux会话名称，使用时间戳避免冲突
session_name="training_seq_$(date +%Y%m%d_%H%M%S)"

# 创建/清空 train_log 文件
> train_log

# 创建新的tmux会话
tmux new-session -d -s "$session_name"

# 创建主窗口用于显示训练状态
tmux rename-window -t "$session_name:0" "training"

# 初始化conda并激活环境
tmux send-keys -t "$session_name:0" "source ~/anaconda3/etc/profile.d/conda.sh" C-m
tmux send-keys -t "$session_name:0" "conda activate x_gaussian" C-m

# 一开始设置GPU
gpu="0"
# 设置GPU分配
declare -A gpu_map
gpu_map["chest"]=$gpu
gpu_map["foot"]=$gpu
gpu_map["head"]=$gpu
gpu_map["abdomen"]=$gpu
gpu_map["pancreas"]=$gpu

# 计算总任务数
total_tasks=$((${#configs[@]} * ${#train_nums[@]}))
current_task=1

# 在主窗口中串行执行所有训练任务
for config in "${configs[@]}"; do
    for train_num in "${train_nums[@]}"; do
        # 获取对应的GPU
        gpu="${gpu_map[$config]}"
        
        # 构建训练命令
        cmd="echo '\n============== 开始训练 ${config} (train_num=${train_num}) ==============' 2>&1 | tee -a train_log && "
        cmd+="CUDA_VISIBLE_DEVICES=$gpu python train.py --train_num $train_num --config config/${config}.yaml 2>&1 | tee -a train_log"
        
        # 在tmux窗口中执行训练命令
        tmux send-keys -t "$session_name:0" "$cmd" C-m
        
        # 等待当前任务完成后再继续下一个
        sleep 1
        
        ((current_task++))
    done
done

echo "所有训练任务已在tmux会话 '$session_name' 中启动"
echo "使用以下命令连接到tmux会话："
echo "tmux attach-session -t $session_name"

# 自动连接到tmux会话
tmux attach-session -t "$session_name" 