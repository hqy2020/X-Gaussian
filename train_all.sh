#!/bin/bash

# 定义配置文件数组
configs=("chest" "foot" "head" "abdomen" "pancreas")
train_nums=(3 6 9)

# 用于跟踪运行的进程数量
process_count=0

# 存储所有后台进程的信息
declare -A pids
declare -A status

# 信号处理函数
cleanup() {
    echo -e "\n正在终止所有训练进程..."
    for pid in "${!pids[@]}"; do
        kill -9 $pid 2>/dev/null
    done
    print_summary
    exit 1
}

# 打印训练结果统计
print_summary() {
    echo -e "\n训练结果统计："
    echo "----------------------------------------"
    echo "配置文件    训练样本数    GPU    状态"
    echo "----------------------------------------"
    for key in "${!pids[@]}"; do
        if wait $key 2>/dev/null; then
            status[$key]="成功"
        else
            status[$key]="失败"
        fi
        echo "${pids[$key]} -> ${status[$key]}"
    done
    echo "----------------------------------------"
    
    # 统计成功和失败的数量
    success_count=0
    fail_count=0
    for key in "${!status[@]}"; do
        if [ "${status[$key]}" == "成功" ]; then
            ((success_count++))
        else
            ((fail_count++))
        fi
    done
    
    echo -e "\n总结："
    echo "成功: $success_count"
    echo "失败: $fail_count"
    echo "总计: $process_count"
}

# 注册SIGINT信号处理
trap cleanup SIGINT SIGTERM

# 遍历所有组合
for config in "${configs[@]}"; do
    for train_num in "${train_nums[@]}"; do
        # 根据配置文件选择GPU
        if [[ "$config" == "abdomen" || "$config" == "pancreas" ]]; then
            gpu="1"
        else
            gpu="0"
        fi
        
        # 运行训练命令
        CUDA_VISIBLE_DEVICES=$gpu python train.py \
            --train_num $train_num \
            --config "config/${config}.yaml" &
            
        # 存储进程信息
        pid=$!
        pids[$pid]="$config    $train_num        $gpu"
        
        # 增加进程计数
        ((process_count++))
        
        echo "Started training: config=${config}, train_num=${train_num}, GPU=${gpu}, PID=$pid"
    done
done

# 等待所有进程完成
wait

# 打印最终统计结果
print_summary

echo "所有训练进程已完成！"
