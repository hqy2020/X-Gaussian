#!/bin/bash

# 定义配置
config="chest"

# 定义训练视角数组
view_nums=(3 6 9)

# 初始化存储结果的数组
declare -A results

# 为每个视角数量运行训练
for train_num in "${view_nums[@]}"; do
    echo "Starting training with ${train_num} views..."
    
    # 运行训练并将输出保存到临时文件
    temp_log="temp_${train_num}.log"
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --train_num ${train_num} \
        --scene ${config} \
        --source_path data/${config} \
        --model_path output/${config}_${train_num}views \
        --config config/${config}.yaml \
        --test_iterations 2000 20000 \
        --save_iterations 20000 \
        --gaussiansN 2 \
        --coreg \
        --coprune \
        2>&1 | tee $temp_log
    
    # 从输出中提取关键信息
    echo "" >> $log_file
    echo "Results for ${train_num} views:" >> $log_file
    echo "--------------------------------" >> $log_file
    
    # 提取关键指标
    body_part=$(grep "Body part:" $temp_log | tail -n 1)
    test_speed=$(grep "Testing Speed:" $temp_log | tail -n 1)
    total_time=$(grep "Total time:" $temp_log | tail -n 1)
    test_ssim=$(grep "Test SSIM:" $temp_log | tail -n 1)
    test_psnr=$(grep "Test PSNR:" $temp_log | tail -n 1)
    points_count_0=$(grep "Gaussian0 final points count:" $temp_log | tail -n 1)
    points_count_1=$(grep "Gaussian1 final points count:" $temp_log | tail -n 1)
    final_loss=$(grep "Final loss:" $temp_log | tail -n 1)
    initial_loss=$(grep "Initial loss:" $temp_log | tail -n 1)
    best_loss=$(grep "Best loss:" $temp_log | tail -n 1)
    train_ssim=$(grep "Train SSIM:" $temp_log | tail -n 1)
    train_psnr=$(grep "Train PSNR:" $temp_log | tail -n 1)
    save_path=$(grep "Save path:" $temp_log | tail -n 1)
    
    # 写入日志文件
    echo "$body_part" >> $log_file
    echo "$test_speed" >> $log_file
    echo "$total_time" >> $log_file
    echo "$test_ssim" >> $log_file
    echo "$test_psnr" >> $log_file
    echo "$points_count_0" >> $log_file
    echo "$points_count_1" >> $log_file
    echo "$final_loss" >> $log_file
    echo "$initial_loss" >> $log_file
    echo "$best_loss" >> $log_file
    echo "$train_ssim" >> $log_file
    echo "$train_psnr" >> $log_file
    echo "$save_path" >> $log_file
    echo "" >> $log_file
    
    # 删除临时文件
    rm $temp_log
done

# 输出汇总信息
echo "Training completed for all view configurations!"
echo "Summary has been saved to $log_file"
echo "================================="
cat $log_file
