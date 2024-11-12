import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# from pytorch_msssim import ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel_Xray
from utils.general_utils import safe_state, gen_log
from tqdm import tqdm
from utils.image_utils import psnr, time2file_name
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import datetime
import time
import yaml
import shutil

from pdb import set_trace as stx

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, gaussiansN, coreg, coprune, coprune_threshold):
    

    training_start_time = time.time()
    assert gaussiansN >= 1 and gaussiansN <= 2
    
    first_iter = 0
    exp_logger = prepare_output_and_logger(dataset)
    exp_logger.info("Training parameters: {}".format(vars(opt)))

    # 初始化第一个高斯场
    gaussians = GaussianModel_Xray(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 创建高斯场字典
    GsDict = {}
    for i in range(gaussiansN):
        if i == 0:
            GsDict[f"gs{i}"] = gaussians
        elif i > 0:
            GsDict[f"gs{i}"] = GaussianModel_Xray(dataset.sh_degree)
            GsDict[f"gs{i}"].create_from_pcd(scene.init_point_cloud, scene.cameras_extent)
            GsDict[f"gs{i}"].training_setup(opt)
            print(f"Create gaussians{i}")
    print(f"GsDict.keys() is {GsDict.keys()}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # 更新学习率
        for i in range(gaussiansN):
            GsDict[f"gs{i}"].update_learning_rate(iteration)

        # 每1000轮增加SH度数
        if iteration % 1000 == 0:
            for i in range(gaussiansN):
                GsDict[f"gs{i}"].oneupSHdegree()

        # 选择随机相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 为每个高斯场渲染
        RenderDict = {}
        LossDict = {}
        for i in range(gaussiansN):
            render_pkg = render(viewpoint_cam, GsDict[f"gs{i}"], pipe, bg)
            RenderDict[f"image_gs{i}"] = render_pkg["render"]
            RenderDict[f"viewspace_point_tensor_gs{i}"] = render_pkg["viewspace_points"]
            RenderDict[f"visibility_filter_gs{i}"] = render_pkg["visibility_filter"]
            RenderDict[f"radii_gs{i}"] = render_pkg["radii"]

        # 计算每个高斯场的损失
        gt_image = viewpoint_cam.normalized_image.cuda()
        for i in range(gaussiansN):
            Ll1 = l1_loss(RenderDict[f"image_gs{i}"], gt_image)
            LossDict[f"loss_gs{i}"] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(RenderDict[f"image_gs{i}"], gt_image))
            LossDict[f"loss_gs{i}"].backward()

        iter_end.record()

        with torch.no_grad():
            # 更新进度条
            loss = LossDict["loss_gs0"]
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 评估和保存
            training_report(exp_logger, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                          testing_iterations, scene, render, (pipe, background))

            # 保存模型
            if iteration in saving_iterations:
                exp_logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if gaussiansN == 2:
                    pcd_path = os.path.join(scene.model_path, "point_cloud_gs2/iteration_{}".format(iteration))
                    os.makedirs(pcd_path, exist_ok=True)
                    GsDict["gs1"].save_ply(os.path.join(pcd_path, "point_cloud.ply"))

            # 密集化处理
            if iteration < opt.densify_until_iter:
                for i in range(gaussiansN):
                    viewspace_point_tensor = RenderDict[f"viewspace_point_tensor_gs{i}"]
                    visibility_filter = RenderDict[f"visibility_filter_gs{i}"]
                    radii = RenderDict[f"radii_gs{i}"]
                    GsDict[f"gs{i}"].max_radii2D[visibility_filter] = torch.max(
                        GsDict[f"gs{i}"].max_radii2D[visibility_filter], 
                        radii[visibility_filter])
                    GsDict[f"gs{i}"].add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    for i in range(gaussiansN):
                        GsDict[f"gs{i}"].densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

            # 优化器步进
            if iteration < opt.iterations:
                for i in range(gaussiansN):
                    GsDict[f"gs{i}"].optimizer.step()
                    GsDict[f"gs{i}"].optimizer.zero_grad(set_to_none = True)

            # 重置不透明度
            for i in range(gaussiansN):
                if iteration % opt.opacity_reset_interval == 0:
                    GsDict[f"gs{i}"].reset_opacity()

            # 协同剪枝
            if coprune and iteration > opt.densify_from_iter and iteration % 500 == 0:
                for i in range(gaussiansN):
                    for j in range(gaussiansN):
                        if i != j:
                            source_cloud = o3d.geometry.PointCloud()
                            source_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{i}"].get_xyz.clone().cpu().numpy())
                            target_cloud = o3d.geometry.PointCloud()
                            target_cloud.points = o3d.utility.Vector3dVector(GsDict[f"gs{j}"].get_xyz.clone().cpu().numpy())
                            trans_matrix = np.identity(4)
                            evaluation = o3d.pipelines.registration.evaluate_registration(source_cloud, target_cloud, coprune_threshold, trans_matrix)
                            correspondence = np.array(evaluation.correspondence_set)
                            mask_consistent = torch.zeros((GsDict[f"gs{i}"].get_xyz.shape[0], 1)).cuda()
                            mask_consistent[correspondence[:, 0], :] = 1
                            GsDict[f"mask_inconsistent_gs{i}"] = ~(mask_consistent.bool())
                
                for i in range(gaussiansN):
                    GsDict[f"gs{i}"].prune_from_mask(GsDict[f"mask_inconsistent_gs{i}"].squeeze(), iter=iteration)



def prepare_output_and_logger(args):    
    if not args.model_path:
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        args.model_path = os.path.join("./output/", args.scene, date_time)
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    exp_logger = gen_log(args.model_path)

    return exp_logger



def training_report(exp_logger, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, GsDict=None):
    if exp_logger and (iteration == 0 or (iteration+1) % 100 == 0):
        exp_logger.info(f"Iter:{iteration}, L1 loss={Ll1.item():.4g}, Total loss={loss.item():.4g}, Time:{int(elapsed)}")

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                            {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                psnr_test = 0.0
                ssim_test = 0.0
                start = time.time()
                for idx, viewpoint in enumerate(config['cameras']):
                    if GsDict is not None and len(GsDict) > 1:
                        for i in range(len(GsDict)):
                            render_results = renderFunc(viewpoint, GsDict[f"gs{i}"], *renderArgs)
                            # ... 处理每个高斯场的渲染结果 ...
                    else:
                        render_results = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                        # ... 处理单个高斯场的渲染结果 ...

                    image = torch.clamp(render_results["render"], 0.0, 1.0)
                    image_backnorm = (viewpoint.max_value - viewpoint.min_value) * image + viewpoint.min_value # 反归一化

                    # 通道平均
                    image = image.mean(dim=0, keepdim=True)
                    image_backnorm = image_backnorm.mean(dim=0, keepdim=True)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_image_norm = viewpoint.normalized_image.to("cuda")

                    ssim_test += ssim(image_backnorm, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image_norm).mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                end = time.time()
                exp_logger.info(f"Testing Speed: {len(config['cameras'])/(end-start)} fps")
                exp_logger.info(f"Testing Time: {end-start} s")
                exp_logger.info("\n[ITER {}] Evaluating {}: SSIM = {}, PSNR = {}".format(iteration, config['name'], ssim_test, psnr_test))

        if exp_logger:
            exp_logger.info(f'Iter:{iteration}, total_points:{scene.gaussians.get_xyz.shape[0]}')
        torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters") 
    lp = ModelParams(parser)                      # 
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/chest.yaml', help='Path to the configuration file')
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 2_000, 4_000, 6_000, 8_000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 2_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000,])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu_id", default="1", help="gpu to use")
    parser.add_argument('--gaussiansN', type=int, default=2)
    parser.add_argument("--coreg", action='store_true', default=True)
    parser.add_argument("--coprune", action='store_true', default=True)
    parser.add_argument('--coprune_threshold', type=int, default=5)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # 输出parser的配置
    print(args)
    print("Optimizing " + args.model_path)
    

    safe_state(args.quiet)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    for key, value in config.items():
        setattr(args, key, value)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.gaussiansN, args.coreg, args.coprune, args.coprune_threshold)

    print("\nTraining complete.")