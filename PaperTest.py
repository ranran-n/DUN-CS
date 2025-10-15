import math
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from time import time


# 参数设置
class Args:
    def __init__(self):
        self.block_size = 32  # 块大小
        self.sensing_rate = 0.5  # 采样率
        self.test_name = "test"  # 测试名称
        self.lr = 0.001  # 学习率（保留用于兼容性）
        self.result_dir = "results"  # 结果保存目录


args = Args()


def calculate_image_metrics(original_img_path, reconstructed_img_path):
    """
    计算原始图片和重建图片之间的PSNR和SSIM指标
    :param original_img_path: 原始图片路径
    :param reconstructed_img_path: 重建图片路径
    :return: PSNR, SSIM, 计算时间
    """
    # 创建结果目录
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # 读取图片
    original_img = cv2.imread(original_img_path, 1)
    reconstructed_img = cv2.imread(reconstructed_img_path, 1)

    if original_img is None:
        raise ValueError(f"无法加载原始图片: {original_img_path}")
    if reconstructed_img is None:
        raise ValueError(f"无法加载重建图片: {reconstructed_img_path}")

    # 确保两张图片尺寸相同
    if original_img.shape != reconstructed_img.shape:
        # 调整重建图片尺寸以匹配原始图片
        reconstructed_img = cv2.resize(reconstructed_img,
                                       (original_img.shape[1], original_img.shape[0]))

    # 记录开始时间
    start = time()

    # 转换为YCrCb色彩空间，使用Y通道进行计算
    original_yuv = cv2.cvtColor(original_img, cv2.COLOR_BGR2YCrCb)
    reconstructed_yuv = cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2YCrCb)

    original_y = original_yuv[:, :, 0]
    reconstructed_y = reconstructed_yuv[:, :, 0]

    # 处理图像块（保持原始代码的块处理逻辑）
    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(original_y)
    [_, _, _, rec_Ipad, _, _] = imread_CS_py(reconstructed_y)

    # 提取有效区域并归一化
    X_original = Ipad[:row, :col] / 255.0
    X_rec = rec_Ipad[:row, :col] / 255.0

    # 计算PSNR和SSIM
    rec_PSNR = psnr(X_rec * 255, X_original * 255)
    rec_SSIM = ssim(X_rec * 255, X_original * 255, data_range=255)

    # 记录结束时间
    end = time()
    run_time = end - start

    # 输出结果
    original_name = os.path.split(original_img_path)[1]
    rec_name = os.path.split(reconstructed_img_path)[1]
    print(f"对比图片: {original_name} 和 {rec_name}")
    print(f"运行时间: {run_time:.4f}秒")
    print(f"PSNR: {rec_PSNR:.2f} dB")
    print(f"SSIM: {rec_SSIM:.4f}")

    # 保存对比结果图像
    result_name = f"comparison_{os.path.splitext(original_name)[0]}_{os.path.splitext(rec_name)[0]}"
    result_path = os.path.join(args.result_dir, result_name)

    # 创建并排显示的对比图
    combined = np.hstack((original_img, reconstructed_img))
    cv2.imwrite(f"{result_path}_PSNR_{rec_PSNR:.2f}_SSIM_{rec_SSIM:.4f}.png", combined)

    print("\n指标计算完成")
    return rec_PSNR, rec_SSIM, run_time


def imread_CS_py(Iorg):
    """处理图像块，确保尺寸为块大小的整数倍"""
    block_size = args.block_size
    [row, col] = Iorg.shape
    row_pad = block_size - np.mod(row, block_size) if np.mod(row, block_size) != 0 else 0
    col_pad = block_size - np.mod(col, block_size) if np.mod(col, block_size) != 0 else 0

    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def psnr(img1, img2):
    """计算两张图片的PSNR"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:  # 两张图片完全相同
        return 100.0
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# 使用示例
if __name__ == "__main__":
    # 输入两张图片的路径
    original_image_path = "noisy.png"  # 原始图片
    reconstructed_image_path = "test035.png"  # 重建图片

    # 计算并显示指标
    psnr_value, ssim_value, time_value = calculate_image_metrics(original_image_path, reconstructed_image_path)