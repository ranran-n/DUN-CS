import math
import os
import glob
import torch
from utils import utility
import argparse
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import cv2
from skimage.metrics import structural_similarity as ssim
import argparse
import warnings
from thop import profile
# set the path of test model

model_path = './epochs/25.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)

parser = argparse.ArgumentParser(description='CSDUN-Net')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--dir_data', type=str, default='./dataset', help='dataset directory')
parser.add_argument('--dir', type=str, default='./res_images/', help='save reconstruct images')
parser.add_argument('--test_name', type=str, default='Set11', help='test dataset name')
parser.add_argument('--save_results', default='True', action='store_true', help='save output results')
parser.add_argument('--sensing_rate', type=float, default=0.25, help='set sensing rate')
parser.add_argument('--block_size', type=int, default=32, help='block size (default: 32)')
parser.add_argument('--save_dir', type=str, default='save_OCTUF', help='The directory used to save models')
parser.add_argument('--result_dir', type=str, default='result_octuf', help='result directory')
parser.add_argument('--lr', '--learning_rate', default=5e-4, type=float, help='initial learning rate')

args = parser.parse_args()
warnings.filterwarnings("ignore")


def main():
    global args
    args = parser.parse_args()

    ext = {'/*.jpg', '/*.png', '/*.tif'}
    filepaths = []
    test_dir = os.path.join('./dataset', args.test_name)
    for img_type in ext:
        filepaths = filepaths + glob.glob(test_dir + img_type)

    result_dir = os.path.join(args.result_dir, args.test_name, str(args.sensing_rate))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)
    Time_All = np.zeros([1, ImgNum], dtype=np.float32)
    with torch.no_grad():
        print("\nCS Reconstruction Start")
        for img_no in range(ImgNum):
            imgName = filepaths[img_no]

            Img = cv2.imread(imgName, 1)

            # new_h = min(Img.shape[0], 256)
            # new_w = min(Img.shape[1], 256)
            # Img= Img[:new_h, :new_w]

            Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
            Img_rec_yuv = Img_yuv.copy()

            Iorg_y = Img_yuv[:, :, 0]
            [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
            Img_output = Ipad / 255.

            batch_x = torch.from_numpy(Img_output)
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x = batch_x.to(device)
            batch_x = batch_x.unsqueeze(0).unsqueeze(0)

            start = time()
            x_output = model(batch_x)
            end = time()

            x_output = x_output.squeeze(0).squeeze(0)
            Prediction_value = x_output.cpu().data.numpy()
            X_rec = np.clip(Prediction_value[:row, :col], 0, 1)

            rec_PSNR = psnr(X_rec * 255, Iorg.astype(np.float64))
            rec_SSIM = ssim(X_rec * 255, Iorg.astype(np.float64), data_range=255)

            test_name_split = os.path.split(imgName)
            print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (
                img_no, ImgNum, test_name_split[1], (end - start), rec_PSNR, rec_SSIM))

            Img_rec_yuv[:, :, 0] = X_rec * 255
            im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
            im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

            resultName = "./%s/%s" % (result_dir, test_name_split[1])
            cv2.imwrite("%s_lr_%.4f_ratio_%.2f_PSNR_%.2f_SSIM_%.4f.png" % (
                resultName, args.lr, args.sensing_rate, rec_PSNR, rec_SSIM), im_rec_rgb)
            del x_output

            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM
            if img_no >= 1:
                Time_All[0, img_no] = (end - start)


    print('\n')
    output_data = "CS ratio is %.2f, Avg PSNR/SSIM/Time for %s is %.2f/%.4f/%.4f\n" % (
        args.sensing_rate, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), np.mean(Time_All))
    print(output_data)

    print("CS Reconstruction End")


def imread_CS_py(Iorg):
    block_size = args.block_size
    [row, col] = Iorg.shape
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.math.log10(PIXEL_MAX / np.math.sqrt(mse))


if __name__ == '__main__':
    main()