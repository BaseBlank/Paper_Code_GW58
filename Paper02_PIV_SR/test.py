"""
Ultimate ownership of all code in the repository belongs to this repository owner@BaseBlank of GitHub.
This code repository is subject to AGPL-3.0, and to use this project code you must also comply with AGPL-3.0.
About the specific content of AGPL - 3.0 protocol, you can refer to the following link:
    https://www.gnu.org/licenses/agpl-3.0.en.html
The code reference comes from Lornatang's DRRN-PyTorch code repository. Thanks again for Lornatang's excellent work and open source contribution.
The link to the reference code repository is as follows:
    https://github.com/Lornatang/RDN-PyTorch
"""
# ==============================================================================
import os

import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
import model
from image_quality_assessment import PSNR, SSIM
from utils import make_directory

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main() -> None:
    # Initialize the super-resolution sr_model
    sr_model = model.__dict__[config.model_arch_name](in_channels=config.in_channels,
                                                      out_channels=config.out_channels,
                                                      channels=config.channels)
    sr_model = sr_model.to(device=config.device)
    print(f"Build `{config.model_arch_name}` model successfully.")

    # Load the super-resolution sr_model weights
    checkpoint = torch.load(config.model_weights_path, map_location=lambda storage, loc: storage)
    sr_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{config.model_arch_name}` model weights `{os.path.abspath(config.model_weights_path)}` successfully.")

    # utils custom function, Create a folder of super-resolution experiment results
    make_directory(config.sr_dir)

    # Start the verification mode of the sr_model.
    sr_model.eval()

    # Initialize the sharpness evaluation function
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified sr_model
    psnr_model = psnr_model.to(device=config.device, non_blocking=True)
    ssim_model = ssim_model.to(device=config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.顺序排列
    file_names = natsorted(os.listdir(config.gt_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        gt_image_path = os.path.join(config.gt_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        lr_image_path = os.path.join(config.lr_dir, file_names[index])

        # 将lr_image_path字符串的中的GT替换为LR，区分字母大小写
        lr_image_path = lr_image_path.replace("GT", "LR")
        # 将sr_image_path字符串的中的GT替换为SR，区分字母大小写
        sr_image_path = sr_image_path.replace("GT", "SR")

        print(f"Processing `{os.path.abspath(gt_image_path)}`...")  # 取绝对路径
        # preprocess_one_data已经改了包含[H,W,C]变换[C,H,W]的操作
        lr_tensor, lr_tensor_max, lr_tensor_min = imgproc.preprocess_one_data(lr_image_path, config.device)
        gt_tensor, gt_tensor_max, gt_tensor_min = imgproc.preprocess_one_data(gt_image_path, config.device)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = sr_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)  # 调回(H, W, C)
        SR_image = sr_image * (lr_tensor_max - lr_tensor_min) + lr_tensor_min
        SR_image = SR_image.astype(np.float32)

        np.save(sr_image_path, SR_image)
        # sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        psnr_metrics += psnr_model(sr_tensor, gt_tensor).item()
        ssim_metrics += ssim_model(sr_tensor, gt_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} [dB]\n"
          f"SSIM: {avg_ssim:4.4f} [u]")


if __name__ == "__main__":
    main()
