import torch
from models.base.lpips import LPIPSLoss
from torch.optim.lr_scheduler import StepLR
from skimage.metrics import structural_similarity as compare_ssim


# Not thread safe.
lpips_model = None


def fuse_loss(y_true, y_pred):
    global lpips_model
    if lpips_model is None:
        lpips_model = LPIPSLoss(ckpt_path='vgg.pth').to(y_true)
    if y_true.shape[1] == 3:
        lpips = torch.mean(lpips_model(y_true, y_pred))
    else: # Hint here tensors are in full 13 bands.
        lpips = torch.mean(lpips_model(y_true[:, [3,2,1], :, :], y_pred[:, [3,2,1], :, :]))
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae + lpips