import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms as T

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class DictConfig(object):
    """Creates a Config object from a dict 
       such that object attributes correspond to dict keys.    
    """

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def __str__(self):
        return '\n'.join(f"{key}: {val}" for key, val in self.__dict__.items())

    def __repr__(self):
        return self.__str__()


def get_config(fname):
    with open(fname, 'r') as stream:
        config_dict = yaml.load(stream, Loader)
    config = DictConfig(config_dict)
    return config


def pt_to_image(img):
    return img.detach_().cpu().mul_(0.5).add_(0.5)


def save_states(fname, gen, dis, g_optimizer, d_optimizer, n_iter, config):
    state_dicts = {'G': gen.state_dict(),
                   'D': dis.state_dict(),
                   'G_optim': g_optimizer.state_dict(),
                   'D_optim': d_optimizer.state_dict(),
                   'n_iter': n_iter}
    torch.save(state_dicts, f"{config.checkpoint_dir}/{fname}")
    print("Saved state dicts!")


def output_to_img(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out


@torch.inference_mode()
def infer_deepfill(generator,
                   image,
                   mask,
                   return_vals=['inpainted', 'stage1']):

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    image = (image*2 - 1.)  # map image values to [-1, 1] range
    # 1.: masked 0.: unmasked
    mask = (mask > 0.).to(dtype=torch.float32)

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]  # sketch channel
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels

    x_stage1, x_stage2 = generator(x, mask)

    image_compl = image * (1.-mask) + x_stage2 * mask

    output = []
    for return_val in return_vals:
        if return_val.lower() == 'stage1':
            output.append(output_to_img(x_stage1))
        elif return_val.lower() == 'stage2':
            output.append(output_to_img(x_stage2))
        elif return_val.lower() == 'inpainted':
            output.append(output_to_img(image_compl))
        else:
            print(f'Invalid return value: {return_val}')

    return output

def bbox_border_mask():
    """Generate box border mask tensor.

    Returns:
        torch.Tensor: output with shape [1, 1, 256, 256]

    """
    length = np.random.randint(15, 150)

    i = np.random.randint(4)
    if i == 0:  # Левая сторона (вертикальный прямоугольник)
        bbox = (0, 0, 256, length)  # (y, x, height, width)
    if i == 1:  # Правая сторона (вертикальный прямоугольник)
        bbox = (0, 256 - length, 256, length)
    if i == 2:  # Верхняя сторона (горизонтальный прямоугольник)
        bbox = (0, 0, length, 256)
    if i == 3:  # Нижняя сторона (горизонтальный прямоугольник)
        bbox = (256 - length, 0, length, 256)

    mask = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    mask[:, :, bbox[0]: bbox[0]+bbox[2], bbox[1]: bbox[1]+bbox[3]] = 1.0
    return mask

def noise_mask(path_to_masks, image_files):
    random_image_file = random.choice(image_files)
    mask = Image.open(path_to_masks + random_image_file).convert('L')
    mask = T.functional.to_tensor(mask)
    mask = mask.unsqueeze(0)
    return mask