import os
import shutil
import torch


def save_model_checkpoint(state, is_best, output_dir, exp_name, step):
    file_name = "model_ckpt_{}.tar".format(step)
    output_path = os.path.join(output_dir, exp_name, file_name)
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, output_path)
    if is_best:
        shutil.copyfile(output_path, os.path.join(output_dir, exp_name, "model_ckpt_best.tar".format(step)))
