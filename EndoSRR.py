import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import normalize, threshold
from tqdm import tqdm
from PIL import Image
import datasets
import modelSAM
import utils

from torchvision import transforms
from mmcv.runner import load_checkpoint

# lama
import sys
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from lama_inpaint import inpaint_img_with_lama

# sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
# from saicinpainting.evaluation.utils import move_to_device
# from saicinpainting.training.trainers import load_checkpoint
# from saicinpainting.evaluation.data import pad_tensor_to_modulo

from utils import load_img_to_array, save_array_to_img, dilate_mask


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')

    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']
        # print((batch['gt']).shape)
        # print(inp.shape)

        pred = torch.sigmoid(model.infer(inp))

        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])

        if verbose:
            pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
            pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()

def img_process(path):

    image = Image.open(path).convert('RGB')
    print(in_dir)

    img_trans = transforms.Compose([
        transforms.Resize((1024, 1024)),  # (inp_size, inp_size)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    inp_sam = img_trans(image).unsqueeze(0).cuda()
    inp_inpa = np.array(image)

    return inp_sam, inp_inpa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--lama_config')
    parser.add_argument('--model')
    parser.add_argument('--lama_ckpt')
    parser.add_argument('--input_path')
    parser.add_argument('--save_mask_path')
    parser.add_argument('--save_inpaint_path')
    parser.add_argument('--final_mask_path')
    parser.add_argument('--final_inpaint_path')
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=9,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = modelSAM.make(config['model']).cuda()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)

    if not os.path.exists(args.save_mask_path):
        os.makedirs(args.save_mask_path)

    if not os.path.exists(args.save_inpaint_path):
        os.makedirs(args.save_inpaint_path)

    if not os.path.exists(args.final_mask_path):
        os.makedirs(args.final_mask_path)

    if not os.path.exists(args.final_inpaint_path):
        os.makedirs(args.final_inpaint_path)

    in_dirs = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)]
    in_dirs.sort()

    for in_dir in in_dirs:

        i = 0

        while True:

            print(i)

            if i == 0:
                inp, inp_inp = img_process(in_dir)
            else:
                inp, inp_inp = img_process(inpaint_p)

            # SAM_adaptor pred
            mask = model.pre_infer(inp, (1024, 1280))
            # mask = mask.astype(np.uint8) * 255
            mask = normalize(threshold(mask, 0.0, 0)).reshape(1, 1024, 1280).cpu()

            # path to the mask result
            mask_name = os.path.splitext(os.path.split(in_dir)[-1])[0] + '_' + str(i) + '.png'
            mask_p = os.path.join(args.save_mask_path, mask_name)

            # save the mask
            # save_array_to_img(mask, mask_p)
            transforms.ToPILImage()(mask).save(mask_p)

            # final mask
            if i == 0:
                final_mask = torch.zeros_like(mask)
                final_mask = torch.where(mask == 1, torch.tensor([1]), final_mask)
            else:
                final_mask = torch.where(mask == 1, torch.tensor([1]), final_mask)

            # cal the reflection ratio
            mask = mask.detach().numpy()
            ratio = np.sum(mask == 1) / (mask.shape[1]*mask.shape[2])
            print("ratio: ", ratio)

            # LaMa inpaint
            mask = load_img_to_array(mask_p)
            mask = dilate_mask(mask, args.dilate_kernel_size)

            img_inpainted = inpaint_img_with_lama(
                inp_inp, mask, args.lama_config, args.lama_ckpt, device=device)

            # path for inpaint result
            inpaint_p = os.path.join(args.save_inpaint_path, mask_name)

            # save inpaint result
            save_array_to_img(img_inpainted, inpaint_p)

            # save final result and break
            if ratio < 1.5e-4 or i >= 4:
                # save final mask
                mask_name = os.path.splitext(os.path.split(in_dir)[-1])[0] + '.png'
                mask_p = os.path.join(args.final_mask_path, mask_name)
                transforms.ToPILImage()(final_mask).save(mask_p)

                #save final inpaint
                inpaint_p = os.path.join(args.final_inpaint_path, mask_name)
                save_array_to_img(img_inpainted, inpaint_p)

                break
            i += 1




