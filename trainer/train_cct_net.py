import torch
from typing import List, Dict, Optional, Union, Tuple
from tqdm.auto import tqdm
import numpy as np
import sys
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
import random
#from utils.metrics import AverageMeter
from scipy.ndimage import zoom
from utils.metrics import AverageMeter

def train(
        epoch: int,
        loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        device: torch.device,
        criterion: Dict,
        weights: torch.tensor,
        prossesID: int = None
        ) -> Tuple[int, list]:

    model.train()

    prefix = 'Training'

    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = AverageMeter()

    with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
        for batch_i, sample in enumerate(loader, 0):

            # ================= Extract Data ==================
            batch_img = sample['image'].to(device)
            batch_label = sample['label'].to(device)

            # =================== forward =====================
            output, output_aux1 = model(batch_img)
            output_soft1 = torch.softmax(output, dim=1)
            output_soft2 = torch.softmax(output_aux1, dim=1)

            loss_ce1 = criterion['ce_loss'](output, batch_label[:].long())
            loss_ce2 = criterion['ce_loss'](output_aux1, batch_label[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            beta = random.random() + 1e-10

            pseudo_supervision = torch.argmax(
                (beta * output_soft1.detach() + (1.0-beta) * output_soft2.detach()), dim=1, keepdim=False)

            loss_pse_sup = 0.5 * (criterion['dice_loss'](output_soft1, pseudo_supervision.unsqueeze(
                1)) + criterion['dice_loss'](output_soft2, pseudo_supervision.unsqueeze(1)))

            loss = loss_ce + 0.5 * loss_pse_sup
            # =================== backward ====================
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.update()

            losses.update(loss.item(), batch_img.size(0))
            pbar.set_description(f"Epoch {epoch} - Trainig Loss: {losses.avg:.4f}")

    return losses


def validate(mode: str,
             epoch: int,
             loader: torch.utils.data.DataLoader,
             model: torch.nn.Module,
             device: torch.device,
             criterion: torch.nn.Module,
             weights: torch.tensor,
             prossesID: int = None
             ) -> Tuple[int, list]:

    model.eval()

    if mode == 'validation':
        prefix = 'Validating'
    elif mode == 'test':
        prefix = 'Testing'
    if prossesID is not None:
        prefix = "[{}]{}".format(prossesID, prefix)

    losses = AverageMeter()

    with torch.inference_mode():
        with tqdm(total=len(loader), ascii=True, desc=('{}: {:02d}'.format(prefix, epoch))) as pbar:
            for batch_i, sample in enumerate(loader, 0):

                # ================= Extract Data ==================
                batch_img = sample['image'].to(device)
                batch_gt = sample['gt'].to(device)

                # =================== forward =====================
                outputs = model(batch_img)[0]
                outputs_soft = torch.softmax(outputs, dim=1)
                loss = criterion(outputs_soft, batch_gt.unsqueeze(1).long(), weights)

                # =================== backward ====================
                pbar.update()

                losses.update(loss.item(), batch_img.size(0))
                pbar.set_description(f"Epoch {epoch} - Val Loss: {losses.avg:.4f}")
    return losses
