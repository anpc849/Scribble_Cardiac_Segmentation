import torch
import numpy as np
from hausdorff import hausdorff_distance
from monai.metrics import compute_average_surface_distance #need to convert to one_hot_code
from tqdm.auto import tqdm
import torch.nn.functional as F
from scipy import ndimage

#!pip install -q monai hausdorff

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def border_map(binary_img):
    """
    Creates the border for a 3D or 2D image
    """
    ndims = binary_img.ndim
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    if ndims == 2:
        left = ndimage.shift(binary_map, [-1, 0], order=0)
        right = ndimage.shift(binary_map, [1, 0], order=0)
        superior = ndimage.shift(binary_map, [0, 1], order=0)
        inferior = ndimage.shift(binary_map, [0, -1], order=0)
        cumulative = left + right + superior + inferior
        ndir = 4
    elif ndims == 3:
        left = ndimage.shift(binary_map, [-1, 0, 0], order=0)
        right = ndimage.shift(binary_map, [1, 0, 0], order=0)
        anterior = ndimage.shift(binary_map, [0, 1, 0], order=0)
        posterior = ndimage.shift(binary_map, [0, -1, 0], order=0)
        superior = ndimage.shift(binary_map, [0, 0, 1], order=0)
        inferior = ndimage.shift(binary_map, [0, 0, -1], order=0)
        cumulative = left + right + anterior + posterior + superior + inferior
        ndir = 6
    else:
        raise RuntimeError(f'Image must be of 2 or 3 dimensions, got {ndims}')
    border = ((cumulative < ndir) * binary_map) == 1
    return border
    
def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    border_ref = border_map(ref)
    border_seg = border_map(seg)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg
        
class Evaluator(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def evaluate(self, model, val_loader, device, softmax=True):
        """
        preds: [batch_size, n_class, height, width]
        truth: [batch_size, height, width]
        """
        metrics = {
            'Dice': [],
            'Acc': [],
            'Pre': [],
            'Sen': [],
            'Spe': [],
            'HD': [],
            'ASD': []
        }
        model.to(device)
        model.eval()
        with torch.inference_mode():
            with tqdm(total=len(val_loader), ascii=True, desc='Evaluating') as pbar:
                for batch_i, sample in enumerate(val_loader, 0):

                    # ================= Extract Data ==================
                    img = sample['image'].to(device)
                    label = sample['gt'].to(device) # [batch_size,height,width]

                    # =================== forward =====================

                    logits = model(img)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    logtis = logits.detach()
                    if softmax:
                        logits = F.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1) # [batch_size, height, width]
                    preds = preds.cpu()
                    label = label.cpu()

                    score_list = {
                        'Dice': np.empty(self.n_classes),
                        'Acc': np.empty(self.n_classes),
                        'Pre': np.empty(self.n_classes),
                        'Sen': np.empty(self.n_classes),
                        'Spe': np.empty(self.n_classes),
                        'HD': np.empty(self.n_classes),
                        'ASD': None,
        }
                    # =================== metrics =====================
                    for c in range(0, self.n_classes):
                        score_list['Dice'][c] = self._calc_DSC_Sets(label, preds, c)
                        score_list['Acc'][c] = self._calc_Accuracy_Sets(label, preds, c)
                        score_list['Pre'][c] = self._calc_Precision_Sets(label, preds, c)
                        score_list['Sen'][c] = self._calc_Sensitivity_Sets(label, preds, c)
                        score_list['Spe'][c] = self._calc_Specificity_Sets(label, preds, c)
                        score_list['HD'][c] = self._calc_AverageHausdorffDistance(label, preds, c)
                    score_list['ASD'] = self._calc_ASD(self._one_hot_encoder(label), self._one_hot_encoder(preds))

                    # =================== logging =====================
                    for k, v in score_list.items():
                        metrics[k].append(v)

                    #print(metrics['Dice'])

                    pbar.update()
        #print(metrics)
        for k, v in metrics.items():
            to_arr = np.array(v)
            metrics[k] = to_arr.mean(0)

        return metrics



    def _calc_DSC_Sets(self, truth, pred, c=1):
        # Obtain sets with associated class
        gt = np.equal(truth, c)
        pd = np.equal(pred, c)
        # Calculate Dice
        if (pd.sum() + gt.sum()) != 0:
            dice = 2*np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
        else : dice = 0.0
        # Return computed Dice
        return dice

    def _calc_Accuracy_Sets(self, truth, pred, c=1): #DONE
        # Obtain sets with associated class
        batch_size = truth.shape[0]
        acc_list = []
        for sample in range(batch_size):
            gt = torch.eq(truth[sample], c)
            pd = torch.eq(pred[sample], c)
            not_gt = torch.logical_not(gt)
            not_pd = torch.logical_not(pd)
            # Calculate Accuracy
            acc = (torch.logical_and(pd, gt).sum().item() +
                   torch.logical_and(not_pd, not_gt).sum().item()) / gt.numel()
            acc_list.append(acc)
        acc_score = np.mean(acc_list)
        # Return computed Accuracy
        return acc_score

    def _calc_Precision_Sets(self, truth, pred, c=1): #DONE
        # Obtain sets with associated class
        batch_size = truth.shape[0]
        pre_list = []
        for sample in range(batch_size):
            gt = torch.eq(truth[sample], c)
            pd = torch.eq(pred[sample], c)
            # Calculate precision
            if pd.sum().item() != 0:
                prec = torch.logical_and(pd, gt).sum().item() / pd.sum().item()
            else:
                prec = 0.0
            pre_list.append(prec)
        pre_score = np.mean(pre_list)
        # Return precision
        return pre_score

    def _calc_Sensitivity_Sets(self, truth, pred, c=1): #DONE
        # Obtain sets with associated class
        batch_size = truth.shape[0]
        sens_list = []
        for sample in range(batch_size):
            gt = torch.eq(truth[sample], c)
            pd = torch.eq(pred[sample], c)
            # Calculate sensitivity
            if gt.sum().item() != 0:
                sens = torch.logical_and(pd, gt).sum().item() / gt.sum().item()
            else:
                sens = 0.0
            sens_list.append(sens)
        sens_score = np.mean(sens_list)
        # Return sensitivity
        return sens_score

    def _calc_Specificity_Sets(self, truth, pred, c=1): #DONE
        # Obtain sets with associated class
        batch_size = truth.shape[0]
        spec_list = []
        for sample in range(batch_size):
            not_gt = torch.logical_not(torch.eq(truth[sample], c))
            not_pd = torch.logical_not(torch.eq(pred[sample], c))
            # Calculate specificity
            if not_gt.sum().item() != 0:
                spec = torch.logical_and(not_pd, not_gt).sum().item() / not_gt.sum().item()
            else:
                spec = 0.0
            spec_list.append(spec)
        spec_score = np.mean(spec_list)
        # Return specificity
        return spec_score

    def _calc_ASD(self, truth, pred): #Check for the accuracy
        ##truth and pred must be one-hot
        score = compute_average_surface_distance(pred, truth, symmetric=False, include_background=True)
        score = torch.mean(score, dim=0)
        return score.numpy()
    

    def _calc_AverageHausdorffDistance(self,truth, pred, c=1):
        """
        This functions calculates the average symmetric distance and the
        hausdorff distance between a segmentation and a reference image
        :return: hausdorff distance and average symmetric distance
        """
        # Obtain sets with associated class
        ref = np.equal(truth, c)
        seg = np.equal(pred, c)
        # Compute AHD
        ref_border_dist, seg_border_dist = border_distance(ref, seg)
        hausdorff_distance = np.max([np.max(ref_border_dist),
                                     np.max(seg_border_dist)])
        # Return AHD
        return hausdorff_distance

    def _one_hot_encoder(self, input_tensor): #uilts
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

