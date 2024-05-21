import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import linalg as LA

def sym_reg(pred):
    loss = torch.zeros(pred.shape[0]).to(pred.device)
    l_bones = [(0, 12), (12, 13), (13, 14), (14, 15), (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (7, 21)]
    r_bones = [(0, 16), (16, 17), (17, 18), (18, 19), (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23)]
    for l_bone, r_bone in zip(l_bones, r_bones):
        l_bone_len = LA.norm(pred[:, l_bone[0], :] - pred[:, l_bone[1], :], 2, dim=-1)
        r_bone_len = LA.norm(pred[:, r_bone[0], :] - pred[:, r_bone[1], :], 2, dim=-1)
        loss += torch.abs(l_bone_len - r_bone_len)
    return torch.mean(loss)

def constraint_reg(pred, ratio=3):
    loss = torch.zeros(pred.shape[0]).to(pred.device)
    arms = [(9, 10), (5, 6), (17, 18), (13, 14)]
    hands = [(10, 11), (6, 7), (18, 19), (14, 15)]
    for arm, hand in zip(arms, hands):
        arm_len = LA.norm(pred[:, arm[0], :] - pred[:, arm[1], :], 2, dim=-1)
        hand_len = LA.norm(pred[:, hand[0], :] - pred[:, hand[1], :], 2, dim=-1)
        loss += torch.abs(arm_len/ratio - hand_len)
    return torch.mean(loss)


class ReconLoss(nn.Module):
    def __init__(self, p=2):
        super(ReconLoss, self).__init__()
        self.loss = nn.PairwiseDistance(p)

    def forward(self, pred, gt):
        B, V, C = pred.shape
        loss = self.loss(pred.contiguous().view(-1, C), gt.contiguous().view(-1, C))
        return loss.view(B, V).mean(-1).mean(-1)


class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()
        self.loss = nn.CosineSimilarity(dim=2)

    def forward(self, pred, gt):
        return  self.loss(pred, gt).mean(1).mean(0)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MMCE_weighted(nn.Module):
    """
    Computes MMCE_w loss.
    """
    def __init__(self, device, lamda=0.5):
        super(MMCE_weighted, self).__init__()
        self.device = device
        self.lamda = lamda

    def torch_kernel(self, matrix):
        return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

    def get_pairs(self, tensor1, tensor2):
        correct_prob_tiled = tensor1.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)
        incorrect_prob_tiled = tensor2.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)

        correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)],
                                    dim=2)
        incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)],
                                    dim=2)

        correct_prob_tiled_1 = tensor1.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
        incorrect_prob_tiled_1 = tensor2.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)

        correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)],
                                    dim=2)
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    def get_out_tensor(self, tensor1, tensor2):
        return torch.mean(tensor1*tensor2)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1)  #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, predicted_labels = torch.max(predicted_probs, 1)

        correct_mask = torch.where(torch.eq(predicted_labels, target),
                                    torch.ones(predicted_labels.shape).to(self.device),
                                    torch.zeros(predicted_labels.shape).to(self.device))

        k = torch.sum(correct_mask).type(torch.int64)
        k_p = torch.sum(1.0 - correct_mask).type(torch.int64)
        cond_k = torch.where(torch.eq(k,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        cond_k_p = torch.where(torch.eq(k_p,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        k = torch.max(k, torch.tensor(1).to(self.device))*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2
        k_p = torch.max(k_p, torch.tensor(1).to(self.device))*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
                                            (correct_mask.shape[0] - 2))


        correct_prob, _ = torch.topk(predicted_probs*correct_mask, k)
        incorrect_prob, _ = torch.topk(predicted_probs*(1 - correct_mask), k_p)

        correct_prob_pairs, incorrect_prob_pairs,\
               correct_incorrect_pairs = self.get_pairs(correct_prob, incorrect_prob)

        correct_kernel = self.torch_kernel(correct_prob_pairs)
        incorrect_kernel = self.torch_kernel(incorrect_prob_pairs)
        correct_incorrect_kernel = self.torch_kernel(correct_incorrect_pairs)

        sampling_weights_correct = torch.mm((1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0))

        correct_correct_vals = self.get_out_tensor(correct_kernel,
                                                          sampling_weights_correct)
        sampling_weights_incorrect = torch.mm(incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0))

        incorrect_incorrect_vals = self.get_out_tensor(incorrect_kernel,
                                                          sampling_weights_incorrect)
        sampling_correct_incorrect = torch.mm((1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0))

        correct_incorrect_vals = self.get_out_tensor(correct_incorrect_kernel,
                                                          sampling_correct_incorrect)

        correct_denom = torch.sum(1.0 - correct_prob)
        incorrect_denom = torch.sum(incorrect_prob)

        m = torch.sum(correct_mask)
        n = torch.sum(1.0 - correct_mask)
        mmd_error = 1.0/(m*m + 1e-5) * torch.sum(correct_correct_vals)
        mmd_error += 1.0/(n*n + 1e-5) * torch.sum(incorrect_incorrect_vals)
        mmd_error -= 2.0/(m*n + 1e-5) * torch.sum(correct_incorrect_vals)

        # ce
        ce = F.cross_entropy(input, target)

        return ce + self.lamda * torch.max((cond_k*cond_k_p).type(torch.FloatTensor).to(self.device).detach()*torch.sqrt(mmd_error + 1e-10), torch.tensor(0.0).to(self.device))

class LabelSmoothingCrossEntropy1(nn.Module):
    def __init__(self, alpha=0.5, ignore_index=-100, reduction="mean"):
        super(LabelSmoothingCrossEntropy1, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
            target = target.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(target != self.ignore_index).squeeze()
            input = input[index, :]
            target = target[index]

        confidence = 1. - self.alpha
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.alpha * smooth_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :].to(trans.device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def get_mmd_loss(z, z_prior, y, num_cls):
    y_valid = [i_cls in y for i_cls in range(num_cls)]
    z_mean = torch.stack([z[y==i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
    l2_z_mean= LA.norm(z.mean(dim=0), ord=2)
    mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
    return mmd_loss, l2_z_mean, z_mean[y_valid]

