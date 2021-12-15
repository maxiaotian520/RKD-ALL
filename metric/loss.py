import torch
import torch.nn as nn
import torch.nn.functional as F
from metric.utils import pdist
from metric.utils import angle_equation, numba_list_conversion
import numpy as np
import itertools

__all__ = ['L1Triplet', 'L2Triplet', 'ContrastiveLoss', 'RkdDistance', 'RKdAngle', 'HardDarkRank']


class _Triplet(nn.Module):
    def __init__(self, p=2, margin=0.2, sampler=None, reduce=True, size_average=True):
        super().__init__()
        self.p = p
        self.margin = margin

        # update distance function accordingly
        self.sampler = sampler
        self.sampler.dist_func = lambda e: pdist(e, squared=(p==2))

        self.reduce = reduce
        self.size_average = size_average

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        loss = F.triplet_margin_loss(anchor_embed, positive_embed, negative_embed,
                                     margin=self.margin, p=self.p, reduction='none')

        if not self.reduce:
            return loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class L2Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=2, margin=margin, sampler=sampler)


class L1Triplet(_Triplet):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__(p=1, margin=margin, sampler=sampler)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, sampler=None):
        super().__init__()
        self.margin = margin
        self.sampler = sampler

    def forward(self, embeddings, labels):
        anchor_idx, pos_idx, neg_idx = self.sampler(embeddings, labels)

        anchor_embed = embeddings[anchor_idx]
        positive_embed = embeddings[pos_idx]
        negative_embed = embeddings[neg_idx]

        pos_loss = (F.pairwise_distance(anchor_embed, positive_embed, p=2)).pow(2)
        neg_loss = (self.margin - F.pairwise_distance(anchor_embed, negative_embed, p=2)).clamp(min=0).pow(2)

        loss = torch.cat((pos_loss, neg_loss))
        return loss.mean()


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (ordered_student - torch.stack([torch.logsumexp(ordered_student[:, i:], dim=1) for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


class FitNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.transform = nn.Conv2d(in_feature, out_feature, 1, bias=False)
        self.transform.weight.data.uniform_(-0.005, 0.005)

    def forward(self, student, teacher):
        if student.dim() == 2:
            student = student.unsqueeze(2).unsqueeze(3)
            teacher = teacher.unsqueeze(2).unsqueeze(3)

        return (self.transform(student) - teacher).pow(2).mean()


class AttentionTransfer(nn.Module):
    def forward(self, student, teacher):
        s_attention = F.normalize(student.pow(2).mean(1).view(student.size(0), -1))

        with torch.no_grad():
            t_attention = F.normalize(teacher.pow(2).mean(1).view(teacher.size(0), -1))

        return (s_attention - t_attention).pow(2).mean()


class RKdAngle(nn.Module):

    def forward(self, student, teacher):
        # N x C
        # N x N x C
        #with torch.no_grad():
            #td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            #norm_td = F.normalize(td, p=2, dim=2)
            #t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        #---------------------------------------------------------------------------------------------------------------------------------
        teacher = teacher.tolist()   # (64, 512)
        student = student.tolist()
        
        alpha = 0.3
        # t_angle = angle_equation(numba_list_conversion(teacher), alpha)    #numba_list_conversion(teacher) ---> (64, 512)
        # s_angle = angle_equation(numba_list_conversion(student), alpha)
        t_angle = angle_equation(teacher, alpha)    #numba_list_conversion(teacher) ---> (64, 512)
        s_angle = angle_equation(student, alpha)
        #---------------------------------------------------------------------------------------------------------------------------------

        loss = F.smooth_l1_loss(torch.FloatTensor(s_angle), torch.FloatTensor(t_angle), reduction='mean')
        
        #---------------------------------------------------------------------------------------------------------------------------------
        # print('======file01')
        # fin = open("final_ep.txt", "a+")
        # print('======file02', fin)
        # for line in fin:
        #     print('======file03')
        #     if line == '1':
        #         with open("updated_metric.txt", "a+") as file:
                    
        #             file.write("Teacher Angle: \n")
        #             for element in t_angle:
        #                 file.write(str(element) + "\n")
        #             file.write("Student Angle: \n")
        #             for element in s_angle:
        #                 file.write(str(element) + "\n")
        #             file.write("With a loss of: " + str(loss.item()) + "\n")
        #         file.close()
        # fin.close()

        # with open("updated_metric.txt", "a+") as file:
            
        #     file.write("Teacher Angle: \n")
        #     for element in t_angle:
        #         file.write(str(element) + "\n")
                
        #     file.write("Student Angle: \n")
        #     for element in s_angle:
        #         file.write(str(element) + "\n")
        #     file.write("With a loss of: " + str(loss.item()) + "\n")
        # file.close()

        #---------------------------------------------------------------------------------------------------------------------------------
        
        #sd = (student.unsqueeze(0) - student.unsqueeze(1))
        #norm_sd = F.normalize(sd, p=2, dim=2)
        #s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        #loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss

class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss
