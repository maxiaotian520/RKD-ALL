parser.add_argument('--base', choices=dict(googlenet=backbone.GoogleNet, inception_v1bn=backbone.InceptionV1BN, resnet18=backbone.ResNet18, resnet50=backbone.ResNet50),
                    default=backbone.ResNet50,
                    action=LookupChoices)

student_base = opts.base(pretrained=True)  # resnet18
teacher_base = opts.teacher_base(pretrained=False)  # resnet50

teacher_normalize = get_normalize(teacher_base)
student_normalize = get_normalize(student_base)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class LinearEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=128, normalize=True):
        super(LinearEmbedding, self).__init__()
        self.base = base        # resnet18
        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize

    def forward(self, x, get_ha=False):
        if get_ha:
            b1, b2, b3, b4, pool = self.base(x, True)
        else:
            pool = self.base(x)    # resnet18

        pool = pool.view(x.size(0), -1)
        embedding = self.linear(pool)
        print('embedding=====01', embedding)

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1) #求范数距离
            print('embedding=====02', embedding)

        if get_ha:
            return b1, b2, b3, b4, pool, embedding

        return embedding

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
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

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)    # sum(1) 沿纵轴相加
    prod = e @ e.t()                  # @ 矩阵点乘
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res

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

parser.add_argument('--triplet_sample', choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.DistanceWeighted,
                    action=LookupChoices)

class _Sampler(nn.Module):
    def __init__(self, dist_func=pdist):
        self.dist_func = dist_func
        super().__init__()

    def forward(self, embeddings, labels):
        raise NotImplementedError

class DistanceWeighted(_Sampler):
    cut_off = 0.5
    nonzero_loss_cutoff = 1.4
    """
    Distance Weighted loss assume that embeddings are normalized py 2-norm.
    """
    def forward(self, embeddings, labels):
        with torch.no_grad():
            embeddings = F.normalize(embeddings, dim=1, p=2)
            pos_mask, neg_mask = pos_neg_mask(labels)
            pos_pair_idx = pos_mask.nonzero()

            anchor_idx = pos_pair_idx[:, 0]
            pos_idx = pos_pair_idx[:, 1]

            d = embeddings.size(1)
            dist = (pdist(embeddings, squared=True) + torch.eye(embeddings.size(0), device=embeddings.device, dtype=torch.float32)).sqrt()
            dist = dist.clamp(min=self.cut_off)

            log_weight = ((2.0 - d) * dist.log() - ((d - 3.0)/2.0) * (1.0 - 0.25 * (dist * dist)).log())
            
            weight = (log_weight - log_weight.max(dim=1, keepdim=True)[0]).exp()
            weight = weight * (neg_mask * (dist < self.nonzero_loss_cutoff)).float()

            weight = weight + ((weight.sum(dim=1, keepdim=True) == 0) * neg_mask).float()
            weight = weight / (weight.sum(dim=1, keepdim=True))
            weight = weight[anchor_idx]
            neg_idx = torch.multinomial(weight, 1).squeeze(1)

        return anchor_idx, pos_idx, neg_idx

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
--dist_ratio 1  --angle_ratio 2

student = LinearEmbedding(student_base, output_size=student_base.output_size, embedding_size=opts.embedding_size, normalize=opts.l2normalize == 'true')
teacher = LinearEmbedding(teacher_base, output_size=teacher_base.output_size, embedding_size=opts.teacher_embedding_size, normalize=opts.teacher_l2normalize == 'true')

dist_criterion = RkdDistance()
angle_criterion = RKdAngle()
dark_criterion = HardDarkRank(alpha=opts.dark_alpha, beta=opts.dark_beta)
triplet_criterion = L2Triplet(sampler=opts.triplet_sample(), margin=opts.triplet_margin)
at_criterion = AttentionTransfer()

with torch.no_grad():
    t_b1, t_b2, t_b3, t_b4, t_pool, t_e = teacher(teacher_normalize(images), True)

if isinstance(student.base, backbone.GoogleNet):
    assert (opts.at_ratio == 0), "AttentionTransfer cannot be applied on GoogleNet at current implementation."
    e = student(student_normalize(images))
    at_loss = torch.zeros(1, device=e.device)
else:
    b1, b2, b3, b4, pool, e = student(student_normalize(images), True)
    at_loss = opts.at_ratio * (at_criterion(b2, t_b2) + at_criterion(b3, t_b3) + at_criterion(b4, t_b4))

triplet_loss = opts.triplet_ratio * triplet_criterion(e, labels)   # L2Triplet
dist_loss = opts.dist_ratio * dist_criterion(e, t_e)    # RkdDistance()
angle_loss = opts.angle_ratio * angle_criterion(e, t_e) # RKdAngle()
dark_loss = opts.dark_ratio * dark_criterion(e, t_e)    # HardDarkRank

loss = triplet_loss + dist_loss + angle_loss + dark_loss + at_loss

1. 除了角度, 是否还有距离, 怎么计算距离
2. t1, t2, t3 和s1, s2, s3 是怎么计算的