import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LinearEmbedding"]


class LinearEmbedding(nn.Module):
    def __init__(self, base, output_size=512, embedding_size=128, normalize=True):
        super(LinearEmbedding, self).__init__()
        self.base = base
        self.linear = nn.Linear(output_size, embedding_size)
        self.normalize = normalize

    def forward(self, x, get_ha=False):
        if get_ha:
        # 这几个值来自与Resnet50 的各层输出结果
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # b1 = self.layer1(x)
        # b2 = self.layer2(b1)
        # b3 = self.layer3(b2)
        # b4 = self.layer4(b3)
        # pool = self.avgpool(b4)

            b1, b2, b3, b4, pool = self.base(x, True)
        else:
            pool = self.base(x)

        pool = pool.view(x.size(0), -1)
        embedding = self.linear(pool)   

        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1) #求范数距离   即l2-norm , https://mathworld.wolfram.com/L2-Norm.html

        if get_ha:
            return b1, b2, b3, b4, pool, embedding
        # 返回的是Resnet50 处理后，再经过normalize处理过的特征图
        return embedding
