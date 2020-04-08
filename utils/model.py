import torch.nn as nn


def conv(in_planes, planes, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, planes, padding=padding, kernel_size=kernel,
                  stride=stride),
        nn.BatchNorm2d(planes),
        nn.LeakyReLU(inplace=True, negative_slope=0.1)
    )


class SKModel(nn.Module):
    def __init__(self, dropout=0.0):
        super(SKModel, self).__init__()

        self.l1 = conv(12, 32)
        self.l2 = conv(32, 32)
        self.l3 = conv(32, 32)
        self.l4 = conv(32, 32)
        self.l5 = conv(32, 32)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        self.softm = nn.Softmax2d()

    def forward(self, im):
        im_norm = 2 * (im - 0.5)

        im_l3 = self.l3(self.l2(self.l1(im_norm)))

        if self.dropout is not None:
            im_l3 = self.dropout(im_l3)

        im_fin = self.final_conv(self.l5(self.l4(im_l3)))

        if self.training:
            return im_fin
        
        return self.softm(im_fin)


        # Softmax even on training - a feature, not a bug
        # return self.softm(im_fin)
