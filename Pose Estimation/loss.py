import math

import torch
import torch.nn as nn
import constants


# def negative_log_likelihood(mu, x, sigma, sigma_const=0.3):
#     denominator = torch.sqrt((2 * math.pi * torch.square(torch.abs(sigma) + sigma_const)))
#     zscore = ((x - mu) / (torch.abs(sigma) + sigma_const))
#     squared_zscore = torch.square(zscore)
#     probability_density = torch.exp(-0.5 * squared_zscore) / denominator
#     nll = -torch.log(probability_density + 1e-7)
#     return nll


class PoseEstimationLoss(nn.Module):

    def __init__(self):
        super(PoseEstimationLoss, self).__init__()
        self.gaussian_loss = nn.GaussianNLLLoss(eps=0.3)# was 0.3
        self.angle_constant = 10
        self.pos_constant = 5

    def forward(self, predictions, target):
        x_loss = self.gaussian_loss(
            predictions[..., 0:1]*self.pos_constant,
            target[..., 0:1]*self.pos_constant,
            torch.square(predictions[..., 4:5]*self.pos_constant)
        )
        y_loss = self.gaussian_loss(
            predictions[..., 1:2]*self.pos_constant,
            target[..., 1:2]*self.pos_constant,
            torch.square(predictions[..., 5:6]*self.pos_constant)
        )
        cos_yaw_loss = self.gaussian_loss(
            predictions[..., 2:3]*self.angle_constant,
            target[..., 2:3] * self.angle_constant,
            torch.square(predictions[..., 6:7] * self.angle_constant)
        )
        sin_yaw_loss = self.gaussian_loss(
            predictions[..., 3:4]*self.angle_constant,
            target[..., 3:4] * self.angle_constant,
            torch.square(predictions[..., 7:8] * self.angle_constant)
        )

        loss = (
            x_loss + y_loss + cos_yaw_loss + sin_yaw_loss
            #cos_yaw_loss + sin_yaw_loss
            # x_loss + y_loss
        )
        return torch.mean(loss)

# class PoseEstimationLoss(nn.Module):
#
#     def __init__(self):
#         super(PoseEstimationLoss, self).__init__()
#         self.gaussian_loss = nn.GaussianNLLLoss(eps=.1)
#
#     def forward(self, predictions, target):
#         x_loss = negative_log_likelihood(
#             predictions[..., 0:1],
#             target[..., 0:1],
#             torch.square(predictions[..., 4:5])
#         )
#         y_loss = negative_log_likelihood(
#             predictions[..., 1:2],
#             target[..., 1:2],
#             torch.square(predictions[..., 5:6])
#         )
#         cos_yaw_loss = negative_log_likelihood(
#             predictions[..., 2:3]*10,
#             target[..., 2:3] * 10,
#             torch.square(predictions[..., 6:7]*10)
#         )
#         sin_yaw_loss = negative_log_likelihood(
#             predictions[..., 3:4]*10,
#             target[..., 3:4] * 10,
#             torch.square(predictions[..., 7:8]*10)
#         )
#
#         loss = (
#             x_loss + y_loss + cos_yaw_loss + sin_yaw_loss
#             #cos_yaw_loss + sin_yaw_loss
#             # x_loss + y_loss
#         )
#         return torch.mean(loss)

# class PoseEstimationLoss(nn.Module):
#
#     def __init__(self):
#         super(PoseEstimationLoss, self).__init__()
#         self.mse = nn.L1Loss() #nn.MSELoss()
#
#     def forward(self, predictions, target):
#         x_loss = self.mse(
#             predictions[..., 0:1],
#             target[..., 0:1],
#         )
#         y_loss = self.mse(
#             predictions[..., 1:2],
#             target[..., 1:2],
#         )
#         cos_yaw_loss = self.mse(
#             predictions[..., 2:3] * 10,
#             target[..., 2:3] * 10,
#         )
#         sin_yaw_loss = self.mse(
#             predictions[..., 3:4] * 10,
#             target[..., 3:4] * 10,
#         )
#
#         loss = (
#             x_loss + y_loss + cos_yaw_loss + sin_yaw_loss
#             # cos_yaw_loss + sin_yaw_loss
#             # x_loss + y_loss
#         )
#         return loss