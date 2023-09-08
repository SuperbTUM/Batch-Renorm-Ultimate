import torch
from batch_renorm2d import BatchRenormalization2D


class BatchRenormalization2D_Noniid(BatchRenormalization2D):
    """Dedicated for metric learning where sampling is non-iid"""
    def __init__(self,
                 num_features,
                 num_instance,
                 dict_state=None,
                 eps=1e-05,
                 momentum=0.1,
                 r_d_max_inc_step=1e-5,
                 r_max=1.0,
                 d_max=0.0,
                 inference_statistics=0.):
        """
        In metric learning, the sampling will be non-iid
        :param num_features: channel number
        :param num_instance: used in metric learning, number of labels
        :param dict_state: load from pretrained batch normalization
        :param eps: minor offset for stable purpose
        :param momentum: used to update running average and variance
        :param r_d_max_inc_step: used to update r_max and d_max
        :param r_max: hyper-parameter
        :param d_max: hyper-parameter
        :param inference_statistics: use for inference stage
        """
        super(BatchRenormalization2D_Noniid, self).__init__(num_features, dict_state, eps, momentum, r_d_max_inc_step, r_max, d_max)
        self.num_instance = num_instance
        self.inference_statistics = inference_statistics

    def forward(self, x):
        """
        This is designated to metric learning and there are a few ways of samplings
        This example is based on the sampling of [a, a, a, a, b, b, b, b, c, c, c, c...]
        :param x:
        :return:
        """
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_eval(x)

    def _forward_train(self, x):
        """Looks like group normalization"""
        minibatch_size = x.size(0) // self.num_instance
        x_splits = []
        for i in range(self.num_instance):
            x_split = x[i:len(x):self.num_instance]
            x_splits.append(x_split)
        x_splits = torch.cat(x_splits, dim=0)

        x_normed = torch.zeros_like(x)

        batch_ch_mean = torch.mean(x_splits, dim=(2, 3), keepdim=True)
        batch_ch_var_pre = torch.mean(x_splits ** 2, dim=(2, 3), keepdim=True)

        x_splits = x_splits.view(self.num_instance, minibatch_size, self.num_features, x.size(2), x.size(3))

        batch_ch_mean = batch_ch_mean.view(self.num_instance, minibatch_size, self.num_features, 1, 1)
        batch_ch_var_pre = batch_ch_var_pre.view(self.num_instance, minibatch_size, self.num_features, 1, 1)

        group_ch_mean = batch_ch_mean.mean(dim=1, keepdim=True)
        group_ch_var_pre = batch_ch_var_pre.mean(dim=1, keepdim=True)
        group_ch_var_biased = group_ch_var_pre - group_ch_mean ** 2

        r = torch.clamp(torch.sqrt((group_ch_var_biased + self.eps) / (self.running_avg_var.unsqueeze(0) + self.eps)),
                        1.0 / self.r_max, self.r_max).data
        d = torch.clamp((group_ch_mean - self.running_avg_mean.unsqueeze(0)) / torch.sqrt(self.running_avg_var.unsqueeze(0) + self.eps),
                        -self.d_max,
                        self.d_max).data

        x_splits = ((x_splits - group_ch_mean) * r) / torch.sqrt(group_ch_var_biased + self.eps) + d
        x_splits = self.gamma * x_splits + self.beta

        self.num_tracked_batch += 1
        if self.num_tracked_batch > 500 and self.r_max < self.max_r_max:
            self.r_max += 1.2 * self.r_max_inc_step * x.shape[0]

        if self.num_tracked_batch > 500 and self.d_max < self.max_d_max:
            self.d_max += 4.8 * self.d_max_inc_step * x.shape[0]

        x_splits = x_splits.view(-1, x_splits.size(2), x_splits.size(3), x_splits.size(4))

        indices = torch.arange(0, x.size(0))
        x_normed[self.num_instance * (indices % minibatch_size) + indices // minibatch_size] = x_splits[:]

        batch_ch_mean = batch_ch_mean.view(-1, self.num_features, 1, 1)
        batch_ch_var_pre = batch_ch_var_pre.view(-1, self.num_features, 1, 1)
        batch_ch_var_biased = (batch_ch_var_pre - batch_ch_mean ** 2)

        self.running_avg_mean = self.running_avg_mean + self.momentum * (
                    batch_ch_mean.mean(dim=0, keepdim=True).data - self.running_avg_mean)
        self.running_avg_var = self.running_avg_var + self.momentum * (
                    batch_ch_var_biased.mean(dim=0, keepdim=True).data - self.running_avg_var)

        return x_normed

    def _forward_eval(self, x):
        batch_ch_mean = torch.mean(x, dim=(2, 3), keepdim=True)
        batch_ch_var_pre = torch.mean(x ** 2, dim=(2, 3), keepdim=True)
        batch_ch_var_biased = (batch_ch_var_pre - batch_ch_mean ** 2)

        running_avg_mean = (1 - self.inference_statistics) * self.running_avg_mean + self.inference_statistics * batch_ch_mean
        running_avg_var = (1 - self.inference_statistics) * self.running_avg_var + self.inference_statistics * batch_ch_var_biased
        x = (x - running_avg_mean) / torch.sqrt(running_avg_var + self.eps)
        x = self.gamma * x + self.beta
        return x
