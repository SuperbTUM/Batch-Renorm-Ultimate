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
                 d_max=0.0):
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
        """
        super(BatchRenormalization2D_Noniid, self).__init__(num_features, dict_state, eps, momentum, r_d_max_inc_step, r_max, d_max)
        self.num_instance = num_instance

    def forward(self, x):
        if not self.training:
            self.num_instance = 1 # Let it be iid in inference mode
        x_splits = []
        for i in range(self.num_instance):
            x_split = []
            for j in range(x.size(0) // self.num_instance):
                x_split.append(x[i+self.num_instance*j])
            x_splits.append(torch.stack(x_split))
        x_normed = [torch.tensor(0.) for _ in range(x.size(0))]

        for i, x_mini in enumerate(x_splits):

            batch_ch_mean = torch.mean(x_mini, dim=(0, 2, 3), keepdim=True)
            # in version 2.0: correction, otherwise: unbiased=False
            batch_ch_var_unbiased = torch.var(x_mini, dim=(0, 2, 3), unbiased=True, keepdim=True)
            batch_ch_var_biased = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)

            if self.training:
                r = torch.clamp(torch.sqrt(batch_ch_var_biased / self.running_avg_var), 1.0 / self.r_max, self.r_max).data
                d = torch.clamp((batch_ch_mean - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps), -self.d_max,
                                self.d_max).data

                x_mini = ((x_mini - batch_ch_mean) * r) / torch.sqrt(batch_ch_var_biased + self.eps) + d
                x_mini = self.gamma * x_mini + self.beta

                self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
                self.running_avg_var = self.running_avg_var + self.momentum * (batch_ch_var_unbiased.data - self.running_avg_var)

            else:
                x_mini = (x_mini - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps)
                x_mini = self.gamma * x_mini + self.beta

            for j in range(x_mini.size(0)):
                x_normed[self.num_instance * j + i % self.num_instance] = x_mini[j]

        self.num_tracked_batch += 1
        if self.num_tracked_batch > 5000 and self.r_max < self.max_r_max:
            self.r_max += 0.5 * self.r_max_inc_step * x.shape[0]

        if self.num_tracked_batch > 2000 and self.d_max < self.max_d_max:
            self.d_max += 2 * self.d_max_inc_step * x.shape[0]

        x_normed = torch.stack(x_normed, dim=0)
        return x_normed
