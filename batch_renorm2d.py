import torch
import torch.nn as nn


class BatchRenormalization2D(nn.Module):
    """
    This reimplementation is a bit tricky
    Originally the r_max and d_max are quickly converged
    But it should not be according to the paper (~1/4 of training process)
    """

    def __init__(self,
                 num_features,
                 dict_state=None,
                 eps=1e-05,
                 momentum=0.1,
                 r_d_max_inc_step=1e-5,
                 r_max=1.0,
                 d_max=0.0):
        super(BatchRenormalization2D, self).__init__()

        self.eps = eps
        self.momentum = torch.tensor(momentum)

        self.gamma = nn.Parameter(torch.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, num_features, 1, 1)), requires_grad=True)

        self.register_buffer('running_avg_mean', torch.zeros((1, num_features, 1, 1)))
        self.register_buffer('running_avg_var', torch.ones((1, num_features, 1, 1)))
        self.register_buffer('num_tracked_batch', torch.tensor(0)) # in case momentum is None

        self.max_r_max = 3.0
        self.max_d_max = 5.0

        self.r_max_inc_step = r_d_max_inc_step
        self.d_max_inc_step = r_d_max_inc_step

        self.register_buffer('r_max', torch.tensor(r_max))
        self.register_buffer('d_max', torch.tensor(d_max))

        self.dict_state = dict_state
        self._load_params_from_bn()

    def _load_params_from_bn(self):
        if self.dict_state is None:
            return
        weight = self.dict_state['weight'].data
        weight = weight.reshape(1, weight.size(0), 1, 1)
        bias = self.dict_state['bias'].data
        bias = bias.reshape(1, bias.size(0), 1, 1)
        running_mean = self.dict_state['running_mean'].data
        running_mean = running_mean.reshape(1, running_mean.size(0), 1, 1)
        running_var = self.dict_state['running_var'].data
        running_var = running_var.reshape(1, running_var.size(0), 1, 1)

        self.gamma.data = weight.clone()
        self.beta.data = bias.clone()
        self.running_avg_mean.data = running_mean.clone()
        self.running_avg_var.data = running_var.clone()

    def forward(self, x):

        batch_ch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        # in version 2.0: correction, otherwise: unbiased=False
        batch_ch_var_unbiased = torch.var(x, dim=(0, 2, 3), unbiased=True, keepdim=True)
        batch_ch_var_biased = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)

        if self.training:
            self.num_tracked_batch += 1
            r = torch.clamp(torch.sqrt((batch_ch_var_biased + self.eps) / (self.running_avg_var + self.eps)), 1.0 / self.r_max, self.r_max).data
            d = torch.clamp((batch_ch_mean - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps), -self.d_max, self.d_max).data

            x = ((x - batch_ch_mean) * r) / torch.sqrt(batch_ch_var_biased + self.eps) + d
            x = self.gamma * x + self.beta

            if self.num_tracked_batch > 5000 and self.r_max < self.max_r_max:
                # This should stay flexible
                self.r_max += 0.5 * self.r_max_inc_step * x.shape[0]

            if self.num_tracked_batch > 5000 and self.d_max < self.max_d_max:
                # This should stay flexible
                self.d_max += 2 * self.d_max_inc_step * x.shape[0]

            self.running_avg_mean = self.running_avg_mean + self.momentum * (batch_ch_mean.data - self.running_avg_mean)
            self.running_avg_var = self.running_avg_var + self.momentum * (batch_ch_var_unbiased.data - self.running_avg_var)

        else:
            x = (x - self.running_avg_mean) / torch.sqrt(self.running_avg_var + self.eps)
            x = self.gamma * x + self.beta

        return x
