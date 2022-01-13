import numpy as np
import torch
import torch.nn as nn
from numpy.random import RandomState

from .Model import Model


class QubitE(Model):
    def __init__(self, config):
        super(QubitE, self).__init__(config)
        self.entity_a = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.entity_ai = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.entity_b = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.entity_bi = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.rel_a = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_ai = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_b = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_bi = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_w = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.criterion = nn.Softplus()
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = torch.nn.Dropout(self.config.ent_dropout)
        self.rel_dropout = torch.nn.Dropout(self.config.rel_dropout)
        self.bn = torch.nn.BatchNorm1d(self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        if True:
            r, i, j, k = self.quaternion_init(self.config.entTotal, self.config.hidden_size)
            r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
            self.entity_a.weight.data = r.type_as(self.entity_a.weight.data)
            self.entity_ai.weight.data = i.type_as(self.entity_ai.weight.data)
            self.entity_b.weight.data = j.type_as(self.entity_b.weight.data)
            self.entity_bi.weight.data = k.type_as(self.entity_bi.weight.data)

            s, x, y, z = self.quaternion_init(self.config.entTotal, self.config.hidden_size)
            s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
            self.rel_a.weight.data = s.type_as(self.rel_a.weight.data)
            self.rel_ai.weight.data = x.type_as(self.rel_ai.weight.data)
            self.rel_b.weight.data = y.type_as(self.rel_b.weight.data)
            self.rel_bi.weight.data = z.type_as(self.rel_bi.weight.data)
            nn.init.xavier_uniform_(self.rel_w.weight.data)
        else:
            nn.init.xavier_uniform_(self.entity_a.weight.data)
            nn.init.xavier_uniform_(self.entity_ai.weight.data)
            nn.init.xavier_uniform_(self.entity_b.weight.data)
            nn.init.xavier_uniform_(self.entity_bi.weight.data)
            nn.init.xavier_uniform_(self.rel_a.weight.data)
            nn.init.xavier_uniform_(self.rel_ai.weight.data)
            nn.init.xavier_uniform_(self.rel_b.weight.data)
            nn.init.xavier_uniform_(self.rel_bi.weight.data)

    def _calc(self,
              h1, h2, h3, h4,
              t1, t2, t3, t4,
              r1, r2, r3, r4
              ):
        # h = (h1 + h2 i) |0> + (h3 + h4 i) |1>
        # t = (t1 + t2 i) |0> + (t3 + t4 i) |1>
        # r = (h1 + h2 i) |0> + (h3 + h4 i) |1>

        denominator_b = torch.sqrt(r1 ** 2 + r2 ** 2 + r3 ** 2 + r4 ** 2)
        r1 = r1 / denominator_b
        r2 = r2 / denominator_b
        r3 = r3 / denominator_b
        r4 = r4 / denominator_b

        A = h1 * r1 - h2 * r2 - h3 * r3 - h4 * r4
        B = h1 * r2 + r1 * h2 + h3 * r4 - r3 * h4
        C = h1 * r3 + r1 * h3 + h4 * r2 - r4 * h2
        D = h1 * r4 + r1 * h4 + h2 * r3 - r2 * h3

        score_r = (A * t1 + B * t2 + C * t3 + D * t4)
        # print(score_r.size())
        # score_i = A * x_c + B * s_c + C * z_c - D * y_c
        # score_j = A * y_c - B * z_c + C * s_c + D * x_c
        # score_k = A * z_c + B * y_c - C * x_c + D * s_c
        return -torch.sum(score_r, -1)

    def loss(self, score, regul, regul2):
        # self.batch_y = ((1.0-0.1)*self.batch_y) + (1.0/self.batch_y.size(1)) /// (1 + (1 + self.batch_y)/2) * 
        return (
                torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul + self.config.lmbda * regul2
        )

    def forward(self):
        h1 = self.entity_a(self.batch_h)
        h2 = self.entity_ai(self.batch_h)
        h3 = self.entity_b(self.batch_h)
        h4 = self.entity_bi(self.batch_h)

        t1 = self.entity_a(self.batch_t)
        t2 = self.entity_ai(self.batch_t)
        t3 = self.entity_b(self.batch_t)
        t4 = self.entity_bi(self.batch_t)

        r1 = self.rel_a(self.batch_r)
        r2 = self.rel_ai(self.batch_r)
        r3 = self.rel_b(self.batch_r)
        r4 = self.rel_bi(self.batch_r)

        score = self._calc(h1, h2, h3, h4, t1, t2, t3, t4, r1, r2, r3, r4)
        regul = (torch.mean(torch.abs(h1) ** 2)
                 + torch.mean(torch.abs(h2) ** 2)
                 + torch.mean(torch.abs(h3) ** 2)
                 + torch.mean(torch.abs(h4) ** 2)
                 + torch.mean(torch.abs(t1) ** 2)
                 + torch.mean(torch.abs(t2) ** 2)
                 + torch.mean(torch.abs(t3) ** 2)
                 + torch.mean(torch.abs(t4) ** 2)
                 )
        regul2 = (torch.mean(torch.abs(r1) ** 2)
                  + torch.mean(torch.abs(r2) ** 2)
                  + torch.mean(torch.abs(r3) ** 2)
                  + torch.mean(torch.abs(r4) ** 2))

        '''
        + torch.mean(s_b ** 2)
            + torch.mean(x_b ** 2)
            + torch.mean(y_b ** 2)
            + torch.mean(z_b ** 2)
        '''

        return self.loss(score, regul, regul2)

    def predict(self):
        h1 = self.entity_a(self.batch_h)
        h2 = self.entity_ai(self.batch_h)
        h3 = self.entity_b(self.batch_h)
        h4 = self.entity_bi(self.batch_h)

        t1 = self.entity_a(self.batch_t)
        t2 = self.entity_ai(self.batch_t)
        t3 = self.entity_b(self.batch_t)
        t4 = self.entity_bi(self.batch_t)

        r1 = self.rel_a(self.batch_r)
        r2 = self.rel_ai(self.batch_r)
        r3 = self.rel_b(self.batch_r)
        r4 = self.rel_bi(self.batch_r)

        score = self._calc(h1, h2, h3, h4, t1, t2, t3, t4, r1, r2, r3, r4)
        return score.cpu().data.numpy()

    def quaternion_init(self, in_features, out_features, criterion='he'):

        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(123)

        # Generating randoms and purely imaginary quaternions :
        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        return (weight_r, weight_i, weight_j, weight_k)
