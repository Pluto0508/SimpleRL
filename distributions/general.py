import torch
from torch import distributions

from distributions.base import dist

class Normal(dist):
    def __init__(self,mu,sig):
        super(Normal,self).__init__()
        self.dist=distributions.Normal(mu,sig)

    def transform_data(self,data,a=None,b=None,method='cdf'):
        if method=='cdf':
            return self.dist.cdf(data)
        elif method=='tanh':
            return torch.tanh(data)
        else:
            raise ValueError('No such method!{cdf,tanh} are available!')

class Beta(dist):
    def __init__(self,alpha,beta):
        super(Beta,self).__init__()
        self.dist=distributions.Beta(alpha,beta)

    def transform_data(self,data,a=None,b=None,method='cdf'):
        if method=='cdf':
            norm_dist=distributions.Normal(0,1)
            return norm_dist(data)
        elif method=='tanh':
            return torch.tanh(data)
        else:
            raise ValueError('No such method!{cdf,tanh} are available!')


