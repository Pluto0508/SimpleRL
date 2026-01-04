class dist:
    def __init__(self):
        self.dist=None
        self.dist_transformed=None
        self._last_dist=self.dist
    
    def sample(self,size):
        action=self.dist.sample(size)
        log_prob=self.dist.log_prob(action).sum(-1)
        return {
            'action':action,
            'log_prob':log_prob
        }

    def transform_data(self,data,a=None,b=None,method=None):
        pass
    
