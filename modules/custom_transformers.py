import torch

class ToYCbYr(object):
    rgb_to_ycbyr_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.5],
        [0.5, -0.419, -0.081]
    ])
    
    def __call__(self, pic):
        return torch.einsum('ij,jkl->ikl', [ToYCbYr.rgb_to_ycbyr_matrix, pic])
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

