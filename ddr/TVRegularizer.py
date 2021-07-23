import torch
import torch.nn.functional

class TVRegularizerL2(torch.nn.Module): # l2

    def __init__(self):
        super(TVRegularizerL2, self).__init__()
        Kx = torch.Tensor([[0, 0, 0],
                          [0, -1, 1],
                          [0, 0, 0]])

        Ky = torch.Tensor([[0, 0, 0],
                           [0, -1, 0],
                           [0, 1, 0]])

        self.K = torch.stack([Kx, Ky]).unsqueeze(1)

    def forward(self, x):

        return self.grad(x)

    def energy(self, x):
        # was ist energy:
        # positiv, gibt irgendwo 0,
        # i think here nabla x?
        _,_,h,w = x.shape
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), "constant", 0) #  # zero pad
        Kx = torch.nn.functional.conv2d(padded, self.K, padding=0)
        E = (Kx**2).sum(dim=1,keepdim=True)
        self.grad_E = Kx
        # self.fun.get_energy(Kx)
        # self.fun.get_grad()
        return E


    def grad(self, x):
        # i think i need to do here: nabla.T nabla x?
        E = self.energy(x) # already computes gradients
        #print(f'{E.shape=}')

        # conv_transpose does 0 padding
        cT = torch.nn.functional.conv_transpose2d(self.grad_E, self.K, padding=1)
        return cT

    def get_theta(self):
        return self.named_parameters()

    def get_vis(self):
        raise NotImplementedError

class TVRegularizerL1(torch.nn.Module): # l2
 # https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FProximal-operator-and-Moreaus-envelop-of-the-absolute-function-Proximal-operator-of_fig3_317013217&psig=AOvVaw0pTk02A6QTWSlBlN4BmZnp&ust=1627141244838000&source=images&cd=vfe&ved=0CAoQjRxqFwoTCMCvg9HD-fECFQAAAAAdAAAAABAJ

    def __init__(self):
        super(TVRegularizerL1, self).__init__()
        Kx = torch.Tensor([[0, 0, 0],
                          [0, -1, 1],
                          [0, 0, 0]])

        Ky = torch.Tensor([[0, 0, 0],
                           [0, -1, 0],
                           [0, 1, 0]])

        self.K = torch.stack([Kx, Ky]).unsqueeze(1)

    def forward(self, x):

        return self.grad(x)

    def energy(self, x):
        # was ist energy:
        # positiv, gibt irgendwo 0,
        # i think here nabla x?
        _,_,h,w = x.shape
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), "constant", 0) #  # zero pad
        Kx = torch.nn.functional.conv2d(padded, self.K, padding=0)
        E = (Kx).abs().sum(dim=1, keepdim=True)
        self.grad_E = torch.sign(Kx)*( (Kx!=0).float() )
        # self.fun.get_energy(Kx)
        # self.fun.get_grad()
        return E


    def grad(self, x):
        # i think i need to do here: nabla.T nabla x?
        E = self.energy(x) # already computes gradients
        #print(f'{E.shape=}')

        # conv_transpose does 0 padding
        cT = torch.nn.functional.conv_transpose2d(self.grad_E, self.K, padding=1)
        return cT

    def get_theta(self):
        return self.named_parameters()

    def get_vis(self):
        raise NotImplementedError