import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.distributions import Normal, Independent, kl
import numpy as np
import matplotlib.pyplot as plt

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

class SPSNet(nn.Module):
    def __init__(self):
        super(SPSNet, self).__init__()
        self.get_mu = nn.Sequential(
            nn.Linear(2,8,bias=False),
            nn.ReLU(),
            nn.Linear(8,2,bias=False)
        )

        self.get_logvar = nn.Sequential(
            nn.Linear(2,8,bias=False),
            nn.ReLU(),
            nn.Linear(8,2,bias=False)
        )

        self.gene = nn.Sequential(
            nn.Linear(4,16,bias=False),
            nn.ReLU(),
            nn.Linear(16,2,bias=False)
        )

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div
    
    def get_training_loss(self, gt_center):

        loss_reg = torch.mean(F.smooth_l1_loss(self.center_pred, gt_center))

        normal_dist = Independent(Normal(loc=torch.zeros_like(self.mu), scale=torch.ones_like(self.logvar)), 1)
        kl_loss = torch.mean(self.kl_divergence(self.dist, normal_dist))

        L2_loss = l2_regularisation(self.get_mu) + l2_regularisation(self.get_logvar) + l2_regularisation(self.gene)

        loss = loss_reg + kl_loss*5e-5 + L2_loss*5e-5

        return loss




    def forward(self, features, gt):
        self.mu = self.get_mu(features)
        self.logvar = self.get_logvar(features)
        self.dist = Independent(Normal(loc=self.mu, scale=torch.exp(self.logvar)+3e-22), 1)

        z_noise_prior = self.reparametrize(self.mu, self.logvar)

        self.center_pred  = self.gene(torch.cat([features, z_noise_prior], dim = -1)).view(-1,2)

        loss = self.get_training_loss(gt)
        return loss, self.logvar
    

def draw_3d(fig_id):

    fig = plt.figure(fig_id)  #定义新的三维坐标轴
    ax3 = plt.axes(projection='3d')

    #定义三维数据
    xx = np.arange(0,1,0.01)
    yy = np.arange(0,1,0.01)
    X, Y = np.meshgrid(xx, yy)
    Z = np.sin(X)+np.cos(Y)

    #作图
    ax3.plot_surface(X,Y,Z,cmap='rainbow')
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    plt.show()

def draw_scatter(fig_id,x,y,color = 'r',alpha = 1):

    fig = plt.figure(fig_id)  #定义新的三维坐标轴

    # 定义数据
    # x = np.random.rand(10)  # 取出10个随机数
    # y = x + x ** 2 - 10  # 用自定义关系确定y的值
    # y = np.random.rand(10)

    # 绘图
    # 1. 确定画布
    # 2. 绘图
    plt.scatter(x,  # 横坐标
            y,  # 纵坐标
            c=color,
            alpha = alpha)  # 标签 即为点代表的意思

      # 显示所绘图形



if __name__ == "__main__":

    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1'

    data = torch.rand(500,2)
    draw_scatter(0,data.numpy()[:,0],data.numpy()[:,1],alpha = 0.5)

    data1 = torch.randn(100,2)+0.5
    draw_scatter(0,data1.numpy()[:,0],data1.numpy()[:,1],color='blue',alpha = 1)

    plt.savefig('./data.png',dpi = 600)

    data = torch.cat([data, data1], dim = 0)

    # mask1 = (data[:,0] > 0) & (data[:,1] > 0)
    # mask = (data[:,0] < 1) & (data[:,1] < 1)
    # data = data[mask & mask1]

    gt = torch.zeros_like(data)

    model = SPSNet()
    
    iters = 300

    opti = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        data = data.to(device)
        model.cuda()
        gt = gt.cuda(device)
    
    model.train()
    opti.zero_grad()
    for i in range(iters):

        loss,_ = model(data, gt)

        loss.backward()
        opti.step()

        print(loss.item())

        if i == iters-1:
            loss,logvarx = model(data, gt)

            v = torch.sum(logvarx.mul(0.5).exp_(),dim = -1).view(-1,1)

            _,topk = torch.topk(-v, 100,dim=0)
            topk = topk.squeeze()

            top_data_1 = data[topk[topk>500]]
            draw_scatter(1,top_data_1.cpu().numpy()[:,0],top_data_1.cpu().numpy()[:,1],color='blue')

            top_data_2 = data[topk[topk<=500]]
            draw_scatter(1,top_data_2.cpu().numpy()[:,0],top_data_2.cpu().numpy()[:,1])

            print(topk)

            plt.savefig('./topk.png',dpi = 600)

            plt.show()

        
        
