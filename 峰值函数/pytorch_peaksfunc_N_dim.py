"""
 Env: /anaconda3/python3.7
 Time: 2021/9/8 9:30
 Author: karlieswfit
 File: pytorch_peaksfunc_N_dim.py
 Describe:  传入任意维度 多元函数 给定初始点 求得极值
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SGD_Peaks:
    def __init__(self, function, start_point, lr=0.001):
        self.function = function  #目标函数
        self.start_point=start_point
        self.lr = lr #学习率

    def gradient_ascend(self, max_iter):
        x=torch.tensor(self.start_point,dtype=float,requires_grad=True)
        optimizer=torch.optim.Adam([x],lr=self.lr)
        for i in  range(max_iter):
            fun=-self.function(x)
            optimizer.zero_grad()
            fun.backward()
            optimizer.step()
            if i%10==0:
                print("第{index}次迭代的{c}峰值:{peaks}".format(index=i + 1, c="max", peaks=np.append(x.detach().numpy(),[-fun.detach().numpy()])))
        return np.append(x.detach().numpy(),[fun.detach().numpy()])

    def gradient_descent(self, max_iter):
        # x=self.start_point
        x=torch.tensor(self.start_point,dtype=float,requires_grad=True)
        optimizer=torch.optim.SGD([x,],lr=self.lr)
        fun=0
        for i in  range(max_iter):
            fun=self.function(x)
            optimizer.zero_grad()
            fun.backward()
            optimizer.step()
            if i%10==0:
                print("第{index}次迭代的{c}峰值:{peaks}".format(index=i + 1, c="min", peaks=np.append(x.detach().numpy(),[fun.detach().numpy()])))
        return np.append(x.detach().numpy(),[fun.detach().numpy()])


    def fit(self, max_iter=200, c="min"):
        if c == "min":
            result = self.gradient_descent(max_iter=max_iter)
        else:
            result = self.gradient_ascend(max_iter=max_iter)
        return result


    def show_graph_3D(self,section_x,section_y):
        if len(self.start_point) !=2:
            print("必须是2元函数")
            return
        fig = plt.figure()
        ax = Axes3D(fig,auto_add_to_figure = False)
        fig.add_axes(ax)
        X = np.arange(section_x[0], section_x[1], 0.1)
        Y = np.arange(section_y[0], section_y[1], 0.1)
        X, Y = np.meshgrid(X, Y)
        X,Y=torch.from_numpy(X),torch.from_numpy(Y)
        Z = self.function([X,Y])
        X, Y,Z = X.numpy(),Y.numpy(),Z.numpy()
        ax.plot_surface(X, Y, Z, cmap="rainbow")
        plt.show()
        plt.contourf(X, Y, Z, 10, alpha=1, cmap='rainbow')
        C = plt.contour(X, Y, Z, 50)
        plt.clabel(C, inline=True, fontsize=10)
        plt.show()





def fun(x):
    return 3 * (1 - x[0]) ** 2 * torch.exp(-x[0] ** 2 - (x[1] + 1) ** 2) - 10 * (x[0] / 5 - x[0] ** 3 - x[1] ** 3) * torch.exp(-x[0] ** 2 - x[1] ** 2) - torch.exp(-(x[0] + 1) ** 2 - x[1] ** 2) / 3


def fun0(x):
    return x[0]**2


def fun1(x):
    return 10-(x[0]**2+x[1]**2+x[2]**2+x[3]**2)
def fun2(x):
    return 3+(x[0]**2+x[1]**2+x[2]**2+x[3]**2)

if __name__ == '__main__':
    # model = SGD_Peaks(function=fun0, start_point=[100], lr=0.005)
    # peaks = model.fit(max_iter=1000, c="min") #第第991次迭代的min峰值:[4.72582729e-03 2.27869030e-05]

    model = SGD_Peaks(function=fun, start_point=[-1,0], lr=0.005)
    # model.show_graph_3D([-3,3],[-3,3])
    peaks = model.fit(max_iter=1000, c="min") #第991次迭代的min峰值:[-1.3519091   0.18744787 -3.03864071]

    # model = SGD_Peaks(function=fun1, start_point=[2,2,2,2], lr=0.005)
    # peaks = model.fit(max_iter=1000, c="max")   #第991次迭代的max峰值:[0.01319987 0.01319987 0.01319987 0.01319987 9.99929158]

    # model = SGD_Peaks(function=fun2, start_point=[2,2,2,2], lr=0.005)
    # peaks = model.fit(max_iter=1000, c="min") #第991次迭代的min峰值:[9.45165459e-05 9.45165459e-05 9.45165459e-05 9.45165459e-05 3.00000004e+00]



