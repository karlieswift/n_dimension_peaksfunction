"""
 Env: /anaconda3/python3.7
 Time: 2021/9/7 15:21
 Author: karlieswfit
 File: ga_peaksfunc_N_dim.py
 Describe: 传入任意维度 多元函数 给定每个维度的区间 求得极值
 遗传算法  适者生存主要体现在谁的概率大
 1-编码translate_DNA
 2-选择一个适应度get_max 和 get_min
 3-交叉与变异 繁殖后代 cross
 4-适者生存 select

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GA_Model:
    def __init__(self, function, section, dim, DAN_len, variation=0.6, Number=100, n=200):
        self.function = function  # 目标函数
        self.DAN_len = DAN_len  # 基因长度(x1,x2,x3.......的总编码长度)
        self.Number = Number  # 群体数量
        self.section = section  # 规定区间
        self.variation = variation #变异概率
        self.n = n  # 迭代次数
        self.dim = dim  # dim维度=多元函数

    def translate_DNA(self, group):
        """
        :param group: 群体 二进制序列
        :return: 每个个体的目标值
        """

        list = []
        if self.DAN_len % self.dim != 0:
            print("DAN_len必须是dim的整数倍")
            return
        length = self.DAN_len / self.dim
        temp = np.array(2 ** np.arange(0, length)[::-1])
        for i in range(self.Number):
            dan = np.array_split(group[i], self.dim)
            inner_list = []
            for j in range(len(dan)):
                X = np.dot(dan[j], temp)
                X = X * (self.section[j][1] - self.section[j][0]) / (2 ** length - 1) + self.section[j][0]
                inner_list.append(X)
            list.append(inner_list)
        list = np.array(list)
        return list

    def get_max(self, X_Y):
        z_score = []
        list = self.translate_DNA(X_Y)
        for i in range(self.Number):
            z = self.function(list[i])
            z_score.append(z)
        max_index = np.argmax(z_score)
        peaks = [list[max_index], z_score[max_index]]
        return z_score - min(z_score) + 1e-2, peaks

    def get_min(self, X_Y):
        z_score = []
        list = self.translate_DNA(X_Y)
        for i in range(self.Number):
            z = self.function(list[i])
            z_score.append(z)
        min_index = np.argmin(z_score)
        peaks = [list[min_index], z_score[min_index]]
        return np.max(z_score)-z_score + 1e-2, peaks

    def cross_variation(self, X_Y):
        new_DNA = []
        for father in X_Y:
            child = father
            i = np.random.randint(0, self.Number)
            mother = X_Y[i]
            j = np.random.randint(0, self.DAN_len)
            child[:j] = mother[:j]
            if np.random.rand() < self.variation:
                j = np.random.randint(0, self.DAN_len)
                if child[j] == 0:
                    child[j] = 1
                else:
                    child[j] = 0
            new_DNA.append(child)
        return new_DNA

    def select(self, X_Y, Z_score):
        index = np.random.choice(np.arange(self.Number), size=self.Number, replace=True, p=Z_score / Z_score.sum())
        X_Y = np.array(X_Y)
        return X_Y[index]

    def fit(self, c):
        X_Y = np.random.randint(2, size=(self.Number, self.DAN_len))
        for i in range(self.n):
            if "max" == c:
                z_score_DNA, peaks = self.get_max(X_Y)
            else:
                z_score_DNA, peaks = self.get_min(X_Y)
            X_Y = self.select(X_Y, z_score_DNA)
            if i % 20 == 0:
                print("第{index}次迭代的{c}峰值:{peaks}".format(index=i + 1, c=c, peaks=peaks))
            X_Y = np.array(self.cross_variation(X_Y))  # 产生新的基因
        return peaks

    def show_graph_3D(self,section_x,section_y):
        if self.dim !=2:
            print("必须是2元函数")
            return
        fig = plt.figure()
        ax = Axes3D(fig,auto_add_to_figure = False)
        fig.add_axes(ax)
        X = np.arange(section_x[0], section_x[1], 0.1)
        Y = np.arange(section_y[0], section_y[1], 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = self.function([X,Y])
        ax.plot_surface(X, Y, Z, cmap="rainbow")
        plt.show()
        plt.contourf(X, Y, Z, 10, alpha=1, cmap='rainbow')
        C = plt.contour(X, Y, Z, 50)
        plt.clabel(C, inline=True, fontsize=10)
        plt.show()



def fun(x):
    return 3 * (1 - x[0]) ** 2 * np.exp(-x[0] ** 2 - (x[1] + 1) ** 2) - 10 * (
                x[0] / 5 - x[0] ** 3 - x[1] ** 3) * np.exp(-x[0] ** 2 - x[1] ** 2) - np.exp(
        -(x[0] + 1) ** 2 - x[1] ** 2) / 3


def fun0(x):
    return x[0] ** 2


def fun1(list):
    return 100 - (list[0] ** 2 + list[1] ** 2 + list[2] ** 2)


if __name__ == '__main__':
    model = GA_Model(function=fun, section=[[-10,10]],DAN_len=20,dim=2,n=1000)
    model.show_graph_3D([-3,3],[-3,3])
    #
    # model = GA_Model(function=fun0, section=[[-10,10]],DAN_len=10,dim=1,n=1000)
    # model.show_graph_3D([-3,3],[3,3])
    # peaks = model.fit("min")
    '''
第921次迭代的min峰值:[array([-0.02932551]), 0.0008599857242369761]
第941次迭代的min峰值:[array([-0.00977517]), 9.555396935966402e-05]
第961次迭代的min峰值:[array([-0.10752688]), 0.011562030292519346]
第981次迭代的min峰值:[array([-0.02932551]), 0.0008599857242369761]
    '''

    # model = GA_Model(function=fun1, section=[[-20, 10], [-10, 10], [-10, 10]], DAN_len=30, dim=3, n=1000)
    # peaks = model.fit("max")
    ''''
第881次迭代的max峰值:[array([ 1.23167155,  1.9257087 , -0.61583578]), 94.39537748117826]
第901次迭代的max峰值:[array([ 0.46920821,  0.40078201, -0.4398827 ]), 99.42572064414841]
第921次迭代的max峰值:[array([-1.08504399,  0.08797654, -0.26392962]), 98.74528082833825]
第941次迭代的max峰值:[array([ 0.6744868 ,  0.49853372, -0.55718475]), 98.9860768311246]
第961次迭代的max峰值:[array([ 0.20527859,  0.92864125, -0.20527859]), 99.0533468255538]
第981次迭代的max峰值:[array([-0.85043988,  1.51515152, -0.18572825]), 96.94657290911192]
    '''
