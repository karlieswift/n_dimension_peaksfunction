"""
 Env: /anaconda3/python3.7
 Time: 2021/9/7 15:21
 Author: karlieswfit
 File: peaksfunc.py
 Describe: 遗传算法  适者生存主要体现在谁的概率大
 1-编码translate_DNA
 2-选择一个适应度get_fit
 3-交叉与变异 繁殖后代 cross
 4-适者生存 select
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GA_Model:
    def __init__(self,function,section_x, section_y,variation=0.6,DAN_len=20,Number=200,n=200):
        self.function=function #目标函数
        self.DAN_len=DAN_len  #基因长度(x和y的总编码长度)
        self.Number=Number #群体数量
        self.section_x=section_x #规定的x区间
        self.section_y=section_y #规定的y区间
        self.variation=variation
        self.n=n #迭代次数

    # section = [
    #     [-2, -1, 0, 1],
    #     [0, 1, -2, -1],
    #     [0, 1, 0, 1],
    #     [-1, 0, -1, 0],
    #     [1, 2, -1, 0.5],
    #     [-0.5, 0.5, 1, 2]
    # ][1.1759530791788857, 0.09384164222873892, 3.4650308586797203]

    def translate_DNA(self,X_Y):
        i = int(self.DAN_len / 2)
        X = np.array(X_Y[:, 0:i])  # 表示X=[0,1,0,1,1....]
        Y = np.array(X_Y[:, i:self.DAN_len])  # 表示y
        m = np.array(2 ** np.arange(i)[::-1])
        X = np.dot(X, m)  # X十进制
        Y = np.dot(Y, m)  # X十进制

        # 将X,Y的十进制等比例映射到指定区间 （例如132/1024=x/区间大小）
        # X的真实值
        # print(self.section_x)
        X = X * (self.section_x[1] - self.section_x[0]) / (2 ** i-1) + self.section_x[0]
        Y = Y * (self.section_y[1] - self.section_y[0]) / (2 ** i-1) + self.section_y[0]

        return X, Y

    def get_max(self,X_Y):
        #1-返回基因(x,y)对应的z，因为目标是求z的最大值，通所占比大小进行选择
        #2-返回peaks当前最优值[x,y,z]
        x, y = self.translate_DNA(X_Y)
        z = self.function(x, y)
        max_index = np.argmax(z)
        peaks=[x[max_index], y[max_index],z[max_index]]
        return z - min(z) + 1e-3 ,peaks


    def get_min(self,X_Y):
        #1-返回基因(x,y)对应的z，因为目标是求z的最大值，通所占比大小进行选择
        #2-返回peaks当前最优值[x,y,z]
        x, y = self.translate_DNA(X_Y)
        z = self.function(x, y)
        min_index = np.argmin(z)
        peaks=[x[min_index], y[min_index],z[min_index]]
        return -(z - np.max(z)) + 1e-3 ,peaks


    def cross_variation(self,X_Y):
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


    def select(self,X_Y, Z_score):
        index = np.random.choice(np.arange(self.Number),size=self.Number,replace=True, p=Z_score /Z_score.sum())
        X_Y = np.array(X_Y)
        return X_Y[index]




    def fit(self,c):
        X_Y = np.random.randint(2, size=(self.Number, self.DAN_len * 2))
        for i in range(self.n):
            if "max"==c:
                z_score_DNA,peaks=self.get_max(X_Y)
            else:
                z_score_DNA, peaks = self.get_min(X_Y)
            X_Y=self.select(X_Y, z_score_DNA)
            if i%20==0:
                print("第{index}次迭代的{c}峰值(x,y,z):{peaks}".format(index=i+1,c=c,peaks=peaks))
            X_Y=np.array(self.cross_variation(X_Y)) #产生新的基因
        return peaks





    def show_graph(self,section_x,section_y):

        # 创建 3D 图形对象
        fig = plt.figure()
        ax = Axes3D(fig)
        # 生成数据
        X = np.arange(section_x[0], section_x[1], 0.1)
        Y = np.arange(section_y[0], section_y[1], 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = self.function(X,Y)
        ax.plot_surface(X, Y, Z, cmap="rainbow")
        plt.show()
        plt.contourf(X, Y, Z, 10, alpha=1, cmap='rainbow')
        # add contour lines
        C = plt.contour(X, Y, Z, 50, lw=2)
        # 显示各等高线的数据标签cmap=plt.cm.hot
        plt.clabel(C, inline=True, fontsize=10)
        plt.show()



def fun(x,y):
    return 3 * (1 - x) ** 2 * np.exp(-x ** 2 - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 3) * np.exp(
        -x ** 2 - y ** 2) - np.exp(-(x + 1) ** 2 - y ** 2) / 3


# 指定6个区间
section=[
    [-2,-1,0,1],
    [0,1,-2,-1],
    [0,1,0,1],
    [-1,0,-1,0],
    [1,2,-1,0.5],
    [-0.5,0.5,1,2]
]


if __name__ == '__main__':

    #观察图形和等高线 从而得出极值点的个数和大致区间
    # model=GA_Model(function=fun,section_x=section_x,section_y=section_y)
    # model.show_graph([-3,3],[-3,3])

    # ==========极小值=================
    min_peaks=[]
    section_1=section[:3]
    for i in range(len(section_1)):
        model = GA_Model(function=fun, section_x=section_1[i][0:2], section_y=section_1[i][2:4],n=1000)
        peaks=model.fit("min")
        min_peaks.append(peaks)
    # model.show_graph([-3, 3], [-3, 3])

    # ==========极大值=================
    max_peaks=[]
    section_2=section[3:]
    for i in range(len(section_2)):
        model = GA_Model(function=fun, section_x=section_2[i][0:2], section_y=section_2[i][2:4],n=1000)
        peaks=model.fit("max")
        max_peaks.append(peaks)

    print("极小值点[x,y,z]:")
    for i in range(len(min_peaks)):
        print(min_peaks[i])

    print("极大值点[x,y,z]:")
    for i in range(len(min_peaks)):
        print(max_peaks[i])




'''
output:
极小值点[x,y,z]:
[-1.3225806451612905, 0.22678396871945258, -3.0259359944144597]
[0.41642228739002934, -1.2394916911045943, -2.6635957766840432]
[0.18572825024437928, 0.16226783968719452, 0.1691106399845125]
极大值点[x,y,z]:
[-0.4486803519061584, -0.5347018572825024, 3.0182451099978813]
[1.2404692082111437, 0.0014662756598240456, 3.576069040825659]
[0.01319648093841641, 1.2150537634408602, 4.085981828587517]
'''









