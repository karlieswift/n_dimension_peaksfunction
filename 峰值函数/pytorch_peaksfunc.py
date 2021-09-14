"""
 Env: /anaconda3/python3.7
 Time: 2021/9/8 9:30
 Author: karlieswfit
 File: pytorch_peaksfunc.py
 Describe: 
"""

import torch


class SGD_Peaks:
    def __init__(self, function, start_point, lr=0.001):
        self.function = function  #目标函数
        self.start_point = start_point #起始点[x,y]
        self.lr = lr #学习率

    def gradient_ascend(self, max_iter):
        peaks = ()
        x = self.start_point[0]
        y = self.start_point[1]
        x.requires_grad = True
        y.requires_grad = True
        for i in range(max_iter):
            z = self.function(x, y)
            z.backward()
            x.data += self.lr * x.grad
            y.data += self.lr * y.grad
            if torch.sqrt(x.grad.data**2+y.grad.data**2)<0.0001:
                break
            x.grad.data.zero_()
            y.grad.data.zero_()
            peaks = (x.item(), y.item(), z.item())
            if i%100==0:
                print("第{index}次迭代的{c}峰值(x,y,z):{peaks}".format(index=i + 1, c="max", peaks=peaks))
        return peaks

    def gradient_descent(self, max_iter):
        peaks = ()
        x = self.start_point[0]
        y = self.start_point[1]
        x.requires_grad = True
        y.requires_grad = True
        for i in range(max_iter):
            z = self.function(x, y)
            z.backward()
            x.data -= self.lr * x.grad
            y.data -= self.lr * y.grad
            if torch.sqrt(x.grad.data**2+y.grad.data**2)<0.0001:
                break
            x.grad.data.zero_()
            y.grad.data.zero_()
            peaks = (x.item(), y.item(), z.item())
            if i%100==0:
                print("第{index}次迭代的{c}峰值(x,y,z):{peaks}".format(index=i + 1, c="min", peaks=peaks))

        return peaks

    def fit(self, max_iter=200, c="min"):
        #参数c 极大值max和极小值min
        result = ()
        if c == "min":
            result = self.gradient_descent(max_iter=max_iter)
        else:
            result = self.gradient_ascend(max_iter=max_iter)
        return result



def fun(x, y):
    return 3 * (1 - x) ** 2 * torch.exp(-x ** 2 - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 3) * torch.exp(
        -x ** 2 - y ** 2) - torch.exp(-(x + 1) ** 2 - y ** 2) / 3


x_y = torch.tensor([
    [-1, 0],
    [0.5, -1],
    [0, 0],
    [-0.5, -1],
    [2, 0],
    [0, 2]
])

if __name__ == '__main__':

    # 观察图形和等高线 从而得出极值点的个数和大致区间
    # model=GA_Model(function=fun,section_x=section_x,section_y=section_y)
    # model.show_graph([-3,3],[-3,3])

    # ==========极小值=================
    min_peaks = []
    x_y_1 = x_y[:3]
    for i in range(len(x_y_1)):
        model = SGD_Peaks(function=fun,start_point=x_y_1[i],lr=0.001)
        peaks = model.fit(max_iter=2000,c="min")
        min_peaks.append(peaks)
    # model.show_graph([-3, 3], [-3, 3])

    # ==========极大值=================
    max_peaks = []
    x_y_2 = x_y[3:]
    for i in range(len(x_y_2)):
        model = SGD_Peaks(function=fun,start_point=x_y_2[i],lr=0.001)
        peaks = model.fit(max_iter=1000,c="max")
        max_peaks.append(peaks)

    print("极小值点[x,y,z]:")
    for i in range(len(min_peaks)):
        print(min_peaks[i])

    print("极大值点[x,y,z]:")
    for i in range(len(min_peaks)):
        print(max_peaks[i])

"""
极小值点[x,y,z]:
(-1.3519119024276733, 0.1874384582042694, -3.0386404991149902)
(0.41418319940567017, -1.2488884925842285, -2.6641674041748047)
(0.31566673517227173, 0.15940694510936737, 0.030224647372961044)
极大值点[x,y,z]:
(-0.45039650797843933, -0.5195642709732056, 3.0199780464172363)
(1.2856884002685547, -0.004839953035116196, 3.592489719390869)
(-0.05086343362927437, 1.2194403409957886, 4.104348182678223)

"""
