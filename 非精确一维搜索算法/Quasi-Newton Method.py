from time import *
import numpy as np
import math
import random


def inexact_one_dimensional_search(f, x, d):
    """
    基于Wolfe-Powell准则的非精确一维搜索算法
    """
    k = 0
    maxk = 10000  # 迭代上限
    alpha0 = 1
    rho = 0.25
    sigma = 0.5

    phi0 = f(x)
    phi0_ = np.inner(grad(f, x), d)

    a1 = 0.
    a2 = alpha0
    phi1 = phi0
    phi1_ = phi0_
    alpha = a2*0.5

    while k < maxk:
        phi = f(x+alpha*d)
        if phi > phi0+rho*alpha*phi0_:
            alpha1 = a1+(a1-alpha)**2*phi1_*1. / \
                (2*((phi1-phi)-(a1-alpha)*phi1_))
            a2 = alpha
            alpha = alpha1
            continue
        phi_ = np.inner(grad(f, x+alpha*d), d)
        if phi_ >= sigma*phi0_:
            return True, alpha
        alpha1 = alpha-(a1-alpha)*phi_*1./(phi1_-phi_)
        a1 = alpha
        alpha = alpha1
        phi1 = phi
        phi1_ = phi_
        k += 1
    print("非精确一维搜索的迭代次数超过上限！")
    return False, alpha


def quasi_newton_method(f, x):
    """
    BFGS校正的拟牛顿法
    输入：函数f，初始点x
    输出：函数f的极小点x
    """
    n = len(x)
    H = np.zeros([n, n], dtype=float)
    for i in range(n):
        H[i][i] = 1.
    k = 0
    maxk = 10000
    while k < maxk:
        d = np.zeros(n, dtype=float)
        g = grad(f, x)  # 梯度
        d = -1*np.matmul(H, g)  # 计算搜索方向
        judge, alpha = inexact_one_dimensional_search(f, x, d)  # 确定步长因子
        if not judge:
            return x, k+1, False
            break
        tmpx = x.copy()
        x = x+alpha*d  # 更新迭代点
        s = x-tmpx
        if np.linalg.norm(s, ord=2) < 1e-7 or np.linalg.norm(g, ord=2) < 1e-7:
            return x, k+1, True
        y = grad(f, x)-g
        H = bfgs_correction(H, y, s)
        k += 1
    print("拟牛顿法的迭代次数超过上限！")
    return x, k, False


def bfgs_correction(H, y, s):
    """
    对BFGS校正
    """
    alpha = 1./np.inner(s, y)
    beta = (1.+vectors_mcl_with_matrix(H, y)*alpha)*alpha
    M = H+beta*vectors_to_matrix(s, s)-alpha*(
        vectors_to_matrix(np.matmul(H, y), s)+np.matmul(vectors_to_matrix(s, y), H))
    return M


def vectors_to_matrix(x, y):
    """
    计算矩阵x*y^T
    """
    n = len(x)
    H = np.zeros([n, n], dtype=float)
    for i in range(n):
        for j in range(n):
            H[i][j] = x[i]*y[j]
    return H


def vectors_mcl_with_matrix(H, x):
    """
    计算x^T*H*x
    """
    n = len(x)
    y = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            y[i] += H[i][j]*x[j]
    return np.inner(x, y)


def grad(f, x):
    n = len(x)
    h = 1e-10
    g = np.zeros(n, dtype=float)
    for i in range(n):  # 计算所有偏导
        tmp_val = x[i]
        x[i] = tmp_val+h
        fxh1 = f(x)
        x[i] = tmp_val-h
        fxh2 = f(x)
        g[i] = (fxh1-fxh2)*1./(2*h)
        x[i] = tmp_val
    return g


def main():
    l=0
    # 数据集1.
    # x = np.zeros(2, dtype=float) # 初始点
    
    # def f(x):  # 定义函数
    #     ans = 0.
    #     for i in range(1):
    #         ans += 100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2
    #     return ans
    # 数据集2.
    x = np.zeros(6, dtype=float) # 初始点
    
    def f(x):  # 定义函数
        ans = 0.
        for i in range(5):
            ans += 100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2
        return ans
    # 数据集X. 因性质过于病态而不作为实验数据，仅供调试参考
    # x = np.zeros(2, dtype=float)  # 初始点

    # def f(x):  # 定义函数
    #     return (4*x[0]**2+2*x[1]**2+4*x[0]*x[1]+2*x[1]+1)*math.exp(x[0])
    # 数据集3.
    # x = np.zeros(5, dtype=float)  # 初始点

    # def f(x):  # 定义函数
    #     A = [[10, 1, 2, 3, 4], [1, 9, -1, 2, -3], [2, -1, 7, 3, -5],
    #          [3, 2, 3, 12, -1], [4, -3, -5, -1, 15]]
    #     b = [12, -27, 14, -17, 12]
    #     return np.linalg.norm(np.matmul(A, x)-b)          
    while l<5:
        n=len(x)
        for i in range(n):
            x[i]=random.uniform(-10,10)
        print("初始点：",x)
        time_start = time()
        x0, k, judge = quasi_newton_method(f, x)
        time_end = time()
        if judge:
            print("极小值点：", x0)
            print("极小值：", f(x0))
        print("拟牛顿法的迭代次数：", k)
        print("算法运行时间：", time_end-time_start, "单位：s",'\n')
        l+=1


if __name__ == '__main__':
    main()
