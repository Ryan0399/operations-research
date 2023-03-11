import networkx as nx
import numpy as np
import math
import time


def algorithm(D, g, f, d):
    """
    输入有向图D=(V,E)以及容量函数和成本函数g,f,d
    """
    V = D.number_of_nodes()
    E = D.number_of_edges()
    x = {}
    while (1):
        n = 0
        d_1 = {}
        for (i, j) in D.edges():
            if f[(i, j)] != g[(i, j)]:
                d_1[(i, j)] = d[(i, j)]
            else:
                d_1[(i, j)] = 0
                n += 1
        if n == E:
            break
        m = max(d_1.values())
        k = V*math.sqrt(E)*1.0/m
        for (i, j) in D.edges():
            d_1[(i, j)] *= k
        d_2 = {}
        for (i, j) in D.edges():
            d_2[(i, j)] = math.floor(d_1[(i, j)])
        a, x, pi = out_of_kilter(D, g, f, d_2)
        for (u, v) in D.edges():
            if math.fabs(d_1[(u, v)]+pi[u]-pi[v]) >= V:
                f[(u, v)] = x[(u, v)]
                g[(u, v)] = x[(u, v)]
    if a:
        print("不存在可行流！")
    return a, x, g, f


def out_of_kilter(D, g, f, d):
    """
    利用瑕疵算法计算最小费用流x以及势向量pi
    """
    a = False  # 存在可行流
    V = D.number_of_nodes()
    x = {}
    for (u, v) in D.edges():
        x[(u, v)] = 0
    pi = np.zeros(V, dtype=int)
    while True:
        # step1
        # 构造残量网络N(x)
        N, d_1, g_1 = residual_network(D, x, g, f, d)
        # 计算N(x)每条弧的kilter值
        c = {}
        for (i, j) in N.edges():
            c[(i, j)] = d_1[(i, j)]-pi[i]+pi[j]
        for (i, j) in N.edges():
            k = kilter(D, c, x, g, f, g_1, i, j)
            if k < 1e-12:
                k = 0
            if k > 0:  # 若弧(p,q)是瑕疵弧，跳出检索进入step2
                p = i
                q = j
                break
        if k == 0:
            return a, x, pi  # 若没有瑕疵弧，则返回最小费用流x和势向量pi
        # step2
        if N.has_edge(q, p):
            N.remove_edge(q, p)
        # 构造以max{0，C^{pi}_{ij}}为边长的，由N得到的带权有向图M
        M = nx.MultiDiGraph()
        for (i, j) in N.edges():
            M.add_weighted_edges_from([(i, j, max(0, c[(i, j)]))])
        # 使用Bellman-ford计算从节点q到所有节点i的最短路径
        y = np.zeros(len(pi), dtype=int)
        for i in M.nodes():
            if nx.has_path(M, q, i):
                y[i] = nx.bellman_ford_path_length(M, source=q, target=i)
            else:
                a = True
                return a, x, pi
        P = nx.bellman_ford_path(M, source=q, target=p)
        pi = np.subtract(pi, y)
        # step3
        c[(p, q)] = d_1[(p, q)]-pi[p]+pi[q]
        k_1 = kilter(D, c, x, g, f, g_1, p, q)  # 计算弧(p,q)的kilter数
        if k_1 < 1e-12:
            k_1 = 0
        if k_1 != 0:
            # 确定增广圈
            Q = nx.MultiDiGraph()
            Q.add_edge(p, q)
            for i in range(0, len(P)-1):
                Q.add_edge(P[i], P[i+1])
            r = g_1[(p, q)]
            for (i, j) in Q.edges():
                if r > g_1[(i, j)]:
                    r = g_1[(i, j)]
            for (i, j) in Q.edges():
                if D.has_edge(i, j):
                    x[(i, j)] += r
                else:
                    x[(j, i)] -= r


def kilter(D, c, x, g, f, g_1, i, j):
    """
    计算弧(i,j)的kilter数
    """
    k = 0.
    if D.has_edge(i, j):
        if f[(i, j)] <= x[(i, j)] and x[(i, j)] <= g[(i, j)]:
            if c[(i, j)] < 0:
                k = g_1[(i, j)]
        if x[(i, j)] < f[(i, j)]:
            if c[(i, j)] >= 0:
                k = g_1[(i, j)]
            if c[(i, j)] < 0:
                k = g[(i, j)]-x[(i, j)]
    elif D.has_edge(j, i):
        if x[(j, i)] > g[(j, i)]:
            if c[(i, j)] < 0:
                k = x[(j, i)]-f[(j, i)]
            if c[(i, j)] >= 0:
                k = g_1[(i, j)]
    return k


def residual_network(D, x, g, f, d):
    """
    构造有向图D的残量网络N
    """
    N = nx.MultiDiGraph()
    g_1 = {}
    d_1 = {}
    for (u, v) in D.edges():
        if x[(u, v)] >= f[(u, v)] and x[(u, v)] <= g[(u, v)]:
            if x[(u, v)] < g[(u, v)]:
                N.add_edge(u, v)
                g_1[(u, v)] = g[(u, v)]-x[(u, v)]
                d_1[(u, v)] = d[(u, v)]
            if x[(u, v)] > f[(u, v)]:
                N.add_edge(v, u)
                g_1[(v, u)] = x[(u, v)]-f[(u, v)]
                d_1[(v, u)] = -d[(u, v)]
        elif x[(u, v)] < f[(u, v)]:
            N.add_edge(u, v)
            g_1[(u, v)] = f[(u, v)]-x[(u, v)]
            d_1[(u, v)] = d[(u, v)]
        elif x[(u, v)] > g[(u, v)]:
            N.add_edge(v, u)
            g_1[(v, u)] = x[(u, v)]-g[(u, v)]
            d_1[(v, u)] = -d[(u, v)]
    return N, d_1, g_1


def main():
    # #定义有向图
    D = nx.DiGraph()
    # 数据集1.
    start_nodes = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    end_nodes = [1, 2, 2, 3, 4, 3, 4, 4, 0, 0]
    for i in range(0, len(start_nodes)):
        D.add_edge(start_nodes[i], end_nodes[i])
    upper_capacities = {(0, 1): 15, (0, 2): 8, (1, 2): 20, (1, 3): 4, (1, 4)
                         : 10, (2, 3): 15, (2, 4): 4, (3, 4): 20, (3, 0): 5, (4, 0): 15}
    lower_capacities = {(0, 1): 1, (0, 2): 2, (1, 2): 2, (1, 3): 1, (1, 4): 0,
                        (2, 3): 1, (2, 4): 4, (3, 4): 9, (3, 0): 5, (4, 0): 15}
    unit_costs = {(0, 1): 4, (0, 2): 4, (1, 2): 2, (1, 3): 2, (1, 4): 6,
                 (2, 3): 1, (2, 4): 3, (3, 4): 2, (3, 0): 0, (4, 0): 0}
    # 数据集2.
    # start_nodes = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5]
    # end_nodes = [1, 3, 2, 3, 3, 5, 4, 2, 5, 0]
    # for i in range(0, len(start_nodes)):
    #     D.add_edge(start_nodes[i], end_nodes[i])
    # upper_capacities = {(0, 1): 8, (0, 3): 7, (1, 2): 9, (1, 3): 5, (2, 3)
    #                     : 2, (2, 5): 5, (3, 4): 9, (4, 2): 6, (4, 5): 10, (5, 0): 14}
    # lower_capacities = {(0, 1): 3, (0, 3): 0, (1, 2): 1, (1, 3): 2,
    #                     (2, 3): 0, (2, 5): 1, (3, 4): 2, (4, 2): 0, (4, 5): 4, (5, 0): 10}
    # unit_costs = {(0, 1): 2, (0, 3): 8, (1, 2): 2, (1, 3): 5, (2, 3)
    #             : 1, (2, 5): 6, (3, 4): 3, (4, 2): 4, (4, 5): 7, (5, 0): 0}
    # 数据集3.
    # start_nodes = [0, 0, 1, 1, 2, 2, 3, 4]
    # end_nodes = [1, 2, 3, 4, 1, 3, 4, 0]
    # for i in range(0, len(start_nodes)):
    #     D.add_edge(start_nodes[i], end_nodes[i])
    # upper_capacities = {(0, 1): 10.3, (0, 2): 8.1, (1, 3): 2.3,
    #                     (1, 4): 7, (2, 1): 5.4, (2, 3): 10, (3, 4): 4, (4, 0): 12}
    # lower_capacities = {(0, 1): 1.3, (0, 2): 0.1, (1, 3): 0, (1, 4)                        : 2.1, (2, 1): 0.3, (2, 3): 2.2, (3, 4): 0, (4, 0): 5}
    # unit_costs = {(0, 1): 4.2, (0, 2): 1.5, (1, 3): 6.3, (1, 4)                  : 1, (2, 1): 2, (2, 3): 3.7, (3, 4): 2.1, (4, 0): 1}
    time_start = time.time()
    judge, minimal_cost_circular_flow, upper_capacities, lower_capacities = algorithm(
        D, upper_capacities, lower_capacities, unit_costs)
    time_end = time.time()
    if not judge:
        print("最小费用循环流的解集合P(f*,g*):")
        print("f*=", lower_capacities)
        print("g*=", upper_capacities)
        mincost = 0
        for (i, j) in D.edges():
            mincost += minimal_cost_circular_flow[(i, j)]*unit_costs[(i, j)]
        print("最小费用为：", mincost)
        print("最小费用循环流为：")
        for (i, j) in D.edges():
            print((i, j), " -> ", minimal_cost_circular_flow[(i, j)])
        print("算法运行时间：", time_end-time_start, "单位：s")


if __name__ == "__main__":
    main()
