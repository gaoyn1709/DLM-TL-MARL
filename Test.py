# 输入六个参数，确定一个网络拓扑
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
import math

# 画布size
LAYOUT_SIZE = 10

# 图的节点个数
NUM_NODE = 50

# 划分邻居的标准
NEI_MIN = LAYOUT_SIZE / 10  # 越大，图越稀疏

# 平均邻居个数
AVER_NUM_NEI = 4
NUM_NEI = random.uniform(2, 6, size=(1, NUM_NODE)).tolist()[0]

# waxman model parameters
gamma = 0.9
beta = 20


def initial():
    # 更新全局参数，设置图形大小
    plt.rcParams.update({
        'figure.figsize': (LAYOUT_SIZE, LAYOUT_SIZE)
    })

    # 创建空的无向图
    G = nx.Graph()
    nodes = list()
    for nd in range(0, NUM_NODE):
        nodes.append(nd)
    G.add_nodes_from(nodes)

    return G, nodes


def adjust_positions(positions):
    min_distance = 0
    while min_distance < NEI_MIN:
        min_d = LAYOUT_SIZE
        for i in range(0, NUM_NODE):
            flag = False
            for j in range(i + 1, NUM_NODE):
                distance = math.sqrt(
                    math.pow(positions[i][0] - positions[j][0], 2) + math.pow(positions[i][1] - positions[j][1], 2))
                if min_d > distance:
                    min_d = distance
                if distance < NEI_MIN:
                    positions[j] = (random.uniform(0, LAYOUT_SIZE), random.uniform(0, LAYOUT_SIZE))
                    flag = True
                    break
            if flag:
                break
        min_distance = min_d
    return positions


# 生成节点，即生成节点的quantum memory 和 topology
def generate_nodes(nodes):
    positions = {}
    for node in nodes:
        positions[node] = (random.uniform(0, LAYOUT_SIZE), random.uniform(0, LAYOUT_SIZE))
    ad_postions = adjust_positions(positions)
    return ad_postions


def generate_edges(G, positions, nodes):
    # 求任意两个节点间的最大距离
    max_l = 0
    for i in range(0, NUM_NODE):
        for j in range(i + 1, NUM_NODE):
            dis = math.sqrt(
                math.pow(positions[i][0] - positions[j][0], 2) +
                math.pow(positions[i][1] - positions[j][1], 2))
            if max_l < dis:
                max_l = dis
    # 存储连接矩阵
    adjacent = []
    edge_num = 0
    edge = {}
    edge_width = []
    for node in nodes:
        save_edge = []
        # 生成邻居
        num_nei = int(NUM_NEI[node])
        # 根据每个节点，计算邻居个最近的邻居: 三个判断条件：距离由近及远、在规定范围内、必须有一条边连接
        node_distance = {}  # num_node - 1
        for ite in range(0, NUM_NODE):
            if ite == node:
                continue
            node_distance[ite] = math.sqrt(math.pow(positions[node][0] - positions[ite][0], 2) +
                                           math.pow(positions[node][1] - positions[ite][1], 2))
        # 排序
        sort_neighbors = sorted(node_distance.items(), key=lambda x: x[1], reverse=False)

        # 初始化边矩阵
        for i in range(0, NUM_NODE):
            save_edge.append(0)

        # 建立边
        ite_neighbors = 0
        for key in sort_neighbors:
            if ite_neighbors < num_nei:
                ite_neighbors = ite_neighbors + 1
            else:
                break

            n = key[0]
            # waxman model判断该边界能不能建立
            l = math.sqrt(math.pow(positions[node][0] - positions[n][0], 2) +
                          math.pow(positions[node][1] - positions[n][1], 2))
            p = gamma * math.exp((-1 * l) / (beta * max_l))
            if random.random() < p:
                if not G.has_edge(node, n):
                    w = int(random.uniform(3, 7))
                    G.add_edge(node, n, name='edge', weight=w)
                    edge_width.append(w)
                    save_edge[n] = 1
                    edge[edge_num] = (node, n)
                    edge_num = edge_num + 1
        adjacent.append(save_edge)
    return adjacent, edge_num, edge, edge_width


def draw(G, nodes):
    positions = generate_nodes(nodes)
    # 生成边：确定邻居边+Waxman判断能否成功建立
    adjacent, edge_num, edge, edge_width = generate_edges(G, positions, nodes)

    # 设置note_size
    node_size = []
    node_size_pic = []
    for node in nodes:
        size = int(random.uniform(10, 14))
        node_size.append(size)
        node_size_pic.append(size * 40)

    # 保存同一种拓扑：点的位置、点的size、连接矩阵
    results = pd.concat([pd.DataFrame(data=positions).T, pd.DataFrame(data=node_size)], axis=1)
    results.columns = ['x', 'y', 'size']
    results.to_csv('positions_size.csv')
    pd.DataFrame(data=adjacent).to_csv('adjacent.csv', header=False)

    # 保存边的最大宽度
    edge_results = pd.concat([pd.DataFrame(data=edge).T, pd.DataFrame(data=edge_width)], axis=1)
    edge_results.columns = ['pre', 'post', 'width']
    edge_results.to_csv('edge_info.csv')

    nx.draw_networkx(G, positions, node_size=node_size_pic,
                     width=[d['weight'] / 1.5 for (u, v, d) in G.edges(data=True)])
    plt.savefig('./topology.png')
    plt.show()
# G, nodes = initial()
# draw(G, nodes)
