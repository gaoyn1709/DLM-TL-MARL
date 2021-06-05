# 读取topology并绘制
# 确认一致性
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import Test

class Utils:
    def __init__(self):
        self.NUM_NODE = Test.NUM_NODE
        self.LAYOUT_SIZE = Test.LAYOUT_SIZE
        # self.orig_adjacent = pd.read_csv('adjacent.csv',header=None, index_col=0)
        self.adjacent = [[0]*self.NUM_NODE for i in range(self.NUM_NODE)]
        # 指定SD对
        # self.source = [13,45,10,14]
        # self.destination = [39,6,35,41]
        # self.source = [13,13]
        # self.destination = [39,46]
        self.source = [13]
        self.destination = [39]
        self.edge_num = 0
        self.neighbors = {}
        self.edge_width = []

    def getSource(self):
        return self.source

    def getDestination(self):
        return self.destination

    def getNodeNum(self):
        return self.NUM_NODE

    def getWidth(self):
        return self.edge_width

    def getNeighbors(self):
        self.neighbor_data = pd.read_csv('./diagonal_adjacent.csv', header=None, index_col=False)
        for node in range(0, self.NUM_NODE):
            nei = []
            for node2 in range(0, self.NUM_NODE):
                if self.neighbor_data.iat[node, node2] == 1:
                    nei.append(node2)
            self.neighbors[node] = nei
        return self.neighbors

    def get_edge_info(self):
        return self.convert_edge_info()

    def getData(self):
        self.data = pd.read_csv('./positions_size.csv')
        positions = []
        node_size = []
        for node in range(0, self.NUM_NODE):
            pos = []
            pos.append(self.data['x'][node])
            pos.append(self.data['y'][node])
            positions.append(pos)
            node_size.append(self.data['size'][node])
        return positions, node_size

    def convert_edge_info(self):
        self.edge_info = pd.read_csv('./edge_info.csv')
        edge_info_width = {}
        for row in range(self.edge_info.shape[0]):
            key_info = str(self.edge_info['pre'][row]) + '-' + str(self.edge_info['post'][row])
            edge_info_width[key_info] = self.edge_info['width'][row]
        return edge_info_width


    def draw(self, positions, node_size, path, name):
        G = nx.Graph()
        nodes = []
        for nd in range(0, self.NUM_NODE):
            nodes.append(nd)
        G.add_nodes_from(nodes)

        edge_info_width = self.convert_edge_info()
        for key, value in edge_info_width.items():
            node_list = key.split('-')
            node_pre = int(node_list[0])
            node_post = int(node_list[1])
            G.add_edge(node_pre, node_post, weight=value)
            self.edge_num = self.edge_num + 1
            self.adjacent[node_pre][node_post] = 1
            self.adjacent[node_post][node_pre] = 1
        # pd.DataFrame(data=self.adjacent).to_csv('diagonal_adjacent.csv', index=False, header=False)

        # -------------------------------------设置点和边界的颜色-------------------------------------------
        # node colors
        # colors = []
        # for node in range(0, self.NUM_NODE):
        #     if node in self.source:
        #         colors.append('#FFD306')
        #         continue
        #     if node in self.destination:
        #         colors.append('#EA0000')
        #         continue
        #     colors.append('#004B97')
        #
        # edge_colors = ['#000000' for i in range(self.edge_num)]

        # 查找给定SD对的最短路径
        # for src,dst in zip(self.source, self.destination):
        #     shortest_path = nx.shortest_path(G, src, dst)
        #     shortest_distance = nx.shortest_path_length(G, src, dst)
        #     # edge colors
        #     ite = 0
        #     for edge in G.edges():
        #         if edge[0] in shortest_path and edge[1] in shortest_path:
        #             edge_colors[ite] = '#00DB00'
        #             ite = ite + 1
        #             continue
        #         ite = ite + 1
        # -------------------------------------设置点和边界的颜色-------------------------------------------

        colors = []
        for node in range(0, self.NUM_NODE):
            if node == path[0]:
                colors.append('#FFD306')
                continue
            if node == path[-1]:
                colors.append('#EA0000')
                continue
            colors.append('#004B97')
        edge_colors = ['#000000' for i in range(self.edge_num)]
        ite = 0
        flag_edge = [False for i in range(G.number_of_edges())]
        for edge in G.edges():
            if edge == (11,10) or edge == (10,11):
                z = 1
            # 判断末尾节点
            if edge[0] == path[-1]:
                if edge[1] == path[-2]:
                    edge_colors[ite] = '#B8860B' if flag_edge[ite] else '#00DB00'
                    flag_edge[ite] = True
                    ite = ite + 1
                    continue
                else:
                    ite = ite + 1
                    continue
            if edge[1] == path[-1]:
                if edge[0] == path[-2]:
                    edge_colors[ite] = '#B8860B' if flag_edge[ite] else '#00DB00'
                    flag_edge[ite] = True
                    ite = ite + 1
                    continue
                else:
                    ite = ite + 1
                    continue

            if edge[0] in path:
                p = path.index(edge[0])
                if edge[1] == path[p+1]:
                    edge_colors[ite] = '#B8860B' if flag_edge[ite] else '#00DB00'
                    flag_edge[ite] = True
                    ite = ite + 1
                    continue
            if edge[1] in path:
                p = path.index(edge[1])
                if edge[0] == path[p+1]:
                    edge_colors[ite] = '#B8860B' if flag_edge[ite] else '#00DB00'
                    flag_edge[ite] = True
                    ite = ite + 1
                    continue
            ite = ite + 1

        # 更新全局参数，设置图形大小
        plt.rcParams.update({
            'figure.figsize': (self.LAYOUT_SIZE, self.LAYOUT_SIZE)
        })
        edge_width_pic = [width/1.5 for width in self.edge_width]
        node_size_pic = [i * 40 for i in node_size]

        nx.draw_networkx_edges(G, positions, width=[d['weight']/1.5 for (u,v,d) in G.edges(data=True)],
                               edge_color=edge_colors)
        nx.draw_networkx_nodes(G, positions, node_size=node_size_pic, node_color=colors)
        nx.draw_networkx_labels(G, positions)

        plt.savefig('./networkx_iql_contend_path'+name+'.png')
        plt.show()

    def test_Utils(self):
        positions, node_size = self.getData()
        self.draw(positions, node_size)
#
# test_util = Utils()
# test_util.test_Utils()