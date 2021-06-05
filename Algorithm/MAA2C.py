from Utils import Utils
import Quantum_network_routing
from MAA2CUtils import Trainer


class MAA2C:
    def __init__(self):
        # parameters
        self.utils = Utils()
        self.NUM_NODE = self.utils.getNodeNum()
        self.source = self.utils.getSource()
        self.destination = self.utils.getDestination()
        self.visit = [False for i in range(self.NUM_NODE)]
        self.width = [0 for i in range(self.NUM_NODE)]
        self.neighbors = self.utils.getNeighbors()
        self.edge_width = self.utils.get_edge_info()
        self.positions, self.node_memory = self.utils.getData()
        self.path = []
        self.w = []

    def init_env(self):
        return Quantum_network_routing()

    def init_agent(self):
        return True

    def train(self):
        env = self.init_env()
        model = self.init_agent()
        trainer = Trainer(env, model)
        trainer.run()




if __name__=='__main__':
    maa2c = MAA2C()
    maa2c.train()
    print('test')

