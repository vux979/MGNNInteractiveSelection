from torch_geometric.data import Data
class PairData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 x_mask=None, edge_index_mask=None, edge_attr_mask=None, y_mask=None):
        super(PairData, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.x_mask = x_mask
        self.edge_index_mask = edge_index_mask
        self.edge_attr_mask = edge_attr_mask
        self.y_mask = y_mask
