from pyverilog.vparser.parser import parse          # type: ignore
from pyverilog.vparser import ast                   # type: ignore
from pyverilog.vparser.ast import Width, ModuleDef  # type: ignore
import logging
from typing import Tuple, List, Dict, Optional
import pickle
import networkx as nx                               # type: ignore
import matplotlib.pyplot as plt                     # type: ignore
import sys
import os
import time
import re

PRIMITIVES = ['and', 'nand', 'or', 'nor', 'xor', 'xnor', 'buf', 'not']
FLOP = ['CDN_flop']

# Configure logging
# logging.basicConfig(level=logging.DEBUG)

def get_module_list(ast:ast) -> List[ModuleDef]:
    module_list = []
    for child in ast.children():
        for c in child.children():
            if c.__class__.__name__ == 'ModuleDef':
                module_list.append(c)
    return module_list

def width(width:Width) -> int:
    bit_width = 1
    if width:
        bit_width += abs(int(width.lsb.value) - int(width.msb.value))
    return bit_width

class GNode():
    def __init__(self, name:str) -> None:
        self.name:str = name
        self.id:int = -1
        self.input_count = 0
        self.output_count = 0
        self.avg_input_bits = 0.0
        self.avg_output_bits = 0.0
        self.logic_count = 0
        self.flop_count = 0
        self.macro_count = 0
        self.avg_logic_bits = 0.0
        self.children_name = None
        self.children: Optional[List[GNode]] = None 
            
    def update_input_count(self, count:int) -> None:
        self.input_count = count
    
    def update_output_count(self, count:int) -> None:
        self.output_count = count
    
    def update_avg_input_bits(self, count:float) -> None:
        self.avg_input_bits = count
        
    def update_avg_output_bits(self, count:float) -> None:
        self.avg_output_bits = count
    
    def update_logic_count(self, count:int) -> None:
        self.logic_count = count
    
    def update_flop_count(self, count:int) -> None:
        self.flop_count = count
    
    def update_macro_count(self, count:int) -> None:
        self.macro_count = count
    
    def update_avg_logic_bits(self, count:float) -> None:
        self.avg_logic_bits = count
    
    def update_children_name(self, children: List[str]) -> None:
        self.children_name = children
    
    def update_children(self, children: List['GNode']) -> None:
        self.children = children
    
    def __eq__(self, other:'GNode') -> bool:
        if isinstance(other, GNode):
            if (self.input_count == other.input_count and
                self.output_count == other.output_count and
                self.avg_input_bits == other.avg_input_bits and
                self.avg_output_bits == other.avg_output_bits and
                self.logic_count == other.logic_count and
                self.flop_count == other.flop_count and
                self.avg_logic_bits == other.avg_logic_bits and
                self.children_name == other.children_name):
                return True
            else:
                return False

def extract_feature(module, module_list: List[str],
                    primitive_list: List[str] = PRIMITIVES) -> GNode:
    # Number of Inputs 
    input_count = 0
    # Number of Outputs
    output_count = 0
    # Average number of Input bits
    total_input_bits = 0
    avg_input_bits = 0
    # Average number of Output bits
    total_output_bits = 0
    avg_output_bits = 0
    # Number of Logic gates
    num_logic_gates = 0
    # Average number of inputs of logic gates
    total_number_inputs_logic_gate = 0
    avg_number_inputs_logic_gate = 0
    # Register Count
    number_flop = 0
    # Memory Macro Count
    number_mem = 0
    child = []
    for item in module.items:
        if item.__class__.__name__ == 'Decl':
            for i in item.children():
                if i.__class__.__name__ == 'Input':
                    input_count += 1
                    total_input_bits += width(i.width)
                elif i.__class__.__name__ == 'Output':
                    output_count += 1
                    total_output_bits += width(i.width)
        if item.__class__.__name__ == 'InstanceList':
            for inst in item.instances:
                if inst.module in module_list:
                    child.append(inst.module)
                elif inst.module in primitive_list:
                    num_logic_gates += 1
                    tmp_input_count = len(inst.portlist) - 1
                    total_number_inputs_logic_gate += tmp_input_count
                elif inst.module == 'CDN_flop':
                    number_flop += 1
                elif re.search(r"^DPRFW\S+",inst.module):
                    number_mem += 1
                else:
                    print(f"[ERROR] Module Def:{module.name} Inst:{inst.name} "
                          f"Inst Module:{inst.module}")
    
    if input_count != 0:
        avg_input_bits = total_input_bits / input_count
    if output_count != 0:
        avg_output_bits = total_output_bits / output_count
    
    if num_logic_gates != 0:
        avg_number_inputs_logic_gate = \
                        total_number_inputs_logic_gate / num_logic_gates
    
    # print(f"Module name:{module.name}")
    gnode = GNode(module.name)
    # print(f"Number of Input:{input_count}")
    gnode.update_input_count(input_count)
    # print(f"Number of Output:{output_count}")
    gnode.update_output_count(output_count)
    # print(f"Average input bit width:{avg_input_bits}")
    gnode.update_avg_input_bits(avg_input_bits)
    # print(f"Average output bit width:{avg_output_bits}")
    gnode.update_avg_output_bits(avg_output_bits)
    # print(f"Number of logic gates:{num_logic_gates}")
    gnode.update_logic_count(num_logic_gates)
    # print("Average number of inputs to "
        #   f"logic gates:{avg_number_inputs_logic_gate}")
    gnode.update_avg_logic_bits(avg_number_inputs_logic_gate)
    # print(f"Number of flops:{number_flop}")
    gnode.update_flop_count(number_flop)
    gnode.update_macro_count(number_mem)
    gnode.update_children_name(child)
    # print(f"Child modules are :")
    # for i in child:
    #     print('\t',i)
    return gnode

def create_graph(gnode: GNode,
                 name_to_node_map: Dict[str, GNode],
                 G: nx.Graph, pid: int,
                id:int) -> int:
    G.add_node(id, name=gnode.name, input_count=gnode.input_count,
               output_count=gnode.output_count, node_id = gnode.id,
               avg_input_bits = gnode.avg_input_bits,
               avg_output_bits = gnode.avg_output_bits,
               num_logic_gates = gnode.logic_count,
               flop_count = gnode.flop_count,
               avg_logic_bits = gnode.avg_logic_bits,
               macro_count = gnode.macro_count)
    node_id = id
    id += 1
    if pid != -1:
        G.add_edge(pid, node_id)
    
    if not gnode.children_name:
        return id
    
    for child in gnode.children_name:
        id = create_graph(name_to_node_map[child], name_to_node_map, G, 
                          node_id, id)
    
    return id


def draw_graph(G:nx.Graph, count:int) -> None:
    '''
    G: Input Graph  
    count: Cound of unique nodes in the graph
      
    Generates graph visualization
    '''
    name_to_nodelist = {}
    for node in G.nodes(data = True): # type: ignore
        if node[1]['name'] not in name_to_nodelist:
            name_to_nodelist[node[1]['name']] = []
        name_to_nodelist[node[1]['name']].append(node)
    
    cmap = plt.get_cmap('coolwarm', count) # type: ignore
    pos = nx.spring_layout(G) # type: ignore
    nx.draw_networkx(G, pos, with_labels=True) # type: ignore
    for key, value in name_to_nodelist.items():
        cid = name_to_nodelist[key][0][1]['node_id']
        nx.draw_networkx_nodes(G, pos, nodelist=[n[0] for n in value], # type: ignore 
                               node_color = [list(cmap(cid)[0:3])])

def gen_graph(ast:ast) -> Tuple[nx.Graph, int]:
    start_time = time.time()
    module_list = get_module_list(ast)
    end_time = time.time()
    logging.debug(f"Module list generation time:{end_time - start_time}s")
    gnode_list = []
    module_names = []
    name_to_node_map = {}
    start_time = time.time()
    for module in module_list:
        gnode_list.append(extract_feature(module, module_names))
        module_names.append(module.name)
        name_to_node_map[module.name] = gnode_list[-1]
    end_time = time.time()
    logging.debug(f"Module feature extraction time:{end_time - start_time}s")
    
    # Find Unique modules
    unique_gnode = []
    unique_gnode_name = []
    start_time = time.time()
    sorted_module_names = sorted(module_names)
    for module_name in sorted_module_names:
        if module_name in FLOP:
            continue
        
        if len(unique_gnode) == 0:
            unique_gnode.append(name_to_node_map[module_name])
            unique_gnode_name.append(module_name)
        
        if unique_gnode[-1] != name_to_node_map[module_name]:
            unique_gnode.append(name_to_node_map[module_name])
            unique_gnode_name.append(module_name)
        else:
            name_to_node_map[module_name] = unique_gnode[-1]
    
    end_time = time.time()
    logging.debug(f"Unique module extraction time:{end_time - start_time}s")
    
    # Update Node id
    j = 0
    for i in unique_gnode_name:
        # print(i)
        unique_gnode[j].id = j
        j += 1
    
    start_time = time.time()
    G = nx.Graph()
    id = create_graph(name_to_node_map[module_names[-2]], \
                    name_to_node_map, G, -1, 0)
    end_time = time.time()
    
    logging.debug(f"Create Graph runtime: {end_time - start_time}s")
    
    return G, j


def gen_graph_from_netlist(netlist:str) -> Tuple[nx.Graph, int]:
    start_time = time.time()
    ast, _ = parse([netlist], debug = False)
    end_time = time.time()
    logging.debug(f"AST generation time: {end_time - start_time}s")
    G, count = gen_graph(ast)
    return G, count

def gen_phy_hier_graph(node_file:str, edge_file:str) -> nx.Graph:
    if not os.path.exists(node_file):
        logging.error(f"{node_file} does not exists.")
    
    if not os.path.exists(edge_file):
        logging.error(f"{edge_file} does not exists")

    G = nx.Graph()
    node_name_to_id = {}
    fp = open(node_file, 'r')
    lines = fp.readlines()
    for i in range(len(lines)):
        if i == 0:    
            continue
        
        items = lines[i].split()
        name = items[0]
        node_features = []
        j = 1
        while j < len(items):
            node_features.append(float(items[j]))
            j += 1
        G.add_node(i-1, name=name, features=node_features)
        node_name_to_id[name] = i-1
    
    fp.close()
    fp = open(edge_file, 'r')
    lines = fp.readlines()
    for i in range(len(lines)):
        if i == 0:
            continue
        items = lines[i].split()
        node1 = items[0]
        node1_id = node_name_to_id[node1]
        node2 = items[1]
        node2_id = node_name_to_id[node2]
        edge_weight = float(items[2])
        G.add_weighted_edges_from([(node1_id, node2_id, edge_weight)])
    
    fp.close()
    return G

def save_graph(G: nx.Graph, file_name: str) -> None:
    save_dir = os.path.dirname(file_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    with open(file_name, "wb") as f:
        pickle.dump(G, f)
    
    return

if __name__ == '__main__':
    netlist = sys.argv[1]
    G, count = gen_graph_from_netlist(netlist)
    if len(sys.argv) == 3:
        file_name = sys.argv[2]
        save_graph(G, file_name)
    else:
        draw_graph(G, count)