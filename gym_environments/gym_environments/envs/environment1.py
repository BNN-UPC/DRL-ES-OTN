import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pandas as pd
import pickle
import json 
import gc
import tensorflow as tf

def process_graph_file(graph_file):
    links_bw = []
    Gbase = nx.MultiDiGraph()

    with open(graph_file) as fd:
        line = fd.readline()
        camps = line.split(" ")
        net_size = int(camps[1])
        # Remove : label x y
        line = fd.readline()
        
        for i in range(net_size):
            links_bw.append({})
        for line in fd:
            if (not line.startswith("Link_") and not line.startswith("edge_")):
                continue
            camps = line.split(" ")
            src = int(camps[1])
            dst = int(camps[2])
            bw = float(camps[4])
            Gbase.add_edge(src, dst)
            # Use the link capacity from the OTN paper, not from topology
            links_bw[src][dst] = 200 #bw
    
    return Gbase, links_bw

class Env1(gym.Env):
    """
    Environment used in the OTN routing problem.
    We are using bidirectional links in this environment and we make the MP between edges.

    self.edge_state[:][0] = link available capacity
    self.edge_state[:][1] = bw allocated (the one that goes from src to dst)
    """
    def __init__(self):
        self.graph = None # Here we store the graph as DiGraph (without repeated edges)
        self.source = None
        self.destination = None
        self.demand = None
        self.initial_state = None

        self.edge_state = None
        self.graph_topology_name = None # Here we store the name of the graph topology from the repetita dataset
        self.dataset_folder_name = None # Here we store the name of the repetita dataset being used: 2015Defo, 2016TopologyZoo_unary,2016TopologyZoo_inverseCapacity, etc. 

        self.diameter = None

        self.first = None
        self.firstTrueSize = None
        self.second = None

        self.K = None
        self.nodes = None # List of nodes to pick randomly from them
        self.edgesDict = dict() # Stores the position id of each edge in order

        self.numNodes = None
        self.numEdges = None
        # Original DQN from OTN paper has the bw marked as one-hot vector
        self.bw_allocated_feature = None

        self.links_bw = None
        self.episode_over = True
        self.reward = 0
        self.allPaths = dict() # Stores the paths for each src:dst pair

    def compute_k_shortest_paths(self):
        # For each src,dst pair we compute the K shortest paths

        self.diameter = nx.diameter(self.graph)
        # Iterate over all node1,node2 pairs from the graph
        for n1 in self.graph:
            for n2 in self.graph:
                if (n1 != n2):
                    # Check if we added the element of the matrix
                    if str(n1)+':'+str(n2) not in self.allPaths:
                        self.allPaths[str(n1)+':'+str(n2)] = []
                    
                    # First we compute the shortest paths taking into account the diameter
                    [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=self.diameter*2)]

                    # We take all the paths from n1 to n2 and we order them according to the path length
                    # sorted() ordena los paths de menor a mayor numero de
                    # saltos y los que tienen los mismos saltos te los ordena por indice
                    self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))

                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                        path = path + 1

                    # Remove paths not needed
                    del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
                    gc.collect()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
    
    def add_features_to_edges(self):
        incId = 1
        for node in self.graph:
            for adj in self.graph[node]:
                if not 'betweenness' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['betweenness'] = 0
                if not 'edgeId' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['edgeId'] = incId
                if not 'numsp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['numsp'] = 0
                if not 'utilization' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['utilization'] = 0
                if not 'capacity' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['capacity'] = 0
                if not 'weight' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['weight'] = 0
                if not 'kshortp' in self.graph[node][adj][0]:
                    self.graph[node][adj][0]['kshortp'] = 0
                if not 'crossing_paths' in self.graph[node][adj][0]: # We store all the src,dst from the paths crossing each edge
                    self.graph[node][adj][0]['crossing_paths'] = dict()
                incId = incId + 1

    def _first_second(self):
        # Link (1, 2) recibe trafico de los links que inyectan en el nodo 1
        # un link que apunta a un nodo envía mensajes a todos los links que salen de ese nodo
        first = list()
        second = list()

        for i in self.graph:
            for j in self.graph[i]:
                neighbour_edges = self.graph.edges(j)
                # Take output links of node 'j'

                for m, n in neighbour_edges:
                    if ((i != m or j != n) and (i != n or j != m)):
                        first.append(self.edgesDict[str(i) +':'+ str(j)])
                        second.append(self.edgesDict[str(m) +':'+ str(n)])

        # Because of the @tf.function call in actor_step and critic_step, we want to pass
        # everything in the tensor format
        self.first = tf.convert_to_tensor(first, dtype=tf.int32)
        self.second = tf.convert_to_tensor(second, dtype=tf.int32)

    def generate_environment(self, listofDemands, dataset_folder_name, graph_topology_name, K):
        self.graph_topology_name = graph_topology_name
        self.dataset_folder_name = dataset_folder_name
        
        self.listofDemands = listofDemands
        self.maxCapacity = 0 # We take the maximum capacity to normalize
        self.maxDemand = np.amax(self.listofDemands) # We store the maximum demand to scale the rewards

        self.graph, self.links_bw = process_graph_file(dataset_folder_name+graph_topology_name+'.graph')
        
        self.add_features_to_edges()
        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        # Here we store the bw_allocated (from each action) for each link using one-hot encoding
        self.bw_allocated_feature = np.zeros((self.numEdges,len(self.listofDemands)))

        self.K = K
        if self.K>self.numNodes:
            self.K = self.numNodes

        self.edge_state = np.zeros((self.numEdges, 2))

        self.compute_k_shortest_paths()

        position = 0
        # Initialize graph features
        for i in self.graph:
            for j in self.graph[i]:
                self.edgesDict[str(i)+':'+str(j)] = position
                self.graph[i][j][0]['capacity'] = self.links_bw[i][j]
                if self.graph[i][j][0]['capacity']>self.maxCapacity:
                    self.maxCapacity = self.graph[i][j][0]['capacity']
                self.edge_state[position][0] = self.graph[i][j][0]['capacity']
                self.graph[i][j][0]['utilization'] = 0.0
                position += 1
        # We store the initial graph state to later use it in the reset()
        self.initial_state = np.copy(self.edge_state)

        self._first_second()
        self.firstTrueSize = len(self.first)

        # We create the list of nodes ids to pick randomly from them
        self.nodes = list(range(0,self.numNodes))

    def step(self, action, demand, source, destination):
        # Action is the chosen K-path
        self.episode_over = True
        self.reward = 0

        currentPath = self.allPaths[str(source) +':'+ str(destination)][action]
        i = 0
        j = 1

        # Iterate over the path and allocate the traffic demand
        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]
            link_capacity = self.links_bw[firstNode][secondNode]
            edge_pos = self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]

            self.edge_state[edge_pos][0] -= demand
            # If the link is too full
            if self.edge_state[edge_pos][0]<0:
                return self.reward, self.episode_over, self.demand, self.source, self.destination 

            i = i + 1
            j = j + 1
        
        # Scale the reward to have values <1
        self.reward = demand/self.maxDemand
        self.episode_over = False

        # We pick a random pair of SOURCE,DESTINATION different nodes
        while True:
            self.demand = random.choice(self.listofDemands)
            self.source = random.choice(self.nodes)
            self.destination = random.choice(self.nodes)
            if self.destination != self.source:
                break
        
        # We desmark the bw_allocated
        self.edge_state[:,1] = 0
        return self.reward, self.episode_over, self.demand, self.source, self.destination
    
    def reset(self):
        self.edge_state = np.copy(self.initial_state)
        self.demand = random.choice(self.listofDemands)
        self.source = random.choice(self.nodes)

        # We pick a random pair of SOURCE,DESTINATION different nodes
        while True:
            self.demand = random.choice(self.listofDemands)
            self.source = random.choice(self.nodes)
            self.destination = random.choice(self.nodes)
            if self.destination != self.source:
                break

        return self.demand, self.source, self.destination

    def mark_action_k_path(self, k_path, source, destination, demand): 
        # In this function we mark the action in the corresponding edges of the SP between src,dst
        
        # We remove the previous actions
        self.bw_allocated_feature.fill(0.0)

        currentPath = self.allPaths[str(source) +':'+ str(destination)][k_path]
        path = 0
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]
            edge_pos = self.edgesDict[str(firstNode)+':'+str(secondNode)]

            self.edge_state[edge_pos][1] = demand
            
            if demand == 8:
                self.bw_allocated_feature[edge_pos][0] = 1
            elif demand == 32:
                self.bw_allocated_feature[edge_pos][1] = 1
            elif demand == 64:
                self.bw_allocated_feature[edge_pos][2] = 1
            
            i = i + 1
            j = j + 1
