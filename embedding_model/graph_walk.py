#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
import networkx as nx


class BasicWalker:
    def __init__(self, nx_G):
        self.G = nx_G

    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        walk = [start_node]

        while len(walk) < walk_length: 
            cur = walk[-1]
            cur_nbrs = G.neighbors(cur)
            if len(cur_nbrs) > 0:
                weight_list = [G[cur][nbr]['weight'] for nbr in cur_nbrs]
                normal_weight_list = [float(w)/sum(weight_list) for w in weight_list]
                next_node = np.random.choice(cur_nbrs, 1, p=normal_weight_list)[0]
                walk.append(next_node)               
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        ''' Repeatedly simulate random walks from each node. '''
        G = self.G
        nodes = list(G.nodes())
        walks = []
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print str(walk_iter+1), '/', str(num_walks)
            random.shuffle(nodes)
            for node in nodes:    
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=node))
        print('walks number..', len(walks))
        return walks