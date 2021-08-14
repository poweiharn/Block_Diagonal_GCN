import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt 
import scipy.sparse as sp
#from block_diagonal_gcn.dataloader import normalize_adj
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # Compute D^{-0.5} * A * D^{-0.5}
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
def clique_partition(edges):

  traverse_list = []
  p_trav_list = []
  count = 0
  cliquescount = 0
  nocliquescount = 0
  p = None
  q = None
  adj_list = []
  indices_list = []
  G = nx.Graph()
  G.add_edges_from(edges)
  print('number print', G.number_of_nodes(), G.number_of_edges())
  #stripped_list = list(set([tuple(set(node)) for node in list(G.nodes)]))
  #print('list length', len(stripped_list))
  while(G.number_of_edges() != 0):
    #print('nodes number', G.number_of_nodes())
    common_neigh = {}
    max_common_neigh = {}
    tie_max = []
    min_degree_list = []
    degree_comp = {}
    common_neigh_list = []

    dup_common_neigh = {}
    degree_list = {}
    for i in G.nodes():

      degree_list[i] = G.degree[i]
    
    #finding p
    if p is None:

      p = min(degree_list, key=degree_list.get)
      
    else:
      for item1, item2 in sorted(degree_list.items(), key=lambda x:x[1]):
  
        if(item1 not in p_trav_list):
          p = item1

          break

    p_trav_list.append(p)

    p_neighbors = list(G.neighbors(p))

    for i in p_neighbors:
      if(p != i):
        
        common_neigh[i] = sorted(nx.classes.function.common_neighbors(G, p, i))
        max_common_neigh[i] = len(list(nx.classes.function.common_neighbors(G, p, i)))

    #finding q

    for iter in p_neighbors:

      if q is None:
        all_values = max_common_neigh.values()
        if(len(all_values)>0):
          max_count = max(all_values)
          

          tie_max = [k for k,v in max_common_neigh.items() if v == max_count]

          if(len(tie_max) == 1):
            q = max(max_common_neigh, key=max_common_neigh.get)
          else:
            for tie in tie_max:
              temp = G.degree(tie)
              degree_comp[tie] = temp
            q = min(degree_comp, key=degree_comp.get)
        else:
          max_count = 0
        
      else:
        for key, value in max_common_neigh.items():
          if key not in traverse_list:
            dup_common_neigh[key] = value
          
        all_values = dup_common_neigh.values()
        if(len(all_values)>0):
          max_count = max(all_values)

          tie_max = [k for k,v in dup_common_neigh.items() if v == max_count]

          if(len(tie_max) == 1):
            q = max(dup_common_neigh, key=dup_common_neigh.get)
          else:
            for tie in tie_max:
              if tie in G.nodes:
                temp = G.degree(tie)
                degree_comp[tie] = temp
            q = min(degree_comp, key=degree_comp.get)
        else:
          max_count = 0

      
      if(max_count !=0):
        traverse_list.append(q)
        for item1, item2 in common_neigh.items():
            for k in item2:
              common_neigh_list.append(k)
        for i in G.nodes:
          #print(common_neigh_list)
          if(i not in common_neigh_list and i!=p and i!=q):

            if((p,i) in G.edges):
              G.remove_edge(p,i)

            if((q,i) in G.edges):
              G.remove_edge(q,i)

        subgraph_list = []
        if p in G.nodes:
          subgraph_list = list(G.neighbors(p))

          subgraph_list.append(p)
      else:
        subgraph_list = []
        if p in G.nodes:
          subgraph_list = list(G.neighbors(p))

          subgraph_list.append(p)
      
        if(q in G.nodes):
          q_list = []
          q_list = list(G.neighbors(q))
          q_list.append(q)
        
          for item in q_list:
            if(item not in subgraph_list):
              subgraph_list.append(item)

      if(len(subgraph_list) != 0):
        
        H = G.subgraph(subgraph_list) 
        n = len(subgraph_list)

        if(H.size() == int((n*(n-1))/2)):
          cliquescount = cliquescount + 1
        else:
          #nocliquescount = nocliquescount + 1
          np_array = np.ones((n,n), dtype=int)
          result_matrix = np_array - np.identity(n, dtype=int)

          H = nx.from_numpy_matrix(result_matrix)
          H=nx.relabel_nodes(H,dict(enumerate(subgraph_list)))
          adj = nx.adjacency_matrix(H)

          cliquescount = cliquescount + 1


       
        adj = nx.adjacency_matrix(H)
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        #without normalization
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        idx = sorted(subgraph_list.copy())
        adj_list.append(adj)
        indices_list.append(idx)
        #print('\nadj matrix of partition ' + str(int(count+1)) + '\n', A.todense())
        count = count + 1
        G.remove_nodes_from([n for n in G if n in set(subgraph_list)]) 
      #    nx.draw_networkx(G)
  if(G.number_of_nodes()!=0):
    for node in list(G.nodes):
      H = G.subgraph(node)
      adj = nx.adjacency_matrix(H)
      adj = sp.coo_matrix(adj, dtype=np.float32)
      adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
      #without normalization
      adj = normalize_adj(adj + sp.eye(adj.shape[0]))
      adj_list.append(adj)
      indices_list.append(list(H.nodes))
      n = len(list(H.nodes))
      if(H.size() == int((n*(n-1))/2)):
          cliquescount = cliquescount + 1
      else:
          nocliquescount = nocliquescount + 1
    print('number of cliques', cliquescount)
    print('number of NON-cliques', nocliquescount)
    G.remove_nodes_from(list(G.nodes)) 


  return adj_list, indices_list
  #print('number of partitions', count)
  #print('number of cliques obtained', cliquescount)
  #print('number of non cliques obtained', nocliquescount)

def general_partition(edges):

  traverse_list = []
  p_trav_list = []
  count = 0
  cliquescount = 0
  nocliquescount = 0
  p = None
  q = None
  adj_list = []
  indices_list = []
  G = nx.Graph()
  G.add_edges_from(edges)
  while(G.number_of_edges() != 0):
    
    common_neigh = {}
    max_common_neigh = {}
    tie_max = []
    min_degree_list = []
    degree_comp = {}
    common_neigh_list = []

    dup_common_neigh = {}
    degree_list = {}
    for i in G.nodes():

      degree_list[i] = G.degree[i]
    
    #finding p
    if p is None:

      p = min(degree_list, key=degree_list.get)
      
    else:
      for item1, item2 in sorted(degree_list.items(), key=lambda x:x[1]):
  
        if(item1 not in p_trav_list):
          p = item1

          break

    p_trav_list.append(p)

    p_neighbors = list(G.neighbors(p))

    for i in p_neighbors:
      if(p != i):
        
        common_neigh[i] = sorted(nx.classes.function.common_neighbors(G, p, i))
        max_common_neigh[i] = len(list(nx.classes.function.common_neighbors(G, p, i)))

    #finding q

    for iter in p_neighbors:

      if q is None:
        all_values = max_common_neigh.values()
        if(len(all_values)>0):
          max_count = max(all_values)
          

          tie_max = [k for k,v in max_common_neigh.items() if v == max_count]

          if(len(tie_max) == 1):
            q = max(max_common_neigh, key=max_common_neigh.get)
          else:
            for tie in tie_max:
              temp = G.degree(tie)
              degree_comp[tie] = temp
            q = min(degree_comp, key=degree_comp.get)
        else:
          max_count=0
        
      else:
        for key, value in max_common_neigh.items():
          if key not in traverse_list:
            dup_common_neigh[key] = value
          
        all_values = dup_common_neigh.values()
        if(len(all_values)>0):
          max_count = max(all_values)

          tie_max = [k for k,v in dup_common_neigh.items() if v == max_count]

          if(len(tie_max) == 1):
            q = max(dup_common_neigh, key=dup_common_neigh.get)
          else:
            for tie in tie_max:
              if tie in G.nodes:
                temp = G.degree(tie)
                degree_comp[tie] = temp
            q = min(degree_comp, key=degree_comp.get)
        else:
          max_count=0

      
      if(max_count !=0):
        traverse_list.append(q)
        for item1, item2 in common_neigh.items():
            for k in item2:
              common_neigh_list.append(k)
        for i in G.nodes:
          #print(common_neigh_list)
          if(i not in common_neigh_list and i!=p and i!=q):

            if((p,i) in G.edges):
              G.remove_edge(p,i)

            if((q,i) in G.edges):
              G.remove_edge(q,i)

        subgraph_list = []
        if p in G.nodes:
          subgraph_list = list(G.neighbors(p))

          subgraph_list.append(p)
      else:
        subgraph_list = []
        if p in G.nodes:
          subgraph_list = list(G.neighbors(p))

          subgraph_list.append(p)
      
        if(q in G.nodes):
          q_list = []
          q_list = list(G.neighbors(q))
          q_list.append(q)
        
          for item in q_list:
            if(item not in subgraph_list):
              subgraph_list.append(item)

      if(len(subgraph_list) != 0):
        
        H = G.subgraph(subgraph_list) 
        n = len(subgraph_list)

        if(H.size() == int((n*(n-1))/2)):
          cliquescount = cliquescount + 1
        else:
          nocliquescount = nocliquescount + 1

       
        adj = nx.adjacency_matrix(H)
        idx = sorted(subgraph_list.copy())
        
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        indices_list.append(idx)
        adj_list.append(adj)
        #print('\nadj matrix of partition ' + str(int(count+1)) + '\n', A.todense())
        count = count + 1
        G.remove_nodes_from([n for n in G if n in set(subgraph_list)]) 
  if(G.number_of_nodes()!=0):
    for node in list(G.nodes):
      H = G.subgraph(node)
      adj = nx.adjacency_matrix(H)
      
      adj = sp.coo_matrix(adj, dtype=np.float32)
      adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
      adj = normalize_adj(adj + sp.eye(adj.shape[0]))
      adj_list.append(adj)
      indices_list.append(list(H.nodes))
      n = len(list(H.nodes))
      if(H.size() == int((n*(n-1))/2)):
          cliquescount = cliquescount + 1
      else:
          nocliquescount = nocliquescount + 1
    print('number of cliques', cliquescount)
    print('number of NON-cliques', nocliquescount)
    G.remove_nodes_from(list(G.nodes)) 

        
  return adj_list, indices_list

def random_partition(edges):
  adj_list = []
  indices_list =[]
  G = nx.Graph()
  G.add_edges_from(edges)
  total_length = len(list(G.nodes))
  
  length1 = int(len(list(G.nodes))*0.20)
  print(total_length, length1)
  node_ids1 = list(G.nodes)[0:length1]
  print('node_ids1', node_ids1)
  subgraph1 = G.subgraph(node_ids1)
  adj1 = nx.adjacency_matrix(subgraph1)
  adj1 = sp.coo_matrix(adj1, dtype=np.float32)
  adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
  adj1 = normalize_adj(adj1 + sp.eye(adj1.shape[0]))
  print(adj1.toarray())

  node_ids2 = list(G.nodes)[length1:length1*2]
  print('node_ids2', node_ids2)
  subgraph2 = G.subgraph(node_ids2)
  adj2 = nx.adjacency_matrix(subgraph2)
  adj2 = sp.coo_matrix(adj2, dtype=np.float32)
  adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
  adj2 = normalize_adj(adj2 + sp.eye(adj2.shape[0]))
  print(adj2.toarray())

  node_ids3 = list(G.nodes)[length1*2:length1*3]
  print('node_ids3', node_ids3)
  subgraph3 = G.subgraph(node_ids3)
  adj3 = nx.adjacency_matrix(subgraph3)
  adj3 = sp.coo_matrix(adj3, dtype=np.float32)
  adj3 = adj3 + adj3.T.multiply(adj3.T > adj3) - adj3.multiply(adj3.T > adj3)
  adj3 = normalize_adj(adj3 + sp.eye(adj3.shape[0]))
  print(adj3.toarray())

  node_ids4 = list(G.nodes)[length1*3:length1*4]
  print('node_ids4', node_ids4)
  subgraph4 = G.subgraph(node_ids4)
  adj4 = nx.adjacency_matrix(subgraph4)
  adj4 = sp.coo_matrix(adj4, dtype=np.float32)
  adj4 = adj4 + adj4.T.multiply(adj4.T > adj4) - adj4.multiply(adj4.T > adj4)
  adj4 = normalize_adj(adj4 + sp.eye(adj4.shape[0]))
  print(adj4.toarray())

  node_ids5 = list(G.nodes)[length1*4:total_length]
  print('node_ids5', node_ids5)
  subgraph5 = G.subgraph(node_ids5)
  adj5 = nx.adjacency_matrix(subgraph5)
  adj5 = sp.coo_matrix(adj5, dtype=np.float32)
  adj5 = adj5 + adj5.T.multiply(adj5.T > adj5) - adj5.multiply(adj5.T > adj5)
  adj5 = normalize_adj(adj5 + sp.eye(adj5.shape[0]))
  print(adj5.toarray())


  adj_list.append(adj1)
  adj_list.append(adj2)
  adj_list.append(adj3)
  adj_list.append(adj4)
  adj_list.append(adj5)
  indices_list.append(list(node_ids1))
  indices_list.append(list(node_ids2))
  indices_list.append(list(node_ids3))
  indices_list.append(list(node_ids4))
  indices_list.append(list(node_ids5))
  return adj_list, indices_list 