import networkx as nx
import numpy as np
import math

def partition(edges):

  traverse_list = []
  p_trav_list = []
  count = 0
  cliquescount = 0
  nocliquescount = 0
  p = None
  q = None
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
        for key, value in max_common_neigh.items():
          if key not in traverse_list:
            dup_common_neigh[key] = value
          
        all_values = dup_common_neigh.values()
        max_count = max(all_values)

        tie_max = [k for k,v in dup_common_neigh.items() if v == max_count]

        if(len(tie_max) == 1):
          q = max(dup_common_neigh, key=dup_common_neigh.get)
        else:
          for tie in tie_max:
            temp = G.degree(tie)
            degree_comp[tie] = temp
          q = min(degree_comp, key=degree_comp.get)

      
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
        subgraph_list = list(G.neighbors(p))

        subgraph_list.append(p)
      else:
        subgraph_list = []
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
          A = nx.adjacency_matrix(H)

          cliquescount = cliquescount + 1


        '''plt.figure(figsize =(9, 12))
        plt.subplot(111)
        nx.draw_networkx(H)'''
        
        #plt.savefig(f'partition {str(int(count+1))}', dpi = fig.dpi)
        if(cliquescount==1):
          A_1 = nx.adjacency_matrix(H)
          idx_1 = subgraph_list.copy()
          #print('A1', A_1.todense())
        elif(cliquescount==2):
          A_2 = nx.adjacency_matrix(H)
          idx_2 = subgraph_list.copy()
        elif(cliquescount==3):
          A_3 = nx.adjacency_matrix(H)
          idx_3 = subgraph_list.copy()
        #print('\nadj matrix of partition ' + str(int(count+1)) + '\n', A.todense())
        count = count + 1
        G.remove_nodes_from([n for n in G if n in set(subgraph_list)]) 
      #    nx.draw_networkx(G)  
  return A_1, A_2, A_3, idx_1, idx_2, idx_3
  #print('number of partitions', count)
  #print('number of cliques obtained', cliquescount)
  #print('number of non cliques obtained', nocliquescount)
