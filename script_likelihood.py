import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
import threading
import os
import time
import pandas as pd

def stationary_distribution(Q):
    # return stationary vector for any transition matrix    
    U, S, Vt = svd(Q.T)
    pi = Vt[-1, :]    
    pi = np.abs(pi)
    pi /= np.sum(pi)
    return pi

class node():
    def __init__(self,id):
        self.id = id
        self.parent = None
        self.children = [] 
        self.dist_to_child = []
        self.isleaf = True
        self.seq = None
        
class tree():

    def __init__(self,msa_path,links_path,branchlength_path):
        
        # read files 
        links = pd.read_table(links_path, delimiter=",", header=None, dtype = str)
        branchlength = pd.read_table(branchlength_path, delimiter=",", header=None)
        msa = pd.read_table(msa_path, delimiter=" ", header=None)
        self.sequences = msa

        # set nodes
        nodes_id = set(links.values.flatten())
        self.nodes = {id:node(id) for id in nodes_id}

        # iterate over each link to set the correct linkage between nodes
        for i,link in links.iterrows() :
            self.nodes[link[0]].children.append(self.nodes[link[1]])
            self.nodes[link[0]].isleaf = False
            self.nodes[link[1]].parent=self.nodes[link[0]]
            self.nodes[link[0]].dist_to_child.append(branchlength.iloc[0, i])
        
        # also save sequences in leaves
        for i, sequence in msa.iterrows():
            self.nodes[sequence[0]].seq = sequence[1]

    def compute_likelihood(self,transition_matrix,mut_rate = 1e-8, thr = False):

        def encode(nucleotide):
            mapping = {
                "A": [1, 0, 0, 0],
                "C": [0, 1, 0, 0],
                "G": [0, 0, 1, 0],
                "T": [0, 0, 0, 1]
            }
            if nucleotide not in mapping:
                raise ValueError(f"Unknown nucleotide: {nucleotide}")
            return np.array(mapping[nucleotide])

        def reccurent_prob_vect(node,from_nuc,to_nuc,Q):  
            if node.isleaf :  # check if leaf
                return (np.transpose(np.array([encode(i) for i in node.seq[from_nuc:(to_nuc+1)]])))      

            else : # if not leaf
                t1,t2 = node.dist_to_child

                proba_child_1 = reccurent_prob_vect(node.children[0],from_nuc,to_nuc,Q)
                proba_child_2 = reccurent_prob_vect(node.children[1],from_nuc,to_nuc,Q)

                proba_child_1 = expm(Q*t1) @ proba_child_1
                proba_child_2 = expm(Q*t2) @ proba_child_2

                return(proba_child_1 * proba_child_2)
            
        def one_thread(results, from_nuc,to_nuc, id_root, Q, pi,thread_id):
                
                result = reccurent_prob_vect(self.nodes[id_root],from_nuc,to_nuc,Q)
                result =  pi @ result 
                
                results[thread_id] = np.sum(np.log(result))

        Q = mut_rate*transition_matrix
        pi = stationary_distribution(Q)
        seq_len = len(self.sequences.iloc[0,1])

        for node_id in self.nodes :
            if self.nodes[node_id].parent == None :
                id_root = node_id
                break

        if thr > 0:
            #n_threads = int(input("How many threads?\n"))

            results = [np.NAN] *thr

            pos_per_thread = np.array_split(range(seq_len), thr)

            threads = []
            for t in range(thr):

                from_nuc = pos_per_thread[t][0]
                to_nuc = pos_per_thread[t][-1]
                x = threading.Thread(target=one_thread, args=(results, from_nuc,to_nuc, id_root, Q, pi,t))
                threads.append(x)
                x.start()
            for t in threads:
                t.join()

            return np.sum(results)

        
        results = [np.NAN ]
        one_thread(results = results, from_nuc = 0,to_nuc=seq_len, id_root=id_root, Q=Q, pi=pi,thread_id=0)

        return np.sum(results)

    
#os.chdir("/home/samuel/python/advance_programming_master/project_likelihood/dataset")
os.chdir("C:\\Users\\Eric\\OneDrive\\Documents\\Travail\\Master\\semester1\\advanced_python_programming\\project_likelihood\\project_likelihood\\dataset")

print("################################")
print("\n")
print("Calculating tree likelihood:")

transition_matrix = np.full((4,4),1)-4*np.eye(4)
print("Transition matrix:")
print(transition_matrix)

tr = tree(msa_path="ENSG00000112282_MED23_NT.msa.dat",links_path="ENSG00000112282_MED23_NT.table.dat",branchlength_path="ENSG00000112282_MED23_NT.branchlength.dat")

mu = 2

print("Without threading:")
t0 = time.time()
likelihood = tr.compute_likelihood(transition_matrix=transition_matrix,mut_rate=mu)
t1 = time.time()
print(f"tree likelihood is {likelihood}")
print(f"time to run: {t1-t0} seconds")
print("------------------------------")

print(f"With threading:")
t0 = time.time()
likelihood = tr.compute_likelihood(transition_matrix=transition_matrix, thr=1,mut_rate=mu)
t1 = time.time()
print(f"tree likelihood is {likelihood}")
print(f"time to run: {t1-t0} seconds")

