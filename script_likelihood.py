import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
import threading
import os
import time

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
        links = np.loadtxt(links_path, delimiter=",",dtype=int)
        branchlength = np.loadtxt(branchlength_path, delimiter=",")
        msa = []   
        with open(msa_path) as f:
            for line in f:
                index, seq = line.strip().split()
                msa.append((int(index), seq))

        # set nodes
        nodes_id = set(links.flatten())
        self.nodes = {id:node(id) for id in nodes_id}

        # iterate over each link to set the correct linkage between nodes
        for i,link in enumerate(links) :
            self.nodes[link[0]].children.append(self.nodes[link[1]])
            self.nodes[link[0]].isleaf = False
            self.nodes[link[1]].parent=self.nodes[link[0]]
            self.nodes[link[0]].dist_to_child.append(branchlength[i])
        
        # also save sequences in leaves
        for id,sequence in enumerate(msa):
            self.nodes[id+1].seq = sequence[1]

    def compute_likelihood(self,transition_matrix,mut_rate = 1e-8):

        def encode(nucleotide):
            mapping = {
                "A": [1, 0, 0, 0],
                "C": [0, 1, 0, 0],
                "G": [0, 0, 1, 0],
                "T": [0, 0, 0, 1]
            }
            return np.array(mapping.get(nucleotide, [0, 0, 0, 0]))

        def reccurent_prob_vect(node,nucleotid_pos,Q):  
            if node.isleaf :  # check if leaf
                return (encode(node.seq[nucleotid_pos]))      

            else : # if not leaf
                t1,t2 = node.dist_to_child

                proba_child_1 = reccurent_prob_vect(node.children[0],nucleotid_pos,Q)
                proba_child_2 = reccurent_prob_vect(node.children[1],nucleotid_pos,Q)

                proba_child_1 = expm(Q*t1) @ proba_child_1
                proba_child_2 = expm(Q*t2) @ proba_child_2

                return(proba_child_1 * proba_child_2)

        Q = mut_rate*transition_matrix
        pi = stationary_distribution(Q)

        for node_id in self.nodes :
            if self.nodes[node_id].parent == None :
                id_root = node_id
                break

        log_results = []  

        for nucleotid_pos in range(len(self.nodes[1].seq)):
            result = reccurent_prob_vect(self.nodes[id_root],nucleotid_pos,Q)
            result *= pi
            log_results.append(np.log(result))

        return np.sum(log_results)

    def compute_likelihood_threading(self, transition_matrix, mut_rate = 1e-8, n_threads = 10):

        def encode(nucleotide):
            mapping = {
                "A": [1, 0, 0, 0],
                "C": [0, 1, 0, 0],
                "G": [0, 0, 1, 0],
                "T": [0, 0, 0, 1]
            }
            return np.array(mapping.get(nucleotide, [0, 0, 0, 0]))

        def reccurent_prob_vect(node,nucleotid_pos,Q):  
            if node.isleaf :  # check if leaf
                return (encode(node.seq[nucleotid_pos]))      

            else : # if not leaf
                t1,t2 = node.dist_to_child

                proba_child_1 = reccurent_prob_vect(node.children[0],nucleotid_pos,Q)
                proba_child_2 = reccurent_prob_vect(node.children[1],nucleotid_pos,Q)

                proba_child_1 = expm(Q*t1) @ proba_child_1
                proba_child_2 = expm(Q*t2) @ proba_child_2

                return(proba_child_1 * proba_child_2)
        
        def one_thread(results, nucl_pos, id_root, Q, pi):
            for nucleotid_pos in nucl_pos:
                result = reccurent_prob_vect(self.nodes[id_root],nucleotid_pos,Q)
                result *= pi
                results[nucleotid_pos] = np.log(result)

        Q = mut_rate*transition_matrix
        pi = stationary_distribution(Q)

        for node_id in self.nodes :
            if self.nodes[node_id].parent == None :
                id_root = node_id
                break

        seq_len = len(self.nodes[1].seq)
        pos_per_thread = np.array_split(range(seq_len), n_threads)
        results = [np.NAN] * seq_len

        threads = []
        for t in range(n_threads):
            x = threading.Thread(target=one_thread, args=(results, pos_per_thread[t], id_root, Q, pi))
            threads.append(x)
            x.start()
        for t in threads:
            t.join()

        return np.sum(results)
    
os.chdir("/home/samuel/python/advance_programming_master/project_likelihood")

print("################################")
print("\n")
print("Calculating tree likelihood:")

transition_matrix = np.full((4,4),1)-4*np.eye(4)
print("Transition matrix:")
print(transition_matrix)

tr = tree(msa_path="msa.dat",links_path="table.dat",branchlength_path="branchlength.dat")

print("Without threading:")
t0 = time.time()
likelihood = tr.compute_likelihood(transition_matrix=transition_matrix)
t1 = time.time()
print(f"tree likelihood is {likelihood}")
print(f"time to run: {t1-t0} seconds")

n_t = 10
print(f"With threading (number of threads = {n_t}):")
t0 = time.time()
likelihood = tr.compute_likelihood_threading(transition_matrix=transition_matrix, n_threads=n_t)
t1 = time.time()
print(f"tree likelihood is {likelihood}")
print(f"time to run: {t1-t0} seconds")

