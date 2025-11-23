import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
import multiprocessing
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

def stationary_distribution(Q):
    # return stationary vector for any transition matrix 
    # chatGPT generated   
    U, S, Vt = svd(Q.T)
    pi = Vt[-1, :]    
    pi = np.abs(pi)
    pi /= np.sum(pi)
    return pi

class node():
    def __init__(self,id): 
        # defining basic node attributes
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
            self.nodes[link[0]].children.append(self.nodes[link[1]]) # parent and children attributes refers to another node
            self.nodes[link[0]].isleaf = False
            self.nodes[link[1]].parent=self.nodes[link[0]]
            self.nodes[link[0]].dist_to_child.append(branchlength.iloc[0, i])
        
        # also save sequences in leaves
        for i, sequence in msa.iterrows():
            self.nodes[sequence[0]].seq = sequence[1]

    def compute_likelihood(self, transition_matrix, mut_rate = 1e-8):

        def encode(nucleotide): # encode nucleotides into numbers
            mapping = {
                "A": [1, 0, 0, 0],
                "C": [0, 1, 0, 0],
                "G": [0, 0, 1, 0],
                "T": [0, 0, 0, 1]
            }
            return np.array(mapping[nucleotide])

        def reccurent_prob_vect(node,Q):  

            if node.isleaf :  # check if leaf
                return (np.transpose(np.array([encode(i) for i in node.seq])))

            else : # if not leaf
                t1,t2 = node.dist_to_child

                proba_child_1 = reccurent_prob_vect(node.children[0],Q)
                proba_child_2 = reccurent_prob_vect(node.children[1],Q)

                proba_child_1 = expm(Q*t1) @ proba_child_1
                proba_child_2 = expm(Q*t2) @ proba_child_2

                return(proba_child_1 * proba_child_2)                
                
        Q = mut_rate*transition_matrix
        pi = stationary_distribution(Q)

        for node_id in self.nodes :
            if self.nodes[node_id].parent == None :
                id_root = node_id
                break
        
        results = reccurent_prob_vect(self.nodes[id_root],Q)
        results =  pi @ results 

        return np.sum(np.log(results))

    
os.chdir("/home/samuel/python/advance_programming_master/project_likelihood/dataset")
#os.chdir("C:\\Users\\Eric\\OneDrive\\Documents\\Travail\\Master\\semester1\\advanced_python_programming\\project_likelihood\\project_likelihood\\dataset")

print("################################")
print("Calculating tree likelihood:")

transition_matrix = np.full((4,4),1)-4*np.eye(4)
print("Transition matrix:")
print(transition_matrix)

tr = tree(msa_path="ENSG00000013016_EHD3_NT.msa.dat",links_path="ENSG00000013016_EHD3_NT.table.dat",branchlength_path="ENSG00000013016_EHD3_NT.branchlength.dat")

mu = 0.01

print("One value:")
t0 = time.time()
likelihood = tr.compute_likelihood(transition_matrix=transition_matrix, mut_rate=mu)
t1 = time.time()
print(f"tree likelihood is {likelihood} with mu = {mu}")
print(f"time to run: {t1-t0} seconds")
print("------------------------------")

def process_screen(results, mus, which_mu, transition_matrix, tree):
    for i in which_mu:
        results[i] = tree.compute_likelihood(transition_matrix, mut_rate = mus[i])

def screen_mut_rate(tree, mus, transition_matrix, n_process = 10):

    if __name__ == "__main__":   

        n_mu = len(mus)
        which_mu = np.array_split(range(n_mu), n_process)
        results = multiprocessing.Array("f", n_mu)

        process = []
        for p in range(n_process):
            x = multiprocessing.Process(target=process_screen, args=(results, mus, which_mu[p], transition_matrix, tree))
            process.append(x)
            x.start()

        for p in process:
            p.join()

        return results

print("Screen mutation rate:")

start = 1e-8
stop = 4
mus = np.linspace(start, stop, 200)

print(f"Testing values from {start} to {stop}")

print()
print("One process:")
t0 = time.time()
r1 = screen_mut_rate(tr, mus, transition_matrix, n_process=1)
t1 = time.time()
print(f"time to run: {t1-t0} seconds")
print()
print("Multiple-processing:")
t0 = time.time()
r2 = screen_mut_rate(tr, mus, transition_matrix, n_process=10)
t1 = time.time()
print(f"time to run: {t1-t0} seconds")

r = np.array(r2)

max_mu = mus[np.where(r == max(r))]

print(f"mu with the highest likelihood is {max_mu}")

plt.figure(figsize=(20,20))
plt.plot(mus, r2)
plt.axvline(max_mu, c="red")
plt.axhline(max(r), c="red")
plt.title(f"Likelihood parameter sweep, best mu = {max_mu}", fontsize=24)
plt.show()