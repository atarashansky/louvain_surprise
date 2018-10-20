from numba import jit, autojit
import numpy as np
   
def contract_graph(nn,cl,clu):
    n=[]
    for i in clu:
        n.append(nn[cl==i,:].sum(0))
    n=np.vstack(n)
    
    n2=[]
    for i in clu:
        n2.append(n[:,cl==i].sum(1)[:,None])
    
    n2=np.hstack(n2)
    return n2
def louvain_surprise(nn, remove_self_loops=True, init = None):
    if remove_self_loops:
        nn = nn.copy()
        np.fill_diagonal(nn,0)
    
    m = nn.sum()
    M = nn.shape[0]*(nn.shape[0]-1)
    
    nc = nn.shape[0]
    
    if(init is None):
        cluster_labels = np.arange(nn.shape[0])
    else:
        cluster_labels = init
        
    active_indices = list(np.unique(cluster_labels))
    mint = 0
    for i in active_indices:
        mint+=nn[cluster_labels==i,:][:,cluster_labels==i].sum()
    
    num_cells = np.ones(cluster_labels.size).astype('int')
    
    num_nodes = np.zeros(num_cells.size)
    num_nodes[active_indices] = np.unique(cluster_labels,return_counts=True)[1]
    
    Mint = 0
    for i in active_indices:
        Mint += num_nodes[i]*(num_nodes[i]-1)
        
    q = mint/m
    qb = Mint/M
    if q == 0 or qb == 0:
        Q=0
    else:
        Q = m*(q*np.log(q/qb) + (1-q)*np.log((1-q)/(1-qb)))
    
    n=[]
    for i in active_indices:
        n.append(nn[cluster_labels==i,:].sum(0)+nn[:,cluster_labels==i].sum(1))
    pairwise=np.vstack(n).T
    
    
    Q,mint,Mint,f=surprise(nn,mint,Mint,m,M,Q,cluster_labels,pairwise,num_nodes,num_cells,active_indices)
    
    n1 = len(active_indices)+1
    n2 = len(active_indices)
    
    memberships=[]
    clu=np.unique(cluster_labels)
    for i in clu:
        memberships.append(list(np.where(cluster_labels==i)[0]))
    
    while n2 < n1:
        num_cells=np.array(num_nodes)[active_indices]
        nn=contract_graph(nn,cluster_labels,clu)
        
        
        cluster_labels = np.arange(nn.shape[0])
        num_nodes = num_cells.copy()
        pairwise = nn + nn.T
        active_indices = list(np.unique(cluster_labels))
        
        Q,mint,Mint,f=surprise(nn,mint,Mint,m,M,Q,cluster_labels,pairwise,num_nodes,num_cells,active_indices)
        n1 = n2
        n2 = len(active_indices)

        clu=np.unique(cluster_labels)
        
        delete=[]
        for i in range(n2):
            idx = np.where(cluster_labels==clu[i])[0]        
            for j in range(1,idx.size):
                memberships[idx[0]].extend(memberships[idx[j]])
                delete.append(idx[j])
        
        for i in sorted(delete,reverse=True):
            del memberships[i]
        
    cluster_labels = np.zeros(nc,dtype='int')
    for i,ii in enumerate(memberships):
        cluster_labels[ii] = i
    return cluster_labels

@jit(nopython=True)
def surprise(nn,mint,Mint,m,M,Q,cluster_labels,pairwise,num_nodes,num_cells,active_indices,eps=1e-5):       
    Qold = Q-1
    Qnew = Q
    f=0
    while Qnew - Qold > eps:
        f+=1
        for i in range(len(cluster_labels)):   
            k = cluster_labels[i]            
            
            dQs=0
            chosen = -1
                        
            sw = nn[i,i]
            fmint=0
            fMint=0
            
            for j in active_indices:   
                if j != k and pairwise[i,j] > 0:
                
                    q = mint/m
                    qb = Mint/M
                                    
                    mintp = mint - pairwise[i,k] + pairwise[i,j] + 2*sw
                    Mintp = Mint + 2*num_cells[i]*(num_cells[i]+num_nodes[j]-num_nodes[k])
                    
                    qp = mintp/m
                    qbp = Mintp/M
                    
                    if q == 0 or qb == 0 or q == 1 or qb == 1:
                        d1 = 0
                    else:
                        d1 = q*np.log(q/qb) + (1-q)*np.log((1-q)/(1-qb))
                    
                    if qp == 0 or qbp == 0 or qp == 1 or qbp == 1:
                        d2 = 0
                    else:   
                        d2 = qp*np.log(qp/qbp) + (1-qp)*np.log((1-qp)/(1-qbp))
                        
                    dQ = m * (d2 - d1)
                    
                    if(dQ > dQs):
                        dQs = dQ
                        chosen = j
                        fmint=mintp
                        fMint=Mintp
            
            if chosen != -1:
                j = chosen            
                
                mint = fmint
                Mint = fMint
                
                num_nodes[k]-=num_cells[i]
                num_nodes[j]+=num_cells[i]                
                
                pairwise[:,k] += -nn[:,i] - nn[i,:]
                pairwise[:,j] += nn[:,i] + nn[i,:]
                
                if(num_nodes[k] == 0):
                    active_indices.remove(k)

                cluster_labels[i] = j                         
                Q+=dQs
            
        Qold = Qnew
        Qnew = Q

    return Q,mint,Mint,f
