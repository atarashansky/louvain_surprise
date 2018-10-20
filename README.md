# Louvain clustering using asymptotic surprise

Python-only implementation based on this paper:

https://journals.aps.org/pre/pdf/10.1103/PhysRevE.92.022816

Function to call:
```
cluster_labels= louvain_surprise(nn,remove_self_loops=True,init = None)
```
`nn`: directed, weighted or unweighted adjacency matrix

`remove_self_loops`: if `True`, remove all self loops in the graph prior to clustering.

`init`: Initial partition. If `None`, each node starts off as its own community.

returns `cluster_labels`: cluster assignments for each node
