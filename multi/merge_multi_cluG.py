from hidef.hidef_finder import *

def jaccard_matrix(matA, matB, threshold=0.75):
    both = matA.dot(matB.T)
    either = (np.tile(matA.getnnz(axis=1), (matB.shape[0],1)) + matB.getnnz(axis=1)[:, np.newaxis]).T -both
    jac = 1.0*both/either
    index = np.where(jac > threshold)
    return index


def collapse_cluster_graph(mat, components, threshold=100):
    collapsed_clusters = []
    for component in components:
        mat_sub = mat[component, :]
        participate_index = np.mean(mat_sub, axis=0)
        threshold_met = participate_index *100 > threshold
        threshold_met = threshold_met.astype(int)
        collapsed_clusters.append(threshold_met)
    return collapsed_clusters


par = argparse.ArgumentParser()
par.add_argument('--g', required=True, help='A tab separated file for the input graph')
par.add_argument('--clugs', nargs='+', help='multiple cluG objects created by the hidef_finder.py script, provide in order')
# par.add_argument('--neighbors', type=int, default=3, help='since it is expensive to pairwise compare everything, compare with neighbors in the order of files')
par.add_argument('--precut', type=int, default=5, help='pre-remove small components from each cluG')
par.add_argument('--out', required=True, help='prefix of output files')
par.add_argument('--j', default=0.75, type=float, help='take this fraction of clusters')
args = par.parse_args()

clug_list = []
for x in args.clugs: # each is a nx.Graph
    clug = pickle.load(open(x, 'rb'))
    clug_list.append(clug)


# prefilter
cluster_long_list = []
cluster_source_id = []
for i in range(len(clug_list)): # need to make sure every file have the same node indices
    cluG = clug_list[i]
    components = [c for c in nx.connected_components(cluG) if len(c) >= args.precut]
    components = sorted(components, key=len, reverse=True)
    for component in components:
        clusters = [cluG.nodes[v]['data'].binary for v in component]
        cluster_long_list.extend(clusters)
        cluster_source_id.extend([i for _ in component])

# concatenate all clusters, and calculate Jaccard

mat = np.stack(cluster_long_list)
matsp = sp.sparse.coo_matrix(mat, )
matsp = matsp.tocsr()

jacmat = jaccard_matrix(matsp, matsp)

# build new cluster graph

Gcli = nx.Graph()
for i in range(len(jacmat[0])):
    na, nb = jacmat[0][i], jacmat[1][i]
    if na != nb:
        Gcli.add_edge(na, nb) # not associated with binary

components_new = []
clic_percolation = list(k_clique_communities(Gcli, 5))
for clic in clic_percolation:
    clic = np.array(list(clic))
    components_new.append(clic)

cluG_collapsed = collapse_cluster_graph(mat, components_new, args.j)
len_components = [len(c) for c in components_new]
id_components = []
for c in components_new:
    id_c = [cluster_source_id[ci] for ci in c]
    id_components.append(id_c)

cluG_collapsed_w_len = [(cluG_collapsed[i], len_components[i], id_components[i]) for i in range(len(cluG_collapsed))]
cluG_collapsed_w_len = sorted(cluG_collapsed_w_len, key=lambda x: np.sum(x[0]), reverse=True)  # sort by cluster size

cluG_collapsed.insert(0, np.ones(len(cluG_collapsed[0]), ))
len_components.insert(0, 0)
weaver = weaver.Weaver()
T = weaver.weave(cluG_collapsed, boolean=True, assume_levels=False,
                 merge=True, cutoff=args.j)  #

G = ig.Graph.Read_Ncol(args.g)

output_nodes(weaver, G, args.o, len_component)
output_edges(weaver, G, args.o)

# write an auxiliary file; which file are members of an ensemble coming from
with open(args.o +'_source_layer.txt', 'w') as fh:
    for i in range(len(cluG_collapsed_w_len)):
        ids = ','.join([str(x) for x in cluG_collapsed_w_len[i][2]])
        fh.write(ids + '\n')