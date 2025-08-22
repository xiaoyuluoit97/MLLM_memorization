import numpy as np
from scipy.stats import zscore
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

tokens_101 = {"en":2733,"ru":713,"es":433,"de":347,"fr":318,"it":162,"pt":146,"pl":130,"nl":73,"tr":71,"ja":164,"vi":116,"id":69,"cs":63,"zh":39,"fa":52,"ar":57,"sv":45,"ro":52,"el":43,"uk":41,"hu":39,"da":29,"fi":25,"no":27,"bg":22,"hi":24,"sk":18,"ko":26,"th":11,"ca":13,"ms":13,"iw":17,"lt":11,"sl":8.8,"mr":14,"bn":7.3,"et":6.9,"lv":7.0,"az":4.4,"gl":2.4,"cy":4.9,"sq":4.0,"ta":3.4,"sr":4.3,"ne":3.2,"lb":1.0,"hy":2.4,"kk":3.1,"ka":2.5,"mt":5.2,"af":1.7,"fil":2.1,"is":2.6,"mk":1.8,"ml":1.8,"mn":2.7,"ur":2.4,"be":2.0,"la":1.3,"eu":1.4,"tg":1.4,"fy":0.4,"te":1.3,"kn":1.1,"ky":1.0,"sw":1.0,"so":1.4,"my":0.9,"uz":0.9,"km":0.6,"sd":1.6,"gu":0.8,"jv":0.3,"zu":0.2,"si":0.8,"eo":0.7,"co":0.2,"ga":0.5,"pa":0.6,"ceb":0.2,"mg":0.2,"ps":0.4,"sn":0.2,"gd":0.4,"ku":0.4,"hmn":0.2,"su":0.1,"ht":0.2,"ha":0.2,"ny":0.1,"am":0.3,"yi":0.3,"lo":0.1,"mi":0.1,"sm":0.09,"ig":0.09,"haw":0.09,"xh":0.06,"st":0.08,"yo":0.05}
tokens_51 = {'en': 47.527, 'de': 45.484, 'ru': 43.763, 'it': 35.484, 'fr': 35.484, 'pt': 29.785, 'es': 26.129, 'vi': 25.054, 'tr': 25.054, 'ja': 22.151, 'ro': 17.957, 'sv': 17.419, 'nl': 17.419, 'pl': 13.548, 'hu': 13.226, 'zh': 13.011, 'da': 10.645, 'ko': 9.892, 'uk': 8.978, 'id': 8.28, 'bg': 7.151, 'ur': 6.398, 'be': 5.86, 'hi': 5.86, 'ar': 5.161, 'ms': 5.0, 'eu': 4.785, 'fa': 4.785, 'tg': 4.247, 'th': 4.032, 'el': 3.925, 'lt': 3.871, 'te': 3.71, 'sw': 3.333, 'fi': 3.333, 'mr': 3.333, 'my': 2.957, 'ky': 2.957, 'uz': 2.957, 'lv': 2.204, 'bn': 1.882, 'az': 1.505, 'ta': 1.344, 'hy': 1.344, 'kk': 1.129, 'ka': 0.86, 'mn': 0.86, 'ml': 0.86, 'iw': 0.86, 'af': 0.86, 'yo': 0.269}

with open('memorizations_mgpt.json', 'r', encoding='utf-8') as f:
    memorizations_gpt101 = json.load(f)
csv_path = "result/lang2vec_phonology_knn.csv"
tokens_dict = tokens_101
meminput = memorizations_gpt101

i_down = 0.1
i_up = 1
i_add = 0.1
CROSS = False

def pearson(dict1,dict2):
    common_keys = sorted(dict1.keys() & dict2.keys())
    x = [dict1[k] for k in common_keys]
    y = [dict2[k] for k in common_keys]

    corr, p_value = pearsonr(x, y)

    print("Pearson correlation:", corr)



thesorld = []
single_point_ratio = []
corr = []

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def save_graph_from_adjacency_with_features(
    A_binary,
    languages,
    tokens,
    memorization,
    filename="graph.png",
    title="Graph"
):

    G = nx.from_numpy_array(A_binary)
    G = nx.relabel_nodes(G, dict(enumerate(languages)))

    components = list(nx.connected_components(G))
    num_subgraphs = len(components)
    print(f"（connected components）: {num_subgraphs}")

    cmap = cm.get_cmap("tab20", num_subgraphs)
    node_colors = {}
    pos = dict()
    subgraph_features = []

    offset_distance = 5.0
    single_point_numbers = 0
    for i, component in enumerate(components):
        subgraph = G.subgraph(component)
        sub_pos = nx.spring_layout(subgraph, k=1.0, iterations=15, seed=42)
        offset = np.array([i * offset_distance, 0.0])
        for node in sub_pos:
            pos[node] = sub_pos[node] + offset
        for node in component:
            node_colors[node] = cmap(i)

        # calculate feature
        n_edges = subgraph.number_of_edges()
        nodes = list(component)

        if n_edges == 0:

            token_sum = np.mean([tokens.get(node, np.nan) for node in nodes])
            mem_sum = np.mean([memorization.get(node, np.nan) for node in nodes])

            single_point_numbers=single_point_numbers+1
        else:

            degs = dict(subgraph.degree())
            all_degs = sum(degs.values())
            token_sum = 0.0
            mem_sum = 0.0
            for node in nodes:
                deg = degs[node]
                deg_weight = deg / all_degs
                t = tokens.get(node, np.nan)
                m = memorization.get(node, np.nan)

                if np.isnan(t) or np.isnan(m):
                    print(f"node {node} miss feature")
                    continue
                if t < -100 or m < -100: 
                    print(f"node {node} wrong value (token={t}, mem={m})")

                token_sum += t * deg_weight
                mem_sum += m * deg_weight

        subgraph_features.append({
            "subgraph_id": i,
            "nodes": nodes,
            "GraphToken": token_sum,
            "GraphMem": mem_sum
        })

    # draw
    edge_weights = [1.5 for _ in G.edges()]
    plt.figure(figsize=(8 + num_subgraphs, 6))
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=[node_colors[n] for n in G.nodes()])
    plt.title(f"{title} (Subgraphs: {num_subgraphs})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"save to: {filename}")


    print("subgraph aggregation:")
    #for feat in subgraph_features:
        #print(f"  Subgraph {feat['subgraph_id']} ({len(feat['nodes'])} nodes): Token={feat['GraphToken']:.4f}, Mem={feat['GraphMem']:.4f}")

    graph_tokens = [f["GraphToken"] for f in subgraph_features]
    graph_mems = [f["GraphMem"] for f in subgraph_features]

    # Pearson correlation
    if len(graph_tokens) >= 2:
        r, p = pearsonr(graph_tokens, graph_mems)
        print(f"\nPearson Token vs Mem）: r = {r:.4f}, p = {p:.4e}")
    else:
        print("\n less subgraph")

    return len(subgraph_features), single_point_numbers,r if len(graph_tokens) >= 2 else None




def is_symmetric(mat, tol=1e-8):
    return np.allclose(mat, mat.T, atol=tol)
def compute_graph_correlation(csv_path, x_dict, y_dict, ignore_labels=None, lpl_norm=True,threshold=None):
    common_langs = list(set(x_dict) & set(y_dict))


    x_origianl = [x_dict[lang] for lang in common_langs]
    y_origianl = [y_dict[lang] for lang in common_langs]


    #corr, pval = pearsonr(x_origianl, y_origianl)
    #print(f"✅ Pearson correlation: {corr:.4f}")
    #print(f"📉 p-value: {pval:.4g}")
    #print(f"🧩 Common languages used: {len(common_langs)}")


    df = pd.read_csv(csv_path, index_col=0)


    if ignore_labels:
        df = df.drop(index=ignore_labels, columns=ignore_labels, errors='ignore')


    languages = df.columns.tolist()


    missing_in_x = [lang for lang in languages if lang not in x_dict]
    missing_in_y = [lang for lang in languages if lang not in y_dict]

    if missing_in_x:
        print(f"⚠️ x_dict language miss: {missing_in_x}")
    if missing_in_y:
        print(f"⚠️ y_dict language miss: {missing_in_y}")


    valid_languages = [
        lang for lang in languages
        if lang in x_dict and lang in y_dict
    ]


    df = df.loc[valid_languages, valid_languages]
    A = df.values
    A_weighted = df.values.copy()

    if threshold is not None:
        A = (A >= threshold).astype(float)


    np.fill_diagonal(A, 0)

    D = np.diag(np.sum(A, axis=1))

    L = D - A

    if lpl_norm:

        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))  
        L = D_inv_sqrt @ L @ D_inv_sqrt


    x = np.array([x_dict[lang] for lang in valid_languages])
    y = np.array([y_dict[lang] for lang in valid_languages])
    x = np.log(x + 1e-8)


    pearson_corr = pearsonr(x, y)[0]

    total_nodes = len(valid_languages)
    degrees = np.sum(A, axis=1)
    isolated_nodes = np.sum(degrees == 0)

    print(f"threshold = {threshold}")
    print(f"node number: {total_nodes}")
    print(f"isolation node: {isolated_nodes}")
    print(f"pearson corr: {pearson_corr}")



    xLx = x.T @ L @ x
    yLy = y.T @ L @ y
    xLy = x.T @ L @ y
    rho = xLy / (np.sqrt(xLx * yLy))
    print(f"🔗 ρ_G(x, y) (Graph correlation): {rho:.6f}")
    thesorld.append(round(float(threshold), 2) if threshold is not None else None)
    single_point_ratio.append(round(float(isolated_nodes/total_nodes), 2))
    corr.append(round(float(rho), 2))
    return {
        "languages": valid_languages,
        "x^T L x": xLx,
        "y^T L y": yLy,
        "x^T L y": xLy,
        "rho_G(x, y)": rho,
        "adjacency": A,
        "edge_weights": A_weighted,
        "tokens_dict":x,
        "memorization_dict":y,
    }


cross_cor_list = []
single_point_list = []
sub_graph_list = []
while i_down <= i_up:
    results = compute_graph_correlation(csv_path, tokens_dict, meminput, ignore_labels=ignore,
                                        lpl_norm=False, threshold=i_down)

    if CROSS:
        tokens_dict_temp = dict(zip(results["languages"], results["tokens_dict"]))
        memorization_dict_temp = dict(zip(results["languages"], results["memorization_dict"]))
        sub_graph, single_numbers,cross_cor = save_graph_from_adjacency_with_features(results["adjacency"], results["languages"],
                                                                   tokens_dict_temp, memorization_dict_temp,
                                                                   filename="your_graph.png")
        cross_cor_list.append(round(float(cross_cor or 0.0), 2))

        sub_graph_list.append(sub_graph)
        single_point_list.append(single_numbers)
    i_down += i_add

if not CROSS:
    results = compute_graph_correlation(csv_path, tokens_dict, meminput, ignore_labels=ignore,lpl_norm=False, threshold=None)



#sub_graph,cross_cor=save_graph_from_adjacency_with_features(results["adjacency"], results["languages"], tokens_dict , memorization_dict, filename="your_graph.png")

print("thesorld:",thesorld)
print("sub graph:",sub_graph_list)
print("signle number:",single_point_list)
print("inter-corr:",corr)
print("cross-corr:",cross_cor_list)
print("single_point_ratio:",single_point_ratio)

from scipy.stats import pearsonr
import numpy as np
import pandas as pd


def compare_models(mem1, mem2, name1, name2):
    common_langs = set(mem1) & set(mem2)
    v1 = np.array([mem1[lang] for lang in common_langs])
    v2 = np.array([mem2[lang] for lang in common_langs])

    r, _ = pearsonr(v1, v2)
    delta_median = np.median(v2 - v1)
    delta_total = np.sum(v2) - np.sum(v1)

    return {
        'Model Pair': f"{name2} vs. {name1}",
        'Pearson r': round(r, 4),
        'Median Δ (Mem)': round(delta_median, 4),
        'Total Δ (Mem)': round(delta_total, 4),
        '# Languages': len(common_langs)
    }

import numpy as np

def get_median(mem_dict_or_list):

    if isinstance(mem_dict_or_list, dict):
        values = list(mem_dict_or_list.values())
    elif isinstance(mem_dict_or_list, list):
        values = mem_dict_or_list
    else:
        raise ValueError("Input must be a dict or list.")

    median_val = np.median(values)
    return round(median_val, 4)


results = [
    #compare_models(memorizations_gpt1_3B, memorizations_gpt1_3B_150, "100", "150"),
    #compare_models(memorizations_bleu_gpt1_3B, memorizations_bleu_gpt1_3B_150, "100", "150"),
    #compare_models(memorizations_rougel_gpt1_3B, memorizations_rougel_gpt1_3B_150, "100", "150")
]
#print(get_median(memorizations_gpt1_3B))
#df = pd.DataFrame(results)
#print(df)
