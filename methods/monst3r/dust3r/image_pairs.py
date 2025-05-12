# --------------------------------------------------------
# utilities needed to load image pairs
# --------------------------------------------------------
import numpy as np
import torch
from typing import List, Tuple
import random

def make_pairs_clique(
    imgs,
    vertex_list: List[Tuple[int,int]],
    scene_graph: str = 'complete',
    prefilter=None,
    symmetrize: bool = True,
    force_symmetrize: bool = False,
    num_cross_edges: int = 5
):
    """
    Like make_pairs, but first builds edges *inside* each [start,end] block
    in vertex_list, then randomly adds `num_cross_edges` *across* each
    pair of adjacent blocks.
    """
    # 1) Intra‑cluster
    pairs = []
    for start, end in vertex_list:
        # inclusive ranges
        sub_imgs = imgs[start:end]
        # build *just* the intra‑cluster edges (no symm / no prefilter yet)
        sub_pairs = make_pairs(
            sub_imgs,
            scene_graph=scene_graph,
            prefilter=prefilter,
            symmetrize=symmetrize,
        )
        pairs.extend(sub_pairs)

    # 2) Cross‑cluster edges
    #    for each adjacent cluster pair
    if num_cross_edges > 0 and len(vertex_list) >= 2:
        # build list of index ranges
        clusters = [list(range(s, e)) for s,e in vertex_list]
        for k in range(len(clusters)-1):
            c1, c2 = clusters[k], clusters[k+1]
            # all possible cross edges
            all_cross = [(i, j) for i in c1 for j in c2 if i != j]
            # sample without replacement
            chosen = random.sample(all_cross, min(num_cross_edges, len(all_cross)))
            for i,j in chosen:
                pairs.append((imgs[i], imgs[j]))
    
    if (symmetrize and 
        not scene_graph.startswith('oneref') and
        not scene_graph.startswith('swin-1')) \
       or len(imgs) == 2 or force_symmetrize:

        # add reversed
        pairs += [(b,a) for (a,b) in pairs]


    # 5) Deduplicate (so no duplicate unordered pairs)
    seen = set()
    final = []
    for a,b in pairs:
        ia, ib = a['idx'], b['idx']
        key = (min(ia,ib), max(ia,ib))
        if key not in seen:
            seen.add(key)
            final.append((a,b))
    print("there are final pairs =", len(final))
    return final    




def make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True, force_symmetrize=False):
    pairs = []
    if scene_graph == 'complete':  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('swin'):
        iscyclic = not scene_graph.endswith('noncyclic')
        try:
            winsize = int(scene_graph.split('-')[1])
        except Exception as e:
            winsize = 3
        pairsid = set()
        if scene_graph.startswith('swinstride'):
            stride = 2
        elif scene_graph.startswith('swin2stride'):
            stride = 3
        else:
            stride = 1
        if scene_graph.startswith('swinskip_start'):
            start = 2
        else:
            start = 1
        for i in range(len(imgs)):
            for j in range(start, stride*winsize + start, stride):
                idx = (i + j)
                if iscyclic:
                    idx = idx % len(imgs)  # explicit loop closure
                if idx >= len(imgs):
                    continue
                pairsid.add((i, idx) if i < idx else (idx, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('logwin'):
        iscyclic = not scene_graph.endswith('noncyclic')
        try:
            winsize = int(scene_graph.split('-')[1])
        except Exception as e:
            winsize = 3
        offsets = [2**i for i in range(winsize)]
        pairsid = set()
        for i in range(len(imgs)):
            ixs_l = [i - off for off in offsets]
            ixs_r = [i + off for off in offsets]
            for j in ixs_l + ixs_r:
                if iscyclic:
                    j = j % len(imgs)  # Explicit loop closure
                if j < 0 or j >= len(imgs) or j == i:
                    continue
                pairsid.add((i, j) if i < j else (j, i))
        for i, j in pairsid:
            pairs.append((imgs[i], imgs[j]))
    elif scene_graph.startswith('oneref'):
        refid = int(scene_graph.split('-')[1]) if '-' in scene_graph else 0
        for j in range(len(imgs)):
            pairs.append((imgs[refid], imgs[j]))

    if (symmetrize and not scene_graph.startswith('oneref') and not scene_graph.startswith('swin-1')) or len(imgs) == 2 or force_symmetrize:
        pairs += [(img2, img1) for img1, img2 in pairs]

    # now, remove edges
    if isinstance(prefilter, str) and prefilter.startswith('seq'):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]))

    if isinstance(prefilter, str) and prefilter.startswith('cyc'):
        pairs = filter_pairs_seq(pairs, int(prefilter[3:]), cyclic=True)

    return pairs


def sel(x, kept):
    if isinstance(x, dict):
        return {k: sel(v, kept) for k, v in x.items()}
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x[kept]
    if isinstance(x, (tuple, list)):
        return type(x)([x[k] for k in kept])


def _filter_edges_seq(edges, seq_dis_thr, cyclic=False):
    # number of images
    n = max(max(e) for e in edges) + 1

    kept = []
    for e, (i, j) in enumerate(edges):
        dis = abs(i - j)
        if cyclic:
            dis = min(dis, abs(i + n - j), abs(i - n - j))
        if dis <= seq_dis_thr:
            kept.append(e)
    return kept


def filter_pairs_seq(pairs, seq_dis_thr, cyclic=False):
    edges = [(img1['idx'], img2['idx']) for img1, img2 in pairs]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    return [pairs[i] for i in kept]


def filter_edges_seq(view1, view2, pred1, pred2, seq_dis_thr, cyclic=False):
    edges = [(int(i), int(j)) for i, j in zip(view1['idx'], view2['idx'])]
    kept = _filter_edges_seq(edges, seq_dis_thr, cyclic=cyclic)
    print(f'>> Filtering edges more than {seq_dis_thr} frames apart: kept {len(kept)}/{len(edges)} edges')
    return sel(view1, kept), sel(view2, kept), sel(pred1, kept), sel(pred2, kept)
