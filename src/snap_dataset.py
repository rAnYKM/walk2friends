import os
import time
import numpy as np
import pandas as pd
import networkx as nx

# SNAP DIR
SNAP_DATASET_DIR = "dataset/snap_raw"
SNAP_DATASET_NAMES = ['Brightkite', 'Gowalla']
EDGE_SUFFIX = "_edges.txt"
CHECKIN_SUFFIX = "_totalCheckins.txt"

#CONSTANT
FZERO = 0.0000001

def __checkin_process(line):
    items = line.strip().split('\t')
    if len(items) < 5:
        print(items)
        return None
    if abs(float(items[2])) < FZERO and abs(float(items[3])) < FZERO:
        # print(items)
        return None
    return {'user': items[0],
            'time': time.mktime(time.strptime(items[1], "%Y-%m-%dT%H:%M:%SZ")),
            'latitude': float(items[2]),
            'longitude': float(items[3]),
            'location': items[4]}


def load_snap_dataset(name):
    with open(os.path.join(SNAP_DATASET_DIR, name + EDGE_SUFFIX), 'r') as fp:
        edges = [line.strip().split('\t') for line in fp]

    with open(os.path.join(SNAP_DATASET_DIR, name + CHECKIN_SUFFIX), 'r') as fp:
        checkins_list = []
        for line in fp:
            checkin = __checkin_process(line)
            if checkin is None:
                continue
            else:
                checkins_list.append(checkin)
        checkins = pd.DataFrame(checkins_list)

    return edges, checkins


def gen_w2f_dataset(name, active_threshold, granularity):
    edges, checkins = load_snap_dataset(name)
    print(edges)
    checkins['counts'] = checkins.groupby(['user'])['user'].transform(np.size)
    # filter < active_threshold
    fil = checkins[checkins['counts'] >= active_threshold]
    # apply granularity
    enlarge = 1/granularity
    fil['loc'] = fil.apply(lambda x: '%dL%d' % (int(x['latitude'] * enlarge),
                                         int(x['longitude'] * enlarge)),
                           axis=1)

    # only keep location id, user
    new_table = fil[['user', 'loc']]
    print(len(pd.unique(new_table['loc'])))
    print(new_table)

    user_list = pd.unique(new_table['user'])
    # new_edges = [{'u1': e[0], 'u2': e[1]} for e in edges
    #              if e[0] in user_list and e[1] in user_list]

    g = nx.DiGraph()
    g.add_edges_from(edges)
    sub = g.subgraph(user_list)
    new_edges = [{'u1': e[0], 'u2': e[1]} for e in sub.edges()]

    edge_table = pd.DataFrame(new_edges)
    new_table.rename(columns={'user': 'uid', 'loc': 'locid'}, inplace=True)
    new_table.to_csv('dataset/%s_%d.checkin' % (name, active_threshold))
    edge_table.to_csv('dataset/%s_%d.friends' % (name, active_threshold),
                      index=False)
    print('done')


def remap_locid(checkin_file):
    table = pd.read_csv('dataset/%s.checkin' % checkin_file,
                        index_col=0)
    locs = pd.unique(table['locid'])
    ind = {loc: i for i, loc in enumerate(locs)}
    table['locid'] = table.apply(lambda x: ind[x['locid']], axis=1)
    table.to_csv('dataset/%s_2.checkin' % checkin_file)


def dataset_summary(name):
    checkin = pd.read_csv('dataset/%s_20.checkin' % name, index_col=0)
    user_num = pd.unique(checkin['uid'])
    loc_num = pd.unique(checkin['locid'])
    total_check = checkin.shape
    print(name, len(user_num), len(loc_num), total_check)
    friends = pd.read_csv('dataset/%s_20.friends' % name)
    total_link = friends.shape
    print(total_link)


# gen_w2f_dataset(SNAP_DATASET_NAMES[1], 20, 0.01)
# remap_locid("Gowalla_20")
dataset_summary(SNAP_DATASET_NAMES[0])