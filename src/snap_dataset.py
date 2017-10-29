import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans, DBSCAN

# SNAP DIR
SNAP_DATASET_DIR = "dataset/snap_raw"
SNAP_DATASET_NAMES = ['Brightkite', 'Gowalla']
EDGE_SUFFIX = "_edges.txt"
CHECKIN_SUFFIX = "_totalCheckins.txt"

#CONSTANT
FZERO = 0.0000001

#REGION
REGIONS = {'na': [(9, -170), (71, -46)],
           'usw': [(32, -123), (49, -104)]}


def __checkin_process(line):
    items = line.strip().split('\t')
    if len(items) < 5:
        print(items)
        return None
    if abs(float(items[2])) < FZERO and abs(float(items[3])) < FZERO:
        # print(items)
        return None
    return {'user': items[0],
            #'time': time.mktime(time.strptime(items[1], "%Y-%m-%dT%H:%M:%SZ")),
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
    # print(edges)
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


def region_dataset(name, active_threshold, region):
    edges, checkins = load_snap_dataset(name)

    min_lat, min_lon = REGIONS[region][0]
    max_lat, max_lon = REGIONS[region][1]
    checkins = checkins[(checkins.latitude >= min_lat) &
                        (checkins.latitude <= max_lat) &
                        (checkins.longitude >= min_lon) &
                        (checkins.longitude <= max_lon)]
    checkins = checkins.assign(counts=checkins.groupby(['user'])['user'] \
                               .transform(np.size))
    fil = checkins[checkins.counts >= active_threshold]
    fil = fil[['user', 'latitude', 'longitude']]
    user_list = pd.unique(fil['user'])
    print(len(user_list), fil.shape)
    g = nx.DiGraph()
    g.add_edges_from(edges)
    sub = g.subgraph(user_list)
    new_edges = [{'u1': e[0], 'u2': e[1]} for e in sub.edges()]

    edge_table = pd.DataFrame(new_edges)
    fil.to_csv(os.path.join(SNAP_DATASET_DIR,
                            '_'.join([name, region, str(active_threshold)])
                            + CHECKIN_SUFFIX)
               )
    edge_table.to_csv(os.path.join(SNAP_DATASET_DIR,
                                   '_'.join([name,
                                             region,
                                             str(active_threshold)])
                                + EDGE_SUFFIX),
                      index=False)
    print('done')


def gen_region_w2f_dataset(name, active_threshold, region, granularity, clustering=None):
    fil = pd.read_csv(os.path.join(SNAP_DATASET_DIR,
                                   '_'.join([name,
                                             region,
                                             str(active_threshold)])
                                   + CHECKIN_SUFFIX),
                      index_col=0)
    edge = pd.read_csv(os.path.join(SNAP_DATASET_DIR,
                                    '_'.join([name,
                                              region,
                                              str(active_threshold)])
                                    + EDGE_SUFFIX))
    # apply granularity
    enlarge = 1 / granularity
    fil.loc[:, 'latitude'] = fil.apply(lambda x: int(x['latitude'] * enlarge),
                                        axis=1)
    fil.loc[:, 'longitude'] = fil.apply(lambda x: int(x['longitude'] * enlarge),
                                        axis=1)
    locs = set(zip(fil.latitude, fil.longitude))
    # location num
    print(len(locs))

    par = 0

    if clustering is None:
        aux = {loc: i for i, loc in enumerate(locs)}
        fil.loc[:, 'locid'] = fil.apply(lambda x: aux[(x['latitude'],
                                                       x['longitude'])],
                                      axis=1)

    elif clustering['model'] == 'DBSCAN':
        cluster = DBSCAN(clustering['eps'], clustering['min_samples'],
                         n_jobs=clustering['n_jobs']).fit(list(locs))
        aux = {loc: cluster.labels_[i] for i, loc in enumerate(locs)}
        print(len(set(cluster.labels_)), len(pd.unique(fil['user'])))
        fil.loc[:, 'locid'] = fil.apply(lambda x: aux[(x['latitude'],
                                                       x['longitude'])],
                                      axis=1)
        par = clustering['eps']

    fil = fil[['user', 'locid']]
    fil.rename(columns={'user': 'uid'}, inplace=True)
    fil.to_csv('dataset/%s_%s_%d_%d.checkin' % (name,
                                                region,
                                                par,
                                                active_threshold))
    edge.to_csv('dataset/%s_%s_%d_%d.friends' % (name,
                                                 region,
                                                 par,
                                                 active_threshold))



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


def check_friend_list(name):
    checkin = pd.read_csv('dataset/%s_20.checkin' % name, index_col=0)
    user_num = pd.unique(checkin['uid'])
    friends = pd.read_csv('dataset/%s_20.friends' % name)
    user = set(pd.unique(friends['u1'])) | set(pd.unique(friends['u2']))
    count = 0
    for u in user:
        if u not in user_num:
            count += 1

    print(count)


def gen_Kmeans_dataset(name, active_threshold, granularity, k):
    edges, checkins = load_snap_dataset(name)
    # print(edges)
    checkins['counts'] = checkins.groupby(['user'])['user'].transform(np.size)
    # filter < active_threshold
    fil = checkins[checkins['counts'] >= active_threshold]
    # k means
    # locations = list(zip(fil['latitude'], fil['longitude']))
    # fil['loc'] = KMeans(k, n_jobs=-1).fit_predict(locations)

    # apply granularity
    enlarge = 1 / granularity
    fil['latitude'] = fil.apply(lambda x:int(x['latitude'] * enlarge),
                                axis=1)
    fil['longitude'] = fil.apply(lambda x: int(x['longitude'] * enlarge),
                                 axis=1)
    locs = list(zip(fil.latitude, fil.longitude))
    print(len(locs))
    print('here')
    # fil['loc'] = DBSCAN(eps=1, min_samples=100, n_jobs=-1).fit_predict(locs)
    fil['loc'] = KMeans(20, n_jobs=-1).fit_predict(locs)

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
    new_table.to_csv('dataset/%s_%dM_%d.checkin' % (name, k, active_threshold))
    edge_table.to_csv('dataset/%s_%dM_%d.friends' % (name, k, active_threshold),
                      index=False)
    print('done')


# gen_w2f_dataset(SNAP_DATASET_NAMES[1], 20, 0.01)
# remap_locid("Gowalla_20")
# gen_Kmeans_dataset(SNAP_DATASET_NAMES[0], 20, 0.001, 10000)
# dataset_summary(SNAP_DATASET_NAMES[0] + '_10000M')
# region_dataset(SNAP_DATASET_NAMES[1], 20, 'na')
"""
for i in [5, 10, 15, 25, 30]:
    gen_region_w2f_dataset(SNAP_DATASET_NAMES[1], 20, 'na', 0.001,
                            {'model': 'DBSCAN',
                            'eps': i,
                            'min_samples': 1,
                            'n_jobs': -2})
"""

check_friend_list('Gowalla_na_5')
