import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN

# SNAP DIR
SNAP_DATASET_DIR = "dataset/snap_raw"
SNAP_DATASET_NAMES = ['Brightkite', 'Gowalla']
EDGE_SUFFIX = "_edges.txt"
CHECKIN_SUFFIX = "_totalCheckins.txt"

#CONSTANT
FZERO = 0.0000001
KM_PER_RAD = 6371.0088

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
            # 'location': items[4]
            }


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


def gen_w2f__cluster_dataset(name, active_threshold, granularity,
                           par=None, clustering=None):
    edges, checkins = load_snap_dataset(name)
    # print(edges)
    checkins['counts'] = checkins.groupby(['user'])['user'].transform(np.size)
    # filter < active_threshold
    fil = checkins[checkins['counts'] >= active_threshold]
    # apply granularity
    enlarge = 1 / granularity
    fil.loc[:, 'latitude'] = fil.apply(lambda x: int(x['latitude'] * enlarge)
                                                 / enlarge,
                                        axis=1)
    fil.loc[:, 'longitude'] = fil.apply(lambda x: int(x['longitude'] *
                                                      enlarge) / enlarge,
                                        axis=1)
    locs = set(zip(fil.latitude, fil.longitude))
    # location num
    print(len(locs))

    if par is None:
        par = 0

    if clustering is None:
        aux = {loc: i for i, loc in enumerate(locs)}
        fil.loc[:, 'locid'] = fil.apply(lambda x: aux[(x['latitude'],
                                                       x['longitude'])],
                                      axis=1)

    elif clustering['model'] == 'DBSCAN':
        loc_array = np.array(list(locs))
        rads = np.radians(loc_array)
        cluster = DBSCAN(clustering['eps'],
                         clustering['min_samples'],
                         metric='haversine',
                         n_jobs=clustering['n_jobs']).fit(rads)
        aux = {tuple(loc): cluster.labels_[i]
               for i, loc in enumerate(loc_array)}
        cs = list(set(cluster.labels_))
        print(len(cs), max(cs), min(cs), len(pd.unique(fil['user'])))
        fil.loc[:, 'locid'] = fil.apply(lambda x: aux[(x['latitude'],
                                                       x['longitude'])],
                                      axis=1)
        if par is None:
            par = int(clustering['eps'] * KM_PER_RAD * 10)

    user_list = pd.unique(fil['user'])
    g = nx.DiGraph()
    g.add_edges_from(edges)
    sub = g.subgraph(user_list)
    new_edges = [{'u1': e[0], 'u2': e[1]} for e in sub.edges()]

    edge = pd.DataFrame(new_edges)
    print('done')

    fil = fil[['user', 'locid']]
    fil.rename(columns={'user': 'uid'}, inplace=True)
    fil.to_csv('dataset/%s_cluster_%d_%d.checkin' % (name,
                                             par,
                                             active_threshold))
    edge.to_csv('dataset/%s_cluster_%d_%d.friends' % (name,
                                              par,
                                              active_threshold),
                index=False)



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


def gen_region_w2f_dataset(name, active_threshold, region, granularity,
                           par=None, clustering=None):
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
    fil.loc[:, 'latitude'] = fil.apply(lambda x: int(x['latitude'] * enlarge)
                                                 / enlarge,
                                        axis=1)
    fil.loc[:, 'longitude'] = fil.apply(lambda x: int(x['longitude'] *
                                                      enlarge) / enlarge,
                                        axis=1)
    locs = set(zip(fil.latitude, fil.longitude))
    # location num
    print(len(locs))

    if par is None:
        par = 0

    if clustering is None:
        aux = {loc: i for i, loc in enumerate(locs)}
        fil.loc[:, 'locid'] = fil.apply(lambda x: aux[(x['latitude'],
                                                       x['longitude'])],
                                      axis=1)

    elif clustering['model'] == 'DBSCAN':
        loc_array = np.array(list(locs))
        rads = np.radians(loc_array)
        cluster = DBSCAN(clustering['eps'],
                         clustering['min_samples'],
                         metric='haversine',
                         n_jobs=clustering['n_jobs']).fit(rads)
        aux = {tuple(loc): cluster.labels_[i]
               for i, loc in enumerate(loc_array)}
        cs = list(set(cluster.labels_))
        print(len(cs), max(cs), min(cs), len(pd.unique(fil['user'])))
        fil.loc[:, 'locid'] = fil.apply(lambda x: aux[(x['latitude'],
                                                       x['longitude'])],
                                      axis=1)
        if par is None:
            par = int(clustering['eps'] * KM_PER_RAD * 10)



    fil = fil[['user', 'locid']]
    fil.rename(columns={'user': 'uid'}, inplace=True)
    fil.to_csv('dataset/%s_%s_%d_%d.checkin' % (name,
                                                region,
                                                par,
                                                active_threshold))
    edge.to_csv('dataset/%s_%s_%d_%d.friends' % (name,
                                                 region,
                                                 par,
                                                 active_threshold),
                index=False)



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


def place_checkin_distribution(name):
    checkin = pd.read_csv('dataset/%s_20.checkin' % name, index_col=0)
    nc = checkin.groupby(['locid']).size().reset_index(name='counts')
    total = max(nc.counts)
    nc.loc[:, 'ncount'] = nc.apply(lambda x: np.log10(x['counts']/total),
                                   axis=1)
    nc.to_csv('%s_pcd.csv' % name)


def place_user_distribution(name):
    checkin = pd.read_csv('dataset/%s_20.checkin' % name, index_col=0)
    nc = checkin.drop_duplicates(['uid','locid'])
    nc = nc.groupby(['locid']).size().reset_index(name='counts')
    nc.to_csv('%s_pud.csv' % name)


def common_place_distribution(name):
    checkin = pd.read_csv('dataset/%s_20.checkin' % name, index_col=0)
    friends = pd.read_csv('dataset/%s_20.friends' % name)
    uid = pd.unique(checkin['uid'])
    locid = pd.unique(checkin['locid'])
    muid = max(uid) + 1
    checkin.loc[:, 'loc'] = checkin.apply(lambda x: x['locid'] + muid,
                                          axis=1)
    g = nx.from_pandas_dataframe(checkin, 'uid', 'loc')
    fr = nx.from_pandas_dataframe(friends, 'u1', 'u2')

    ran_commons = []
    ct = 0
    nu = len(uid)
    # sampling
    samples = 1000000
    n1 = np.random.choice(uid, samples*4)
    n2 = np.random.choice(uid, samples*4)
    pairs = set([(n1[i], n2[i]) for i in range(samples*4)
                 if n1[i] < n2[i]])
    pair = list(pairs)[:samples]
    print(len(pairs), len(pair))

    for e in pair:
        u, v = e
        common = len(list(nx.common_neighbors(g, u, v)))
        ran_commons.append(common)
    rcom_counter = Counter(ran_commons)


    fri_commons = []
    for e in fr.edges():
        u, v = e
        common = len(list(nx.common_neighbors(g, u, v)))
        fri_commons.append(common)

    fcom_counter = Counter(fri_commons)

    print([rcom_counter[i]/samples for i in range(13)])
    print([fcom_counter[i]/fr.number_of_edges() for i in range(13)])


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


# gen_w2f_dataset(SNAP_DATASET_NAMES[1], 20, 0.01)
# remap_locid("Gowalla_20")
# dataset_summary(SNAP_DATASET_NAMES[0] + '_10000M')

# region_dataset(SNAP_DATASET_NAMES[0], 20, 'na')
"""
for i in [1, 2, 8]:
    gen_region_w2f_dataset(SNAP_DATASET_NAMES[0], 20, 'na', 0.01, i*10,
                            {'model': 'DBSCAN',
                            'eps': i / KM_PER_RAD,
                            'min_samples': 1,
                            'n_jobs': -1},
                           )
"""
for i in [0.5, 1, 2, 4, 8, 16]:
    gen_w2f__cluster_dataset(SNAP_DATASET_NAMES[1], 20, 0.01, i*10,
                             {'model': 'DBSCAN',
                              'eps': i / KM_PER_RAD,
                              'min_samples': 1,
                              'n_jobs': -1},
                             )


# dataset_summary('Gowalla_na_30')
# common_place_distribution('Brightkite_na_80')
# place_checkin_distribution('Brightkite_na_80')
# place_user_distribution('la')
