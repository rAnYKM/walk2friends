import numpy as np
import pandas as pd
import networkx as nx

from process import folder_setup, data_process
from emb import ul_graph_build, para_ul_random_walk, para_ul_random_batch


def workload_gen(num_user, num_location, loc_per_user=20, density=1e-2):
    locs = list(range(num_location))
    number_of_edges = int(density * (num_user - 1) * num_user / 2)
    social = nx.gnm_random_graph(num_user, number_of_edges)
    uid = list(social.nodes())
    # print(max(uid), min(uid))
    checkin = []
    edges = [{'u1': edge[0], 'u2': edge[1]}
             for edge in social.edges()]

    for u in uid:
        checkin += [{'uid': u, 'locid': loc}
                    for loc in np.random.choice(locs,
                                                loc_per_user,
                                                replace=False)]
    table_checkin = pd.DataFrame(checkin)
    table_edges = pd.DataFrame(edges)

    workname = 'workload_' + str(num_user) + '_' + str(num_location) + '_20'
    table_checkin.to_csv('dataset/' + workname + '.checkin')
    table_edges.to_csv('dataset/' + workname + '.friends')
    print('done')


def single_random_walk(num_user, num_location):
    city = 'workload_' + str(num_user) + '_' + str(num_location)
    folder_setup(city)
    cicnt = 20
    checkin, friends = data_process(city, cicnt)

    ul_graph, lu_graph = ul_graph_build(checkin, 'locid')

    model_name = str(cicnt) + '_locid'
    print(model_name)

    walk_len, walk_times = 50, 10 # maximal 100 walk_len, 20 walk_times

    print('walking')
    para_ul_random_walk(city, model_name, checkin.uid.unique(),
                         ul_graph, lu_graph, walk_len, walk_times)
    print('walk done')


def batch_random_walk(num_user, num_location):
    city = 'workload_' + str(num_user) + '_' + str(num_location)
    folder_setup(city)
    cicnt = 20
    checkin, friends = data_process(city, cicnt)

    ul_graph, lu_graph = ul_graph_build(checkin, 'locid')

    model_name = str(cicnt) + '_locid'
    print(model_name)

    walk_len, walk_times = 50, 10  # maximal 100 walk_len, 20 walk_times

    print('walking')
    para_ul_random_batch(city, model_name, checkin.uid.unique(),
                         ul_graph, lu_graph, walk_len, walk_times)
    print('walk done')

if __name__ == '__main__':
    workload_gen(200, 200, 20, 1e-2)
    batch_random_walk(200, 200)
