# rAnYKM test attack module

import sys

from process import folder_setup, data_process
from emb import ul_graph_build, para_ul_random_walk, emb_train, para_ul_random_batch
from predict import feature_construct, unsuper_friends_predict
from ran_figure import load_test_scores


def single_run(city, cicnt=20, wl=100, wt=20, n_feature=128, new_run=False):
    # city = "Brightkite"# ny la london Gowalla Brightkite
    # cicnt = 20

    folder_setup(city)
    checkin, friends = data_process(city, cicnt)

    ul_graph, lu_graph = ul_graph_build(checkin, 'locid')

    model_name = str(cicnt) + '_locid'
    print(model_name)

    walk_len, walk_times = 120, 30 # maximal 100 walk_len, 20 walk_times

    print('walking')
    if new_run:
        para_ul_random_batch(city, model_name, checkin.uid.unique(),
                             ul_graph, lu_graph, walk_len, walk_times)
    print('walk done')

    print('emb training')
    emb_train(city, model_name, wl, wt, n_feature)
    print('emb training done')

    feature_construct(city, model_name, friends, wl, wt, n_feature)
    unsuper_friends_predict(city, model_name, wl, wt, n_feature)
    # load_test_scores(city, model_name)

def multi_eps(city, region, cicnt=20):
    for eps in [50, 80]:
        single_run(city + '_' + region + '_' + str(eps), cicnt)


def walk_parameter_experiment(city, cicnt, walk_lens, walk_times, ns,
                              flag=False):
    for wl in walk_lens:
        for wt in walk_times:
            for n in ns:
                single_run(city, cicnt, wl, wt, n, flag)
                flag = False



if __name__ == '__main__':
    # multi_eps('Brightkite', 'na', 20)
    wls = [10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80 ,90, 100, 110, 120]
    wts = [20]
    np = [128]
    walk_parameter_experiment('Gowalla', 20, wls, wts, np, False)

