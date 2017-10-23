import sys

import pandas as pd
from process import folder_setup, data_process
from defense import para_hiding, para_replace
from emb import ul_graph_build, para_ul_random_walk, emb_train
from predict import feature_construct, unsuper_friends_predict

# city = sys.argv[1]
# cicnt = int(sys.argv[2])

def multi_run(city, cicnt, ratios):
    for ratio in ratios:
        single_run(city, cicnt, ratio)

def single_run(city, cicnt, ratio):
    # ratio = int(sys.argv[3])# 10 20 30 40
    ratio = ratio*1.0/100

    folder_setup(city)
    checkin, friends = data_process(city, cicnt)

    defense_name = str(cicnt) + '_hiding_' + str(int(ratio*100))
    print(defense_name)

    checkin = para_hiding(city, defense_name, checkin, ratio)

    ul_graph, lu_graph = ul_graph_build(checkin, 'locid')

    model_name = str(cicnt) + '_locid_hiding_' + str(int(ratio*100))
    print(model_name)

    walk_len, walk_times = 100, 20 # maximal 100 walk_len, 20 walk_times

    print('walking')
    para_ul_random_walk(city, model_name, checkin.uid.unique(), ul_graph, lu_graph,
                        walk_len, walk_times)
    print('walk done')

    print('emb training')
    emb_train(city, model_name)
    print('emb training done')

    feature_construct(city, model_name, friends)
    unsuper_friends_predict(city, model_name)


def multi_replace(city, cicnt, ratios, steps, fail_to_continue):
    flag = fail_to_continue
    for step in steps:
        for rate in ratios:
            single_replace(city, cicnt, rate, step, flag)
            flag = False


def single_replace(city, cicnt, ratio, step, fail_to_continue=False):
    ratio = ratio * 1.0 / 100

    folder_setup(city)
    checkin, friends = data_process(city, cicnt)

    defense_name = str(cicnt) + '_replace_' + str(int(ratio * 100)) + '_' + str(
        int(step))

    model_name = str(cicnt) + '_locid_replace_' + str(
        int(ratio * 100)) + '_' + str(int(step))

    if not fail_to_continue:
        checkin = para_replace(city, defense_name, checkin, ratio, step)
    else:
        checkin = pd.read_csv('dataset/'+ city + '/defense/' + city + \
                                              '_20_replace_'+
                              str(int(ratio * 100)) + '_' + str(int(step)) +
                              '.checkin')

    ul_graph, lu_graph = ul_graph_build(checkin, 'locid')



    walk_len, walk_times = 100, 20  # maximal 100 walk_len, 20 walk_times

    print('walking')
    if not fail_to_continue:
        para_ul_random_walk(city, model_name, checkin.uid.unique(), ul_graph,
                            lu_graph,
                            walk_len, walk_times)
    print('walk done')

    print('emb training')
    emb_train(city, model_name)
    print('emb training done')

    feature_construct(city, model_name, friends)
    unsuper_friends_predict(city, model_name)


# multi_run('Brightkite', 20, [10, 30, 50, 70, 90])
multi_replace('Brightkite', 20, [70, 90], [15], True)