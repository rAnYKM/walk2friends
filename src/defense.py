import multiprocessing as mp

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from emb import ul_graph_build


def top_hiding_core(city, defense_name, u_checkin, ratio, N):
    """
    Top N Locations Hiding
    :param city:
    :param defense_name:
    :param u_checkin:
    :param ratio:
    :param N:
    :return:
    """
    # determin N
    loc_num = len(u_checkin.locid.unique())
    N = min(N, loc_num)
    total = u_checkin.shape[0]
    hides = int(np.floor(ratio * total))
    threshold = np.floor((total - hides) / loc_num)
    # print(threshold)

    # count the locations

    loc = u_checkin.groupby(['locid']).size().reset_index(name='counts')
    top_locations = loc.sort_values(['counts'], ascending=False).head(N)
    # print(top_locations)
    t_top = sum(top_locations.counts)
    # print(t_top)
    if t_top - threshold * N <= hides:
        top_hides = int(t_top - threshold * N)
        bottom_hides = hides - top_hides
    else:
        top_hides = int(hides)
        bottom_hides = 0
    # print(top_hides, bottom_hides)

    if N == loc_num:
        hiding_mid = np.random.choice(u_checkin.mid, hides, replace=False)
        u_checkin = u_checkin.loc[~u_checkin.mid.isin(hiding_mid)]
    else:
        tops = u_checkin['locid'].isin(top_locations['locid'])
        checkin1 = u_checkin[tops]
        checkin2 = u_checkin[~tops]

        hiding_mid1 = np.random.choice(checkin1.mid, top_hides,
                                       replace=False)
        hiding_mid2 = np.random.choice(checkin2.mid, bottom_hides,
                                       replace=False)
        u_checkin = pd.concat([checkin1[~checkin1.mid.isin(hiding_mid1)],
                               checkin2[~checkin2.mid.isin(hiding_mid2)]])
    u_checkin.to_csv('dataset/' + city + '/defense/' + city + '_' + defense_name + '.checkin',
        header=None, index=False, mode='a')


def para_top_hiding(city, defense_name, checkin, ratio, N):
    ''' para hiding
    Args:
        city: city
        defense_name: 20_hiding_20
        checkin: checkin data
        ratio: obfuscate proportion, 0.1, 0.2, 0.3 ...
    Returns:
        checkin: checkin data
    '''
    # intialize
    pd.DataFrame([checkin.columns]).to_csv('dataset/' + city + '/defense/' + \
                                           city + '_' + defense_name + '.checkin',
                                           index=False, header=None)
    core_num = mp.cpu_count()
    print(core_num)

    Parallel(n_jobs=core_num)(delayed(top_hiding_core)(
        city, defense_name, checkin.loc[checkin.uid == u], ratio, N) for u in
                              checkin.uid.unique())

    checkin = pd.read_csv(
        'dataset/' + city + '/defense/' + city + '_' + defense_name + '.checkin',
        error_bad_lines=False)

    checkin = checkin.dropna().reset_index(drop=True)
    checkin.uid = checkin.uid.apply(float).apply(int)
    checkin.locid = checkin.locid.apply(float).apply(int)

    checkin = checkin.reset_index(drop=True)

    checkin.to_csv('dataset/' + city + '/defense/' + city + '_' + defense_name + '.checkin',
        index=False)

    return checkin


def hiding_core(city, defense_name, u_checkin, ratio):
    ''' hiding core
    Args:
        city: city
        defense_name: 20_hiding_20
        u_checkin: checkin of a single user
        ratio: hiding proportion, 0.1, 0.2, 0.3 ...
    Returns:
    '''
    
    hiding_mid = np.random.choice(u_checkin.mid, int(u_checkin.shape[0]*ratio),\
                                replace=False)
    u_checkin = u_checkin.loc[~u_checkin.mid.isin(hiding_mid)]
    u_checkin.to_csv('dataset/'+city+'/defense/'+city+'_'+defense_name+'.checkin',\
                     header=None, index=False, mode='a')

def para_hiding(city, defense_name, checkin, ratio):
    ''' para hiding
    Args:
        city: city
        defense_name: 20_hiding_20
        checkin: checkin data
        ratio: obfuscate proportion, 0.1, 0.2, 0.3 ...
    Returns:
        checkin: checkin data
    '''
    # intialize
    pd.DataFrame([checkin.columns]).to_csv('dataset/'+city+'/defense/'+\
                city+'_'+defense_name+'.checkin', index=False, header=None)
    core_num = mp.cpu_count()
    print(core_num)

    Parallel(n_jobs = core_num)(delayed(hiding_core)(\
        city, defense_name, checkin.loc[checkin.uid==u], ratio) for u in checkin.uid.unique())

    checkin = pd.read_csv('dataset/'+city+'/defense/'+city+'_'+defense_name+'.checkin',\
                          error_bad_lines=False)

    checkin = checkin.dropna().reset_index(drop=True)
    checkin.uid = checkin.uid.apply(float).apply(int)
    checkin.locid = checkin.locid.apply(float).apply(int)

    checkin = checkin.reset_index(drop=True)
    
    checkin.to_csv('dataset/'+city+'/defense/'+city+'_'+defense_name+'.checkin',\
                   index=False)

    return checkin

def replace_core(city, defense_name, u_checkin, ul_graph, lu_graph, ratio, step):
    ''' replace core
    Args:
        city: city
        defense_name: 20_replace_20
        u_checkin: checkin of a single user
        ul_graph, lu_graph: graph data (pandas df)        
        ratio: hiding proportion, 0.1, 0.2, 0.3 ...
        step: step length, (odd)        
    Returns:
    '''
    
    replace_mid = np.random.choice(u_checkin.mid, int(u_checkin.shape[0]*ratio),\
                                   replace=False)
    keep_u_checkin = u_checkin.loc[~u_checkin.mid.isin(replace_mid)].reset_index(drop=True)
    replace_u_checkin = u_checkin.loc[u_checkin.mid.isin(replace_mid)].reset_index(drop=True)
    
    for i in range(replace_u_checkin.shape[0]):
        u = replace_u_checkin.loc[i, 'uid']
        curr_u = u
        flag = 0 # flag 0, user, flag 1, location
        for j in range(step):
            if flag == 0:# at social network
                temp_val = ul_graph.loc[ul_graph.uid==curr_u]
                flag = 1
            elif flag == 1: # at location
                temp_val = lu_graph.loc[lu_graph.locid==curr_u]
                flag = 0
            # sample with weights
            next_u = np.random.choice(temp_val['node2'].values, 1, p=temp_val['weight'])[0]
            curr_u = next_u
        replace_u_checkin.loc[i, 'locid'] = curr_u

    u_checkin = pd.concat([keep_u_checkin, replace_u_checkin], ignore_index=True)
    u_checkin.to_csv('dataset/'+city+'/defense/'+city+'_'+defense_name+'.checkin',\
                     header=None, index=False, mode='a')

def para_replace(city, defense_name, checkin, ratio, step):
    ''' para replace
    Args:
        city: city
        defense_name: 20_replace_20
        checkin: checkin data
        ratio: obfuscate proportion, 0.1, 0.2, 0.3 ...
        step: step length, (odd)
    Returns:
        checkin: checkin data
    '''
    # intialize
    pd.DataFrame([checkin.columns]).to_csv('dataset/'+city+'/defense/'+\
                 city+'_'+defense_name+'.checkin', index=False, header=None)
    
    ul_graph, lu_graph = ul_graph_build(checkin, 'locid')

    core_num = mp.cpu_count()
    print(core_num)

    Parallel(n_jobs = core_num)(delayed(replace_core)(\
        city, defense_name, checkin.loc[checkin.uid==u], ul_graph, lu_graph, ratio, step) for u in checkin.uid.unique())

    checkin = pd.read_csv('dataset/'+city+'/defense/'+city+'_'+defense_name+'.checkin',\
                          error_bad_lines=False)
    checkin = checkin.dropna().reset_index(drop=True)
    checkin.uid = checkin.uid.apply(float).apply(int)
    checkin.locid = checkin.locid.apply(float).apply(int)

    checkin = checkin.reset_index(drop=True)
    
    checkin.to_csv('dataset/'+city+'/defense/'+city+'_'+defense_name+'.checkin',\
                   index=False)

    return checkin
