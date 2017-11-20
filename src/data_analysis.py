import os
import time
import logging
import numpy as np
import pandas as pd
import networkx as nx


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

# CONST
DATASET_DIR = 'dataset'


class LBSDataManager:
    """
    check_ins: uid, locid
    friends: u1, u2
    """
    def get_common_locs(self, u1, u2):
        return list(nx.common_neighbors(self.ul_graph, u1, u2))

    def loc_uni_visitor(self, loc):
        return self.ul_graph.neighbors(loc)

    def loc_checkins(self, loc):
        total = sum([d['weight']
                    for u, v, d in self.ul_graph.edges(loc, data=True)])
        return total

    def friend_common_locs(self):
        return {e: self.get_common_locs(e[0], e[1])
                for e in self.uu_graph.edges()}

    def common_loc_study(self):
        comm_loc = []
        no_commons = []
        for e, comm in self.friend_common_locs().items():
            if not comm:
                no_commons.append(e)
            comm_loc += comm
        vc = pd.DataFrame([{'locid': loc} for loc in comm_loc])
        vc = vc.groupby('locid').size().reset_index(name='count')
        print("friend pair sharing no location: %d" % len(no_commons))
        print("location shared by friends: %d" % len(set(comm_loc)))
        ls2p = [(loc, self.ul_graph.neighbors(loc)) for loc in self.locid
                if len(self.ul_graph.neighbors(loc)) >= 2]

        ls2np = [loc for loc, nei in ls2p
                 if nx.subgraph(self.uu_graph, nei).number_of_edges() == 0]
        print("location shared by at least two people: %d/%d" %
              (len(ls2np), len(ls2p)))
        lso2p = [(loc, self.ul_graph.neighbors(loc)) for loc in self.locid
                 if len(self.ul_graph.neighbors(loc)) == 2]

        lso2nf = [loc for loc, pair in lso2p
                  if not self.uu_graph.has_edge(pair[0], pair[1])]
        print(vc)
        # print("location shared by only a pair: %d" %
        #      vc[vc['count']==1].shape[0])
        print('location shared by only two people: %d/%d' %
              (len(lso2nf), len(lso2p)))


    def basic_summary(self):
        print('Users: %d, Locations: %d, Friend pairs: %d' %
              (len(self.uid), len(self.locid), self.uu_graph.number_of_edges()))

    def __init__(self, check_ins, friends, self_check=True):
        self.uid = check_ins['uid'].unique()
        self.locid = check_ins['locid'].unique()
        self.ul_graph = nx.from_pandas_dataframe(check_ins, 'uid', 'locid',
                                                 ['weight'])
        self.uu_graph = nx.from_pandas_dataframe(friends, 'u1', 'u2')
        if self_check:
            uid_cmp = set(friends['u1'].unique()) | set(friends['u2'].unique())
            assert uid_cmp.issubset(set(self.uid))
            logging.debug('self check pass %d/%d' % (len(uid_cmp),
                                                     len(set(self.uid))))

class W2FDataManager(LBSDataManager):
    def __init__(self, dataset, cicnt=20, overlap=True, self_check=True):
        check_ins = pd.read_csv(os.path.join(DATASET_DIR,
                                             '%s_%d.checkin' % (dataset, cicnt),
                                             ),
                                index_col=0)
        friends = pd.read_csv(os.path.join(DATASET_DIR,
                                           '%s_%d.friends' % (dataset, cicnt)))
        check_ins = check_ins.groupby('uid')['locid'].value_counts().\
            reset_index(name='weight')
        if overlap:
            muid = max(check_ins['uid'].unique()) + 1
            check_ins.locid += muid
        LBSDataManager.__init__(self, check_ins, friends, self_check)
        self.check_ins = check_ins
        self.friends = friends


def check_w2f_dataset(dataset, cicnt):
    t0 = time.time()
    wd = W2FDataManager(dataset, cicnt)
    wd.basic_summary()
    logging.debug('load LBS data in %f second(s)' % (time.time() - t0))
    # print(wd.friend_common_locs())
    wd.common_loc_study()

if __name__ == '__main__':
    check_w2f_dataset('london', 20)
