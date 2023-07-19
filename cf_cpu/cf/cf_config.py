from cpp_base import CPPBase
import cf_c


class CFConfig(CPPBase):
    def __init__(self, emb_dim=64, num_negs=4, max_his=8, num_users=128, num_items=128, train_size=128,
                 neg_sampler=0, tile_size=1024, num_subepoches=8, refresh_interval=2048, l2=1.e-3, clip_val=0.1,
                 milestones=[], l_r=1.e-3):
        super().__init__()
        self.c_class = cf_c.modules.CFConfig
        self.emb_dim = emb_dim
        self.num_negs = num_negs
        self.num_users = num_users
        self.num_items = num_items
        self.train_size = train_size
        self.neg_sampler = neg_sampler
        self.tile_size = tile_size
        self.refresh_interval = refresh_interval
        self.num_subepoches = num_subepoches
        self.l2 = l2
        self.clip_val = clip_val
        self.milestones = milestones
        self.l_r = l_r

        # dataset
        self.en_his = True
        self.max_his = max_his

    def init_c_instance(self):
        print(type(self.emb_dim), type(self.num_negs), type(self.num_users), type(self.num_items),
              type(self.train_size),
              type(self.neg_sampler), type(self.tile_size), type(self.refresh_interval), type(self.num_subepoches),
              type(self.l2), type(self.clip_val), type(self.milestones), type(self.l_r))
        print(type(self.milestones[0]), len(self.milestones))
        self.c_instance = self.c_class(emb_dim=self.emb_dim, num_negs=self.num_negs, num_users=self.num_users,
                                       num_items=self.num_items, train_size=self.train_size,
                                       neg_sampler=self.neg_sampler, tile_size=self.tile_size,
                                       refresh_interval=self.refresh_interval, num_subepoches=self.num_subepoches,
                                       l2=self.l2, clip_val=self.clip_val,
                                       milestones=self.milestones, l_r=self.l_r)
