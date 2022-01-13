import config
from  models import *
import json
import os 
con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")
con.set_work_threads(8)
con.set_train_times(40000)
con.set_nbatches(10)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(100)
con.set_lmbda(0.1)
con.set_lmbda_two(0.01)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(10000)
con.set_valid_steps(10000)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./QubitE2_WN18RR_ckpt")
con.set_result_dir("./QubitE2_WN18RR_result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(QubitE2)
con.train()

