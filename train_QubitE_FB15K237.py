import config
from  models import *
import json
import os 
con = config.Config()
con.set_in_path("./benchmarks/FB15K237/")
con.set_work_threads(8)
con.set_train_times(5000)
con.set_nbatches(10)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(100)
con.set_lmbda(0.3)
con.set_lmbda_two(0.3)
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000)
con.set_valid_steps(1000)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./QubitE_FB15k237_ckpt")
con.set_result_dir("./QubitE_FB15k237_result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(QubitE)
con.train()
# no type constraint results:
# metric:                  MRR             MR              hit@10          hit@3           hit@1
# l(raw):                  0.046248        1147.716064     0.097723        0.037721        0.018519
# r(raw):                  0.159142        558.783813      0.269960        0.162709        0.101583
# averaged(raw):           0.102695        853.249939      0.183841        0.100215        0.060051
#
# l(filter):               0.106723        839.072510      0.205463        0.111551        0.056777
# r(filter):               0.242962        527.508179      0.388449        0.264536        0.168523
# averaged(filter):        0.174843        683.290344      0.296956        0.188044        0.112650
# type constraint results:
# metric:                  MRR             MR              hit@10          hit@3           hit@1
# l(raw):                  0.064408        526.594055      0.137301        0.054676        0.025897
# r(raw):                  0.197304        140.227448      0.342617        0.200772        0.125476
# averaged(raw):           0.130856        333.410767      0.239959        0.127724        0.075687
#
# l(filter):               0.181634        217.946152      0.319212        0.188801        0.114287
# r(filter):               0.304943        108.951775      0.495505        0.332161        0.211961
# averaged(filter):        0.243288        163.448959      0.407359        0.260481        0.163124
# triple classification accuracy is 0.761918