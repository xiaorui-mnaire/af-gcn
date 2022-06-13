import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import warnings
warnings.filterwarnings("ignore")
# ==============================
utils.set_seed(world.seed)
print(">>SEED1:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
    # time.strftime()函数接收以时间元组，并返回以可读字符串表示的当地时间，格式由参数format决定。
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    precision_max = 0
    recall_max = 0
    ndcg_max = 0
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            precision_max = max(precision_max, results['precision'])
            recall_max = max(recall_max, results['recall'])
            ndcg_max = max(ndcg_max, results['ndcg'])
            print('precision_max:', precision_max, 'recall_max:', recall_max, 'ndcg_max:', ndcg_max)
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
        end = time.time()
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}'+str(end-start))
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
