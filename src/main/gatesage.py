import torch.multiprocessing as mp
from utils.manager import Manager
from models.GateSage import GateSage
from torch.nn.parallel import DistributedDataParallel as DDP
from models.modules.weighter import *


def main(rank, manager):
    """ train/dev/test the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()

    if manager.weighter == "cnn":
        weighter = CnnWeighter(manager)
    elif manager.weighter == "tfm":
        weighter = TfmWeighter(manager)
    elif manager.weighter == "bert":
        weighter = AllBertWeighter(manager)
    elif manager.weighter == "first":
        weighter = FirstWeighter(manager)

    model = GateSage(manager, weighter).to(manager.device)

    if manager.mode == 'train':
        if manager.world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)
        manager.train(model, loaders)

    elif manager.mode == 'dev':
        # if isinstance(model, DDP):
        #     model.module.dev(manager, loaders, load=True, log=True)
        # else:
        model.dev(manager, loaders, load=True, log=True)

    elif manager.mode == "test":
        model.test(manager, loaders, load=True, log=True)

    elif manager.mode == "inspect":
        manager.load(model)
        model.inspect(manager, loaders)



if __name__ == "__main__":
    config = {
        "validate_step": "0.5e",
        "enable_gate": "weight"
    }
    manager = Manager(config)


    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)