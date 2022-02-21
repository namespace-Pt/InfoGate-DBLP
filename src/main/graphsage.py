import torch.multiprocessing as mp
from utils.manager import Manager
from models.GraphSage import GraphSage
from torch.nn.parallel import DistributedDataParallel as DDP


def main(rank, manager):
    """ train/dev/test the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()
    model = GraphSage(manager).to(manager.device)

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

    elif manager.mode == "encode":
        model.encode(manager, loaders)

        
if __name__ == "__main__":
    config = {
        "validate_step": "0.5e",
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