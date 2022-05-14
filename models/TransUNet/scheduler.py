import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from utils import CustomCosineAnnealingWarmUpRestarts

def get_scheduler(args, optimizer):
    if args.name == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=args.params.step_size, gamma=args.params.gamma
        )
    elif args.name == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.params.milestonse])
    elif args.name == "CyclicLR":
        scheduler = lr_scheduler.CyclicLR(optimizer)
    elif args.name == "CosineAnnealLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.params.T_max, eta_min=args.params.eta_min
        )
    elif args.name == "CosineAnnealingWarmUpRestarts":
        scheduler = optim.CosineAnnealingWarmUpRestarts(
            optimizer, T_0=args.params.T_0, T_mult=args.params.T_mult, eta_min=args.params.eta_min
        )
    elif args.name == "CustomCosineAnnelingWarmupStarts":
        scheduler = CustomCosineAnnealingWarmUpRestarts(
            optimizer, T_0=args.params.T_0, T_mult=args.params.T_mult, eta_max=args.params.eta_max, T_up=args.params.T_up, gamma=args.params.gamma, 
        )
    else:
        raise ValueError("Not a valid scheduler")
    return scheduler
