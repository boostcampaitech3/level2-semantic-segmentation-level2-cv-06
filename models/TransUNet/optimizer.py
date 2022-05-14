import torch.optim as optim
from adamp import AdamP, SGDP

def get_optimizer(args, model):
    param_groups = model.parameters()
    if args.name == "SGD":
        optimizer = optim.SGD(
            param_groups,
            lr=args.params.lr,
            weight_decay=args.params.weight_decay,
            momentum=args.params.momentum,
            nesterov=False,
        )
    elif args.name == "Adam":
        optimizer = optim.Adam(
            param_groups,
            lr=args.params.lr,
            weight_decay=args.params.weight_decay,
        )
    elif args.name == 'AdamP':
        optimizer = AdamP(
            param_groups, 
            lr=args.params.lr,
            weight_decay=args.params.weight_decay,
        )
    elif args.name == 'SGDP':
        optimizer = SGDP(
            param_groups, 
            lr=args.params.lr,
        )
    elif args.name == 'AdamW':
        optimizer = optim.AdamW(
            param_groups, 
            lr=args.params.lr,
            weight_decay=args.params.weight_decay,
        )
    else:
        raise ValueError("Not a valid optimizer")
    
    return optimizer