from config import setup_cfg, validate_cfg, load_cfg, save_cfg, print_cfg
from solver.solver import Solver


def main(args):
    solver = Solver(args)
    if args.mode == 'train':
        solver.train()
    else:
        assert False, f"Unimplemented mode: {args.mode}"


if __name__ == '__main__':
    cfg = load_cfg()
    setup_cfg(cfg)
    validate_cfg(cfg)
    save_cfg(cfg)
    print_cfg(cfg)
    main(cfg)
