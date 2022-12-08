import pickle

import hydra
from omegaconf import DictConfig

from nnsmith.autoinf import AutoInfOpBase


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    gir_path = cfg["debug"]["gir_path"]
    with open(gir_path, "rb") as f:
        gir = pickle.load(f)
    print(gir)
    print("\n === AutoInfOpBase instances: ===")
    for inst in gir.insts:
        if isinstance(inst.iexpr.op, AutoInfOpBase):
            print(
                inst.retvals(), "~", inst.iexpr.op.inst.invoke_str(inst.iexpr.op.attrs)
            )


if __name__ == "__main__":
    main()
