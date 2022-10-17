"""Mutation-based fuzzing driver."""

import ctypes

from omegaconf import DictConfig
import hydra

from nnsmith.gir import GraphIR
from nnsmith.graph_gen import concretize_graph, random_model_gen
from nnsmith.logging import FUZZ_LOG
from nnsmith.materialize import Schedule, TestCase
from nnsmith.cli.fuzz import FuzzingLoop
from nnsmith.memcov import Memcov
from nnsmith.mutator import MutantPool, OpInsertForward, LeafOpRemove, ReviseDataflow


class MutFuzzLoop(FuzzingLoop):
    def __init__(self, cfg: DictConfig):
        assert not cfg["fuzz"][
            "crash_safe"
        ], "Crash-safe mode is not compatible with memcov."
        super().__init__(cfg)
        self.mut_pool = MutantPool(
            mutants=[],
            mutators=[
                OpInsertForward(op_types=self.opset, max_node=cfg["mgen"]["max_nodes"]),
                LeafOpRemove(),
                ReviseDataflow(mutate_try=3),
            ],
        )
        assert len(self.mut_pool.mutators) > 0
        lib_path = cfg["fuzz"]["covlib"]
        self.num_init_gen = cfg["fuzz"]["num_init_gen"]
        assert isinstance(self.num_init_gen, int)
        if self.num_init_gen < 8:
            raise ValueError("`num_init_gen` must be at least 8")
        self.cov = Memcov(ctypes.CDLL(lib_path))
        self.cov.reset_bitmap()  # reset the bitmap to exclude import-time coverage.

    def make_testcase(self, seed) -> TestCase:
        use_mut = len(self.mut_pool.mutants) >= self.num_init_gen
        if use_mut:
            FUZZ_LOG.debug(f"Generate testcase with mutation.")
            mutant = self.mut_pool.make_new_mutant()
            if mutant:  # TODO(@ganler): VISUALIZE MUTANT.
                schedule = mutant.to_schedule()
            else:
                FUZZ_LOG.warning("Failed to make new mutant. Fallback to random gen.")
                use_mut = False

        if not use_mut:
            FUZZ_LOG.debug(f"Generate testcase with random generation")
            mgen_cfg = self.cfg["mgen"]
            gen = random_model_gen(
                opset=self.opset,
                init_rank=mgen_cfg["init_rank"],
                seed=seed,
                max_nodes=mgen_cfg["max_nodes"],
                timeout_ms=mgen_cfg["timeout_ms"],
            )

            fixed_graph, concrete_abstensors = concretize_graph(
                gen.abstract_graph, gen.tensor_dataflow, gen.get_solutions()
            )

            schedule = Schedule.init(fixed_graph, concrete_abstensors)

        model = self.ModelType.from_schedule(schedule)

        self.last_schedule = schedule

        if not use_mut and self.cfg["debug"]["viz"]:
            model.attach_viz(fixed_graph)

        model.refine_weights()  # either random generated or gradient-based.
        oracle = model.make_oracle()

        return TestCase(model, oracle)

    def validate_and_report(self, testcase: TestCase) -> bool:
        prev_cov = self.cov.get_hitbits()
        is_valid = super().validate_and_report(testcase)
        if is_valid and prev_cov < self.cov.get_hitbits():  # new coverage.
            mutant = GraphIR.from_schedule(self.last_schedule)
            self.mut_pool.add_mutant(mutant)
            FUZZ_LOG.info(
                f"Cov++! Add a {len(mutant)}-node mutant. Pool size: {len(self.mut_pool.mutants)}."
            )
        return is_valid


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    MutFuzzLoop(cfg).run()


if __name__ == "__main__":
    main()
