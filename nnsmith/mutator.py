from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field
from copy import deepcopy
import random

import z3

from nnsmith.logging import CORE_LOG
from nnsmith.abstract.op import AbsTensor, AbsOpBase, concretize_op
from nnsmith.gir import GraphIR, InstIR


class Mutator(ABC):
    @abstractmethod
    def mutate(self, graph: GraphIR) -> Optional[GraphIR]:
        """Mutate a graph IR and return a new graph IR if successful, None otherwise.

        Args:
            graph (GraphIR): The graph IR to mutate.

        Returns:
            Optional[GraphIR]: The mutated graph IR if successful, None otherwise.

        Note:
            The mutated IR should be guaranteed to be valid.
        """
        pass

    def skip_if(self, graph: GraphIR) -> bool:
        """Skip mutation if the graph IR is not suitable.

        Args:
            graph (GraphIR): The graph IR to mutate.

        Returns:
            bool: True if the graph IR is not suitable, False otherwise.
        """
        return False


# Mutations::Graph level:
#  - Insert a new node;
#  - Remove leaf node;
#  - Re-link nodes; => change the dataflow;


class LeafOpRemove(Mutator):
    def __init__(self, min_size=1):
        self.min_size = min_size

    def mutate(self, graph: GraphIR) -> Optional[GraphIR]:
        leafs = graph.leafs()
        if not leafs:
            return None
        graph = deepcopy(graph)
        leaf = random.choice(leafs)
        del graph.values[leaf]
        del graph.defs[leaf]
        del graph.users[leaf]
        for var in graph.users:
            graph.users[var] = [x for x in graph.users[var] if x != leaf]
        return graph

    def skip_if(self, graph: GraphIR) -> bool:
        return len(graph) <= self.min_size


class OpInsertForward(Mutator):
    def __init__(self, op_types, max_node=32) -> None:
        super().__init__()
        self.op_types = op_types
        self.max_node = max_node

    def mutate(self, graph: GraphIR) -> Optional[GraphIR]:
        # (rank, dtype) => abs_tensor
        type2name: Dict[AbsTensor, str] = {}
        for name, value in graph.values.items():
            type2name.setdefault(value.downcast_rank(), []).append(name)

        node_t: Type[AbsOpBase] = random.choice(self.op_types)
        op_params = [z3.Int(f"v{i}") for i in range(node_t.get_num_var_param())]

        op: AbsOpBase = node_t(*op_params)

        # match dtypes
        inputs = None
        for dtypes in random.sample(op.in_dtypes, len(op.in_dtypes)):
            # try all ranks.
            if op.same_inp_dims:
                # find a common set of ranks.
                common_ranks = set(op.inp_ranks[0])
                for ranks in op.inp_ranks[1:]:
                    common_ranks.intersection_update(ranks)

                if not common_ranks:
                    continue

                def match_fixed_rank(fixed_rank, dtypes):
                    attempt_inputs = []
                    for dtype in dtypes:
                        atype = AbsTensor(shape=[None] * fixed_rank, dtype=dtype)
                        if atype not in type2name:  # no candidate.
                            return None
                        attempt_inputs.append(random.choice(type2name[atype]))
                    return attempt_inputs

                value_keywords = None
                random.shuffle(list(common_ranks))
                for fixed_rank in common_ranks:
                    value_keywords = match_fixed_rank(fixed_rank, dtypes)
                    if value_keywords:
                        break

            else:  # try random comb.

                def match_relaxed_rank(avai_ranks, dtypes):
                    attempt_inputs = []
                    for i, ranks in enumerate(avai_ranks):
                        for rank in random.sample(ranks, len(ranks)):
                            atype = AbsTensor(shape=[None] * rank, dtype=dtypes[i])
                            if atype in type2name:
                                attempt_inputs.append(random.choice(type2name[atype]))
                                break
                        if len(attempt_inputs) != i + 1:
                            return None
                    return attempt_inputs

                value_keywords = match_relaxed_rank(op.inp_ranks, dtypes)

            # success.
            if value_keywords:
                inputs = [graph.values[k] for k in value_keywords]
                break

        if not inputs:
            return None

        # try to solve the node.
        solver = z3.Solver()

        solver.add(*op.checked_requires(inputs))
        if solver.check() != z3.sat:
            return None

        m = solver.model()

        # rewrite
        concrete_op = concretize_op(op, m)
        outputs = concrete_op.checked_type_transfer(inputs)

        graph = deepcopy(graph)
        name_remap = graph.normalize()

        value_keywords = [name_remap[k] for k in value_keywords]

        new_id = len(graph.values)
        assert len(outputs) == 1, "FIXME: support multiple outputs"
        graph.values[f"v{new_id}"] = outputs[0]
        graph.defs[f"v{new_id}"] = InstIR(concrete_op, tuple(value_keywords))
        graph.users[f"v{new_id}"] = []
        for k in value_keywords:
            graph.users[k].append(f"v{new_id}")
        return graph

    def skip_if(self, graph: GraphIR) -> bool:
        return len(graph.values) > self.max_node


# TODO(@ganler): IMPLEMENT ME!
class OpInsertBackward(Mutator):
    def mutate(self, graph: GraphIR) -> Optional[GraphIR]:
        pass


# TODO(@ganler): IMPLEMENT ME!
class OpReplace(Mutator):
    def mutate(self, graph: GraphIR) -> Optional[GraphIR]:
        pass


class ReviseDataflow(Mutator):
    def __init__(self, mutate_try=2) -> None:
        super().__init__()
        self.mutate_try = mutate_try

    def mutate(self, graph: GraphIR) -> Optional[GraphIR]:
        # group values sharing the same AbsTensor type
        type2values = {}
        for name, value in graph.values.items():
            type2values.setdefault(value, []).append(name)

        # remove those that have only one value
        type2values = {k: v for k, v in type2values.items() if len(v) > 1}
        if not type2values:
            return None

        ret_graph: Optional[GraphIR] = None
        n_tried = 0
        while n_tried < self.mutate_try and ret_graph is None:
            n_tried += 1

            # pick a random type
            type = random.choice(list(type2values.keys()))

            # consumer sites are consumers to `type`
            # defined by caller and the argument index.
            consumer_sites = []

            for name, inst in graph.defs.items():
                for i, arg in enumerate(inst.args):
                    if graph.values[arg] == type:
                        consumer_sites.append((name, i))

            # shuffle sites
            random.shuffle(consumer_sites)

            for name, i in consumer_sites:
                candiates = (
                    set(type2values[type]) - set(graph.expand_users(name)) - {name}
                )
                # `candiates`: Names of dependency-free values under the same type (`name` itself
                #    is also excluded). "Dependency-free" means candidates cannot be `name`'s users.

                if not candiates:
                    continue

                # update graph
                if ret_graph is None:
                    ret_graph = deepcopy(graph)  # init ret_graph with a copy

                # pick a random candidate
                old_arg = graph.defs[name].args[i]
                new_arg = random.choice(list(candiates))

                old_args = ret_graph.defs[name].args
                new_args = (
                    *old_args[:i],
                    new_arg,
                    *old_args[(i + 1) :],
                )  # change arg
                ret_graph.defs[name].args = new_args
                ret_graph.users[old_arg].remove(name)  # refine dep
                ret_graph.users[new_arg].append(name)  # refine dep
                break

        return ret_graph  # Note that a value cannot be consumed by its dependency.


# Mutations::Attribute level:
#  - Relax & resolve constraints ~ build a database of applicable constraints;
#  - Change attributes;
# TODO(@ganler): IMPLEMENT ME!


@dataclass
class MutantPool:
    mutants: List[GraphIR] = field(default_factory=list)
    mutators: List[Mutator] = field(default_factory=list)
    step: int = 1

    def add_mutant(self, mutant: GraphIR) -> None:
        self.mutants.append(mutant)

    def make_new_mutant(self, max_mutate_factor=2) -> GraphIR:
        new_mutant = None
        n_mut_sucess = 0
        n_mut_try = 0
        n_mut_max = max_mutate_factor * self.step

        while n_mut_sucess < self.step and n_mut_try < n_mut_max:
            mutant = random.choice(self.mutants)
            mutator = random.choice([m for m in self.mutators if not m.skip_if(mutant)])
            CORE_LOG.debug(f"Applying {mutator} ::\n{mutant}")
            temp_new_mutant = mutator.mutate(mutant)
            if temp_new_mutant:
                new_mutant = temp_new_mutant
                CORE_LOG.debug(f" :: [old => new] ::\n{new_mutant}")
                new_mutant.normalize()
                n_mut_sucess += 1
            else:
                CORE_LOG.debug(f" ~~ [mut failed] ~~ ")

            n_mut_try += 1

        return new_mutant
