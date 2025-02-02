from dataclasses import dataclass
from typing import Tuple, Dict, Mapping
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.policy import FiniteDeterministicPolicy
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical
from scipy.stats import poisson
from rl.dynamic_programming import policy_iteration_result, value_iteration_result

@dataclass(frozen=True)
class InventoryState:
    store1_on_hand: int
    store1_on_order: int
    store2_on_hand: int
    store2_on_order: int

    def inventory_position(self) -> Tuple[int, int]:
        return (
            self.store1_on_hand + self.store1_on_order,
            self.store2_on_hand + self.store2_on_order
        )


InvOrderMapping = Mapping[
    InventoryState,
    Mapping[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]]
]

class TwoStoreInventoryMDP(FiniteMarkovDecisionProcess[InventoryState, Tuple[int, int, int]]):
    def __init__(
        self,
        capacity1: int,
        capacity2: int,
        poisson_lambda1: float,
        poisson_lambda2: float,
        holding_cost1: float,
        holding_cost2: float,
        stockout_cost1: float,
        stockout_cost2: float,
        supplier_transportation_cost: float,
        two_store_transportation_cost: float

    ):
        self.capacity1: int = capacity1
        self.capacity2: int = capacity2
        self.poisson_lambda1: float = poisson_lambda1
        self.poisson_lambda2: float = poisson_lambda2
        self.holding_cost1: float = holding_cost1
        self.holding_cost2: float = holding_cost2
        self.stockout_cost1: float = stockout_cost1
        self.stockout_cost2: float = stockout_cost2
        self.supplier_transportation_cost  = supplier_transportation_cost
        self.two_store_transportation_cost = two_store_transportation_cost

        self.poisson_distr1 = poisson(poisson_lambda1)
        self.poisson_distr2 = poisson(poisson_lambda2)

        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> InvOrderMapping:
        d: Dict[InventoryState, Dict[Tuple[int, int, int], Categorical[Tuple[InventoryState,
                                                            float]]]] = {}

        for alpha1 in range(self.capacity1 + 1):
            for beta1 in range(self.capacity1 + 1 - alpha1):
                for alpha2 in range(self.capacity2 + 1):
                    for beta2 in range(self.capacity2 + 1 - alpha2):
                        state: InventoryState = InventoryState(alpha1, beta1, alpha2, beta2)
                        ip1, ip2 = state.inventory_position()
                        base_reward: float = - self.holding_cost1 * alpha1 - self.holding_cost2 * alpha2

                        d1: Dict[Tuple[int, int, int], Categorical[Tuple[InventoryState, float]]] = {}

                        for order1 in range(self.capacity1 - ip1 + 1):
                            for order2 in range(self.capacity2 - ip2 + 1):
                                for transfer in range(-alpha1, alpha2 + 1):
                                    new_s1_on_hand = max(0, alpha1 - transfer)
                                    new_s2_on_hand = max(0, alpha2 + transfer)
                                    transfer_cost = self.two_store_transportation_cost if transfer != 0 else 0

                                    sr_probs_dict = {}
                                    for demand1 in range(ip1 + 1):
                                        for demand2 in range(ip2 + 1):
                                            prob = self.poisson_distr1.pmf(demand1) * self.poisson_distr2.pmf(demand2)
                                            remaining_s1 = max(0, new_s1_on_hand - demand1)
                                            remaining_s2 = max(0, new_s2_on_hand - demand2)
                                            reward = base_reward - transfer_cost
                                            reward -= self.stockout_cost1 * max(0, demand1 - new_s1_on_hand)
                                            reward -= self.stockout_cost2 * max(0, demand2 - new_s2_on_hand)
                                            
                                            next_state = InventoryState(
                                                remaining_s1, order1,
                                                remaining_s2, order2
                                            )
                                            sr_probs_dict[(next_state, reward)] = prob
                                
                                d1[order1,order2, transfer] = Categorical(sr_probs_dict)

                        d[state] = d1
        return d
    


if __name__ == '__main__':
    from pprint import pprint

    user_capacity1 = 2
    user_poisson_lambda1 = 1.5
    user_holding_cost1 = 10.5
    user_stockout_cost1 = 14.0
    user_capacity2 = 2
    user_poisson_lambda2 = 1.2
    user_holding_cost2 = 0.7
    user_stockout_cost2 = 12.0
    user_supplier_transportation_cost = 2.5
    user_two_store_transportation_cost = 3.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        TwoStoreInventoryMDP(
            capacity1 = user_capacity1,
            capacity2 = user_capacity2,
            poisson_lambda1 = user_poisson_lambda1,
            poisson_lambda2 = user_poisson_lambda2,
            holding_cost1 = user_holding_cost1,
            holding_cost2 = user_holding_cost2,
            stockout_cost1 = user_stockout_cost1,
            stockout_cost2 = user_stockout_cost2,
            supplier_transportation_cost = user_supplier_transportation_cost, 
            two_store_transportation_cost = user_two_store_transportation_cost
        )

    print("MDP Transition Map")
    print("------------------")
    print(si_mdp)


    print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_pi, opt_policy_pi = policy_iteration_result(
        si_mdp,
        gamma=user_gamma
    )
    pprint(opt_vf_pi)
    print(opt_policy_pi)
    print()

    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()
