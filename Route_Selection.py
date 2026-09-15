import numpy as np
import cyipopt
from dataclasses import dataclass
from itertools import combinations
import time
EPS = 1e-12
DTYPE = float

# ============================================================
# 1) 2D gridworld geometry
# ============================================================
GRID_N = 15
START = (0, 0)
GOAL = (14, 14)

# Risky center zone
C_ZONE = {(7, 7), (7, 8), (8, 7), (8, 8)}

# Physical shop waypoints on each route
SHOP_FAST = (4, 4)
SHOP_TOP = (0, 4)
SHOP_RIGHT = (4, 0)

# Probability of actually stopping at the shop after reaching it
Q_STOP_FAST = 0.6
Q_STOP_TOP = 0.6
Q_STOP_RIGHT = 0.6

# Horizon
H = 35

# Progress probabilities outside the risky C-zone
P_ADV_FAST = 0.9
P_ADV_SAFE = 0.9

# One-time probability of getting stuck forever upon first entering C-zone
Q_STUCK = 0.4

# ============================================================
# Bounded grammar represented as a shared syntax DAG
# ============================================================
#
#   p   ::= T | C | S
#   psi ::= F p | G !p
#   phi ::= psi | (psi_1 ^ ... ^ psi_k), 1 <= k <= MAX_CLAUSES
#
# T: reaching the target, C: visiting the center work zone,
# S: actually stopping at a coffee shop.
#
# The grammar determines the finite hypothesis class. The syntax DAG
# determines how that class is stored: identical atomic propositions and
# temporal clauses are interned once and shared by every candidate root.
# Conjunction is stored as a canonical n-ary node so that clause order and
# parenthesization do not create duplicate formulas.
ATOMIC_PROPOSITIONS = ("T", "C", "S")
MAX_CLAUSES = 3

# Optional syntactic restriction. Leave empty to search the entire bounded
# grammar. For example, use {"T"} if every task candidate must mention the
# target proposition.
REQUIRED_PROPOSITIONS = set()


@dataclass(frozen=True)
class SyntaxNode:
    """One immutable node in the shared LTLf syntax DAG."""

    operator: str
    children: tuple = ()
    proposition: str = None


class SyntaxDAG:
    """Hash-consed, topologically ordered syntax DAG."""

    def __init__(self):
        self.nodes = []
        self._node_index = {}

    def intern(self, operator, children=(), proposition=None):
        """Return the unique node ID for the requested syntax node."""
        children = tuple(children)

        if operator == "AP":
            if proposition not in ATOMIC_PROPOSITIONS:
                raise ValueError(f"Unknown atomic proposition: {proposition}")
            if children:
                raise ValueError("An AP node cannot have children.")

        elif operator in {"F", "G_NOT"}:
            if proposition is not None or len(children) != 1:
                raise ValueError(f"{operator} must have exactly one child.")

        elif operator == "AND":
            if proposition is not None:
                raise ValueError("An AND node cannot carry a proposition.")

            # AND is commutative and idempotent. Sorting and removing repeated
            # children gives one canonical root for each clause set.
            children = tuple(sorted(set(children)))
            if not children:
                raise ValueError("An AND node must have at least one child.")
            if len(children) == 1:
                return children[0]

        else:
            raise ValueError(f"Unknown syntax-DAG operator: {operator}")

        # Children must already exist, which guarantees topological order and
        # prevents cycles.
        if any(child < 0 or child >= len(self.nodes) for child in children):
            raise ValueError("Every child must be an existing DAG node.")

        key = (operator, children, proposition)
        if key not in self._node_index:
            node_id = len(self.nodes)
            self._node_index[key] = node_id
            self.nodes.append(
                SyntaxNode(
                    operator=operator,
                    children=children,
                    proposition=proposition,
                )
            )

        return self._node_index[key]

    def render(self, node_id):
        """Render a candidate root as a readable LTLf formula."""
        node = self.nodes[node_id]

        if node.operator == "AP":
            return node.proposition
        if node.operator == "F":
            return f"F {self.render(node.children[0])}"
        if node.operator == "G_NOT":
            return f"G !{self.render(node.children[0])}"
        if node.operator == "AND":
            clauses = [self.render(child) for child in node.children]
            return "(" + " ^ ".join(clauses) + ")"

        raise ValueError(f"Unknown syntax-DAG operator: {node.operator}")


SYNTAX_DAG = SyntaxDAG()

# Atomic proposition nodes are shared by all temporal-clause nodes.
AP_NODE_IDS = {
    ap: SYNTAX_DAG.intern("AP", proposition=ap)
    for ap in ATOMIC_PROPOSITIONS
}

# Temporal-clause nodes are shared by all candidate conjunction roots.
CLAUSE_NODE_IDS = {}
for ap in ATOMIC_PROPOSITIONS:
    CLAUSE_NODE_IDS[("F", ap)] = SYNTAX_DAG.intern(
        "F", children=(AP_NODE_IDS[ap],)
    )
for ap in ATOMIC_PROPOSITIONS:
    CLAUSE_NODE_IDS[("G_NOT", ap)] = SYNTAX_DAG.intern(
        "G_NOT", children=(AP_NODE_IDS[ap],)
    )


def generate_candidate_roots(max_clauses=MAX_CLAUSES):
    """Generate every valid candidate root in the bounded grammar."""
    clause_specs = list(CLAUSE_NODE_IDS)
    roots = []
    seen_roots = set()

    for number_of_clauses in range(1, max_clauses + 1):
        # combinations removes repetitions and conjunction-order duplicates.
        for specification in combinations(clause_specs, number_of_clauses):
            # F p ^ G !p is inconsistent on every finite trace.
            contradictory = any(
                ("F", ap) in specification
                and ("G_NOT", ap) in specification
                for ap in ATOMIC_PROPOSITIONS
            )
            if contradictory:
                continue

            used_propositions = {ap for _, ap in specification}
            if not REQUIRED_PROPOSITIONS.issubset(used_propositions):
                continue

            clause_ids = tuple(
                CLAUSE_NODE_IDS[clause]
                for clause in specification
            )
            root_id = (
                clause_ids[0]
                if len(clause_ids) == 1
                else SYNTAX_DAG.intern("AND", children=clause_ids)
            )

            if root_id not in seen_roots:
                seen_roots.add(root_id)
                roots.append(root_id)

    return tuple(roots)


CANDIDATE_ROOT_IDS = generate_candidate_roots()
FORMULA_ROOT_IDS = {
    SYNTAX_DAG.render(root_id): root_id
    for root_id in CANDIDATE_ROOT_IDS
}
FORMULAS = list(FORMULA_ROOT_IDS)

# ============================================================
# 2) Explicit 2D routes
# ============================================================
def build_routes():
    # Safe route along left border, then across top border
    route_top = []
    for y in range(0, 15):
        route_top.append((0, y))
    for x in range(1, 15):
        route_top.append((x, 14))

    # Safe route along bottom border, then up right border
    route_right = []
    for x in range(0, 15):
        route_right.append((x, 0))
    for y in range(1, 15):
        route_right.append((14, y))

    # Fast shortcut through the middle
    route_fast = [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),   # physical shop waypoint on fast route
        (5, 5),
        (6, 6),
        (7, 7),   # enters C-zone
        (8, 8),   # still in C-zone
        (9,9),
        (10, 10),
        (11,11),
        (12, 12),
        (13,13),
        (14, 14),
    ]
    return route_fast, route_top, route_right


ROUTE_FAST, ROUTE_TOP, ROUTE_RIGHT = build_routes()

IDX_SHOP_TOP = ROUTE_TOP.index(SHOP_TOP)
IDX_SHOP_RIGHT = ROUTE_RIGHT.index(SHOP_RIGHT)

# ============================================================
# 3) Exact route-level probabilities
     #Dynamic Programming
# ============================================================
def safe_reach_prob(route, target_index, p_adv, H):
    """
    Probability of reaching route[target_index] within horizon H
    on a safe route with progress probability p_adv.
    """
    nu = np.zeros(target_index + 1, dtype=float)
    nu[0] = 1.0

    for _ in range(H):
        nu_next = np.zeros_like(nu)
        for i, mass in enumerate(nu):
            if mass == 0:
                continue
            if i == target_index:
                nu_next[i] += mass
            else:
                nu_next[i + 1] += mass * p_adv
                nu_next[i] += mass * (1.0 - p_adv)
        nu = nu_next

    return float(nu[target_index])


def safe_goal_prob(route, p_adv, H):
    """
    Probability of reaching the goal within horizon H
    on a safe route with progress probability p_adv.
    """
    L = len(route) - 1
    nu = np.zeros(L + 1, dtype=float)
    nu[0] = 1.0

    for _ in range(H):
        nu_next = np.zeros_like(nu)
        for i, mass in enumerate(nu):
            if mass == 0:
                continue
            if i == L:
                nu_next[i] += mass
            else:
                nu_next[i + 1] += mass * p_adv
                nu_next[i] += mass * (1.0 - p_adv)
        nu = nu_next

    return float(nu[L])


def fast_route_probs_with_shop(route, c_zone, shop_coord, q_stuck, q_stop_shop, H, p_adv):
    """
    Returns:
      P(F S), P(F C), P(F T), P(F T ^ F S), P(F S ^ F C), P(F T ^ F S ^ F C)

    Here:
      - S means actual stopping at the shop after reaching the shop waypoint.
        The stop decision is made exactly once, upon first reaching the shop.
      - C means entering the risky center zone. The stuck decision is made
        exactly once, upon first entering C.
      - T means reaching the target

    Outside C, the driver advances with probability p_adv and otherwise stays
    in place. Conditional on surviving the one-time entry hazard, progression
    through C is deterministic.
    """
    L = len(route) - 1
    trap = L + 1

    # dimensions:
    #   route state: 0..L plus trap
    #   stop_s in {0,1}: whether actual shop stop has happened
    #   hit_c in {0,1}: whether C has been visited
    #   shop_decided in {0,1}: whether the one-time stop decision was made
    #
    # shop_decided distinguishes "not yet at the shop" from "reached the
    # shop and chose not to stop." Without it, a driver who remains at the
    # shop could receive repeated chances to stop.
    nu = np.zeros((L + 2, 2, 2, 2), dtype=float)
    nu[0, 0, 0, 0] = 1.0

    for _ in range(H):
        nu_next = np.zeros_like(nu)

        for i in range(L + 2):
            for stop_s in (0, 1):
                for hit_c in (0, 1):
                    for shop_decided in (0, 1):
                        mass = nu[i, stop_s, hit_c, shop_decided]
                        if mass == 0:
                            continue

                        if i == trap:
                            nu_next[trap, stop_s, hit_c, shop_decided] += mass
                            continue

                        if i == L:
                            nu_next[L, stop_s, hit_c, shop_decided] += mass
                            continue

                        coord = route[i]

                        # The 0.90/0.10 advance/stay model applies outside C.
                        # Once the vehicle has survived its first entry into C,
                        # it progresses deterministically through the zone.
                        if coord in c_zone:
                            movement_cases = [(i + 1, mass)]
                        else:
                            movement_cases = [
                                (i + 1, mass * p_adv),
                                (i, mass * (1.0 - p_adv)),
                            ]

                        for next_i, moved_mass in movement_cases:
                            if moved_mass == 0:
                                continue

                            next_coord = route[next_i]

                            # Make the shop decision exactly once, on the first
                            # transition that reaches the shop waypoint.
                            if next_coord == shop_coord and shop_decided == 0:
                                shop_cases = [
                                    (1, 1, moved_mass * q_stop_shop),
                                    (0, 1, moved_mass * (1.0 - q_stop_shop)),
                                ]
                            else:
                                shop_cases = [
                                    (stop_s, shop_decided, moved_mass)
                                ]

                            for stop_s2, shop_decided2, shop_mass in shop_cases:
                                if shop_mass == 0:
                                    continue

                                # The entry hazard is evaluated only on the
                                # first transition from outside C into C. Both
                                # trapped and surviving traces have visited C.
                                first_entry_to_c = (
                                    next_coord in c_zone and hit_c == 0
                                )

                                if first_entry_to_c:
                                    nu_next[
                                        trap, stop_s2, 1, shop_decided2
                                    ] += shop_mass * q_stuck
                                    nu_next[
                                        next_i, stop_s2, 1, shop_decided2
                                    ] += shop_mass * (1.0 - q_stuck)
                                else:
                                    nu_next[
                                        next_i,
                                        stop_s2,
                                        hit_c,
                                        shop_decided2,
                                    ] += shop_mass

        nu = nu_next

    p_FS_fast = float(nu[:, 1, :, :].sum())
    p_FC_fast = float(nu[:, :, 1, :].sum())
    p_FT_fast = float(nu[L, :, :, :].sum())
    p_FT_and_FS_fast = float(nu[L, 1, :, :].sum())
    p_FS_and_FC_fast = float(nu[:, 1, 1, :].sum())
    p_FT_and_FS_and_FC_fast = float(nu[L, 1, 1, :].sum())

    return (
        p_FS_fast,
        p_FC_fast,
        p_FT_fast,
        p_FT_and_FS_fast,
        p_FS_and_FC_fast,
        p_FT_and_FS_and_FC_fast,
    )


# Fast route values
(
    P_FS_FAST,
    P_FC_FAST,
    P_FT_FAST,
    P_FT_AND_FS_FAST,
    P_FS_AND_FC_FAST,
    P_FT_AND_FS_AND_FC_FAST,
) = fast_route_probs_with_shop(
    ROUTE_FAST, C_ZONE, SHOP_FAST, Q_STUCK, Q_STOP_FAST, H, P_ADV_FAST
)

# Safe-route physical shop-reach probabilities
P_REACH_SHOP_TOP = safe_reach_prob(ROUTE_TOP, IDX_SHOP_TOP, P_ADV_SAFE, H)
P_REACH_SHOP_RIGHT = safe_reach_prob(ROUTE_RIGHT, IDX_SHOP_RIGHT, P_ADV_SAFE, H)

# Actual shop-stop probabilities
P_FS_TOP = P_REACH_SHOP_TOP * Q_STOP_TOP
P_FS_RIGHT = P_REACH_SHOP_RIGHT * Q_STOP_RIGHT

# Goal probabilities
P_FT_TOP = safe_goal_prob(ROUTE_TOP, P_ADV_SAFE, H)
P_FT_RIGHT = safe_goal_prob(ROUTE_RIGHT, P_ADV_SAFE, H)

# On safe routes, reaching T implies passing the shop waypoint before T,
# but actual stopping at S is still probabilistic.
P_FT_AND_FS_TOP = P_FT_TOP * Q_STOP_TOP
P_FT_AND_FS_RIGHT = P_FT_RIGHT * Q_STOP_RIGHT

# ============================================================
# Exact syntax-DAG semantics for the selected bounded grammar
# ============================================================
# The selected grammar uses reachability F p and avoidance G !p.
# Consequently, a finite trace is summarized exactly by whether each
# atomic proposition was ever observed:
#
#     outcome = (hit_target, hit_shop, hit_center)
#
# This summary is sufficient for the grammar above. If X, U, or nested
# temporal operators are added later, replace this evaluator with LTLf
# progression or a product construction with a finite automaton.


def validate_distribution(distribution):
    """Validate and clean small floating-point errors in a distribution."""
    cleaned = {}

    for outcome, probability in distribution.items():
        probability = float(probability)
        if probability < -1e-10:
            raise ValueError(
                f"Invalid probability {probability} for outcome {outcome}"
            )
        cleaned[outcome] = max(0.0, probability)

    total = sum(cleaned.values())
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"Outcome probabilities sum to {total}, not 1.")

    return cleaned


def fast_route_outcomes():
    """
    Joint distribution of (F T, F S, F C) on the fast route.

    Every target-reaching fast-route trace passes through the center,
    so F T implies F C on this route.
    """
    p_t = P_FT_FAST
    p_s = P_FS_FAST
    p_c = P_FC_FAST
    p_ts = P_FT_AND_FS_FAST
    p_sc = P_FS_AND_FC_FAST
    p_tsc = P_FT_AND_FS_AND_FC_FAST

    distribution = {
        (1, 1, 1): p_tsc,
        (1, 1, 0): p_ts - p_tsc,
        (1, 0, 1): p_t - p_ts,
        (1, 0, 0): 0.0,
        (0, 1, 1): p_sc - p_tsc,
        (0, 1, 0): p_s - p_ts - p_sc + p_tsc,
        (0, 0, 1): p_c - p_t + p_ts - p_sc,
    }
    distribution[(0, 0, 0)] = 1.0 - sum(distribution.values())

    return validate_distribution(distribution)


def safe_route_outcomes(p_goal, p_shop, p_goal_and_shop):
    """Joint outcome distribution for a route that never visits C."""
    distribution = {
        (1, 1, 0): p_goal_and_shop,
        (1, 0, 0): p_goal - p_goal_and_shop,
        (0, 1, 0): p_shop - p_goal_and_shop,
        (0, 0, 0): 1.0 - p_goal - p_shop + p_goal_and_shop,
    }
    return validate_distribution(distribution)


ROUTE_OUTCOME_DISTRIBUTIONS = [
    fast_route_outcomes(),
    safe_route_outcomes(P_FT_TOP, P_FS_TOP, P_FT_AND_FS_TOP),
    safe_route_outcomes(P_FT_RIGHT, P_FS_RIGHT, P_FT_AND_FS_RIGHT),
]


DAG_OUTCOME_VALUE_CACHE = {}


def evaluate_dag_on_outcome(outcome):
    """
    Evaluate every shared DAG node once for one summarized trace outcome.

    For this restricted grammar, the AP-node value records whether the
    proposition occurred anywhere in the finite trace. Therefore F p equals
    the AP value and G !p equals its complement. This shortcut is exact only
    for the current F p / G !p / conjunction fragment.
    """
    outcome = tuple(int(value) for value in outcome)
    if outcome in DAG_OUTCOME_VALUE_CACHE:
        return DAG_OUTCOME_VALUE_CACHE[outcome]

    hit_target, hit_shop, hit_center = outcome
    occurred = {
        "T": bool(hit_target),
        "S": bool(hit_shop),
        "C": bool(hit_center),
    }

    node_values = np.zeros(len(SYNTAX_DAG.nodes), dtype=bool)

    # Node IDs are topologically ordered because every parent is interned only
    # after all of its children exist.
    for node_id, node in enumerate(SYNTAX_DAG.nodes):
        if node.operator == "AP":
            node_values[node_id] = occurred[node.proposition]

        elif node.operator == "F":
            node_values[node_id] = node_values[node.children[0]]

        elif node.operator == "G_NOT":
            # On a finite trace, G !p is equivalent to not(F p).
            node_values[node_id] = not node_values[node.children[0]]

        elif node.operator == "AND":
            node_values[node_id] = all(
                node_values[child]
                for child in node.children
            )

        else:
            raise ValueError(
                f"Unknown syntax-DAG operator: {node.operator}"
            )

    DAG_OUTCOME_VALUE_CACHE[outcome] = node_values
    return node_values


def route_formula_probability(root_id, outcome_distribution):
    """Exact satisfaction probability of one DAG root on one route."""
    return float(sum(
        probability
        for outcome, probability in outcome_distribution.items()
        if evaluate_dag_on_outcome(outcome)[root_id]
    ))


# Construct the route-wise probability vector [fast, top, right] for every
# automatically generated candidate root. No candidate-specific probability
# expression is written by hand.
ROUTE_FORMULA_VALUES = {}

for formula_name in FORMULAS:
    root_id = FORMULA_ROOT_IDS[formula_name]
    route_values = np.array(
        [
            route_formula_probability(root_id, distribution)
            for distribution in ROUTE_OUTCOME_DISTRIBUTIONS
        ],
        dtype=float,
    )

    ROUTE_FORMULA_VALUES[formula_name] = route_values

# ============================================================
# 4) Policy at the start
# ============================================================
# p = [p_fast, p_top, p_right]
# parameterization:
#   p = softmax([y1, y2, 0])
# ============================================================
def softmax(z):
    z = np.array(z, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / ez.sum()


def logits_to_p(y):
    y = np.asarray(y, dtype=float)
    z = np.array([y[0], y[1], 0.0], dtype=float)
    return softmax(z)


def prob_formula(phi, p):
    vals = ROUTE_FORMULA_VALUES[phi]
    return float(np.dot(vals, p))


# ============================================================
# 5) Demonstrations and MLE
# ============================================================
def generate_first_action_counts(n_trajs=80, p_true=(0.72, 0.14, 0.14), seed=18):
    rng = np.random.default_rng(seed)
    acts = rng.choice(3, size=n_trajs, p=np.array(p_true, dtype=float))
    counts = np.bincount(acts, minlength=3)
    return counts


def multinomial_loglik(p, counts):
    p = np.clip(np.array(p, dtype=float), EPS, 1.0)
    p = p / p.sum()
    return float(np.dot(counts, np.log(p)))


def mle_from_counts(counts):
    counts = np.array(counts, dtype=float)
    return counts / counts.sum()


def logits_from_p(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    p = p / p.sum()
    return np.array([np.log(p[0] / p[2]), np.log(p[1] / p[2])], dtype=float)


# ============================================================
# 6) Worst-case over log-likelihood level set using IPOPT
# ============================================================
def worst_case_ipopt(counts, values, delta, restarts=20, seed=0, max_iter=300, print_level=0):
    counts = np.asarray(counts, dtype=float)
    values = np.asarray(values, dtype=float)

    p_hat = mle_from_counts(counts)
    ll_hat = multinomial_loglik(p_hat, counts)
    ll_min = ll_hat - delta

    y_hat = logits_from_p(p_hat)
    rng = np.random.default_rng(seed)

    lb = np.array([-10.0, -10.0], dtype=DTYPE)
    ub = np.array([10.0, 10.0], dtype=DTYPE)

    cl = np.array([0.0], dtype=DTYPE)
    cu = np.array([1e19], dtype=DTYPE)

    def fd_grad(fun, x, eps=1e-7):
        x = np.asarray(x, dtype=DTYPE)
        f0 = float(fun(x))
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            g[i] = (float(fun(xp)) - f0) / eps
        return g

    class NLP:
        def objective(self, y):
            p = logits_to_p(y)
            return float(np.dot(values, p))

        def gradient(self, y):
            return fd_grad(self.objective, y)

        def constraints(self, y):
            p = logits_to_p(y)
            return np.array([multinomial_loglik(p, counts) - ll_min], dtype=DTYPE)

        def jacobian(self, y):
            eps = 1e-7
            y = np.asarray(y, dtype=DTYPE)
            c0 = float(self.constraints(y)[0])
            J = np.zeros(2, dtype=DTYPE)
            for i in range(2):
                yp = y.copy()
                yp[i] += eps
                J[i] = (float(self.constraints(yp)[0]) - c0) / eps
            return J

        def jacobianstructure(self):
            rows = np.zeros(2, dtype=int)
            cols = np.arange(2, dtype=int)
            return rows, cols

    problem = cyipopt.Problem(
        n=2,
        m=1,
        problem_obj=NLP(),
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    problem.add_option("max_iter", int(max_iter))
    problem.add_option("tol", 1e-8)
    problem.add_option("print_level", int(print_level))
    problem.add_option("mu_strategy", "adaptive")
    problem.add_option("hessian_approximation", "limited-memory")

    starts = [y_hat.copy()]
    for _ in range(restarts - 1):
        starts.append(y_hat + 1.5 * rng.standard_normal(2))

    best = None
    for y0 in starts:
        y0 = np.clip(np.asarray(y0, dtype=DTYPE), lb, ub)

        try:
            y_opt, info = problem.solve(y0)
        except Exception:
            continue

        p_opt = logits_to_p(y_opt)
        if multinomial_loglik(p_opt, counts) < ll_min - 1e-7:
            continue

        val = float(np.dot(values, p_opt))
        if best is None or val < best["val_wc"]:
            best = {
                "val_wc": val,
                "p_wc": p_opt,
                "y_wc": y_opt,
                "p_hat": p_hat,
                "ll_hat": ll_hat,
                "ll_min": ll_min,
                "ipopt_info": info,
            }

    return best


# ============================================================
# 7) Run experiment
# ============================================================
def run():
    counts = generate_first_action_counts(
        n_trajs=80,
        p_true=(0.72, 0.14, 0.14),
        seed=18
    )

    p_hat = mle_from_counts(counts)

    print("=== 2D gridworld with probabilistic shop stop S (IPOPT version) ===")
    print(f"Grid size: {GRID_N}x{GRID_N}")
    print("Start =", START)
    print("Goal  =", GOAL)
    print("C zone =", C_ZONE)
    print("Physical shop waypoints:")
    print("  fast  =", SHOP_FAST, "with stop prob", Q_STOP_FAST)
    print("  top   =", SHOP_TOP, "with stop prob", Q_STOP_TOP)
    print("  right =", SHOP_RIGHT, "with stop prob", Q_STOP_RIGHT)
    print("Horizon H =", H)
    print("Q_stuck =", Q_STUCK)
    print("Grammar: p ::= T | C | S")
    print("         psi ::= F p | G !p")
    print("         phi ::= psi | (phi ^ psi)")
    print("Maximum temporal clauses =", MAX_CLAUSES)
    print("Shared syntax-DAG nodes =", len(SYNTAX_DAG.nodes))
    print("Generated candidate formulas =", len(FORMULAS))
    if REQUIRED_PROPOSITIONS:
        print("Required propositions =", sorted(REQUIRED_PROPOSITIONS))
    print()

    print("Route lengths:")
    print("  fast shortcut =", len(ROUTE_FAST) - 1)
    print("  top safe      =", len(ROUTE_TOP) - 1)
    print("  right safe    =", len(ROUTE_RIGHT) - 1)
    print()

    print("Per-route core probabilities:")
    print("  P(F C | fast)           =", P_FC_FAST)
    print("  P(F T | fast)           =", P_FT_FAST)
    print("  P(F T | top)            =", P_FT_TOP)
    print("  P(F T | right)          =", P_FT_RIGHT)
    print("  P(F T ^ F S | fast)     =", P_FT_AND_FS_FAST)
    print("  P(F T ^ F S | top)      =", P_FT_AND_FS_TOP)
    print("  P(F T ^ F S | right)    =", P_FT_AND_FS_RIGHT)
    print("  P(F S ^ F C | fast)     =", P_FS_AND_FC_FAST)
    print("  P(F T ^ F S ^ F C | fast) =", P_FT_AND_FS_AND_FC_FAST)
    print()

    print("Demonstration counts [fast, top-safe, right-safe] =", counts)
    print("MLE route probabilities p_hat =", np.round(p_hat, 6))
    print()

    # =========================
    # Nominal timing
    # =========================
    print("=== Nominal (MLE) ===")

    nominal_start = time.perf_counter()

    nominal_vals = {}
    for phi in FORMULAS:
        nominal_vals[phi] = prob_formula(phi, p_hat)

    nominal_winner = max(FORMULAS, key=lambda phi: nominal_vals[phi])

    nominal_time = time.perf_counter() - nominal_start


    # Print results AFTER timing
    for phi in FORMULAS:
        print(f"P({phi} | p_hat) = {nominal_vals[phi]:.6f}")

    print("Nominal winner:", nominal_winner)
    print(f"Nominal execution time: {nominal_time:.6f} seconds")
    print()


    # =========================
    # Worst-case timing
    # =========================
    delta = 2.0
    print(
        f"=== Worst-case over log-likelihood level set "
        f"using IPOPT (delta={delta}) ==="
    )

    worstcase_start = time.perf_counter()

    wc_vals = {}
    wc_solutions = {}

    for i, phi in enumerate(FORMULAS):

        wc = worst_case_ipopt(
            counts=counts,
            values=ROUTE_FORMULA_VALUES[phi],
            delta=delta,
            restarts=25,
            seed=10 + i,
            max_iter=400,
            print_level=0,
        )

        wc_vals[phi] = wc["val_wc"]
        wc_solutions[phi] = wc

    robust_winner = max(FORMULAS, key=lambda phi: wc_vals[phi])

    worstcase_time = time.perf_counter() - worstcase_start


    # Print results AFTER timing
    for phi in FORMULAS:
        wc = wc_solutions[phi]
        print(
            f"{phi} worst-case = {wc['val_wc']:.6f} "
            f"at p = {np.round(wc['p_wc'], 6)}"
        )

    print("\nRobust winner:", robust_winner)
    print(
        "Worst-case policy for robust winner:",
        np.round(wc_solutions[robust_winner]["p_wc"], 6)
    )


    # =========================
    # Runtime summary
    # =========================
    print("\n=== Execution time summary ===")
    print(f"Nominal method     : {nominal_time:.6f} seconds")
    print(f"Worst-case method  : {worstcase_time:.6f} seconds")
    print(f"Number of formulas : {len(FORMULAS)}")
    print(
        f"Average WC/formula : "
        f"{worstcase_time / len(FORMULAS):.6f} seconds"
    )

    print("\n=== Nominal vs worst-case satisfaction probabilities ===")
    print("Robustness gap = nominal satisfaction - worst-case satisfaction")
    print(
        f"  {'Formula':32s} {'Nominal':>10s} "
        f"{'Worst-case':>12s} {'Gap':>10s}"
    )
    for phi in sorted(FORMULAS, key=lambda x: nominal_vals[x], reverse=True):
        gap = nominal_vals[phi] - wc_vals[phi]
        print(
            f"  {phi:32s} {nominal_vals[phi]:10.6f} "
            f"{wc_vals[phi]:12.6f} {gap:10.6f}"
        )

    print("\n=== Ranking by nominal ===")
    for phi in sorted(FORMULAS, key=lambda x: nominal_vals[x], reverse=True):
        print(f"  {phi:24s} {nominal_vals[phi]:.9f}")

    print("\n=== Ranking by worst-case ===")
    for phi in sorted(FORMULAS, key=lambda x: wc_vals[x], reverse=True):
        print(f"  {phi:24s} {wc_vals[phi]:.9f}")


if __name__ == "__main__":
    run()
