import numpy as np
import cyipopt
import scipy.sparse as sp
from scipy.special import logsumexp
from scipy.stats import norm, qmc
from dataclasses import dataclass
from itertools import combinations
import time

# ============================================================
# 1) 2D gridworld MDP with a decision in every state
# ============================================================
GRID_N = 15
START = (0, 0)
GOAL = (14, 14)

# Risky center work zone
C_ZONE = {(7, 7), (7, 8), (8, 7), (8, 8)}

# Coffee shops on the fast, top and right routes
SHOPS = ((4, 4), (0, 4), (4, 0))

# Horizon
H = 35

# Progress probabilities. Roads are the diagonal and the four borders, i.e.
# the cells of the three routes. Every other cell is a slower side street.
P_ADV_ROAD = 0.9
P_ADV_SIDE = 0.5

# One-time probability of getting stuck forever upon entering the C-zone
Q_STUCK = 0.4

# Actions. At every cell the driver chooses a monotone move. At a shop cell
# (first decision there) the action is a pair (move, stop/pass).
#   slot k = move_index + 3 * stop,  move_index: 0 = N, 1 = E, 2 = NE
MOVES = (("N", 0, 1), ("E", 1, 0), ("NE", 1, 1))
N_SLOTS = 6

# The three routes are now action sequences of this MDP:
#   fast  : NE x 14
#   top   : N x 14, then E x 14
#   right : E x 14, then N x 14


def in_grid(x, y):
    return 0 <= x < GRID_N and 0 <= y < GRID_N


def on_road(x, y):
    return x == y or x in (0, GRID_N - 1) or y in (0, GRID_N - 1)


def follows_road(x, y, move):
    """True if the move continues along a road the driver is on."""
    if move == "NE":
        return x == y                       # along the diagonal
    if move == "N":
        return x in (0, GRID_N - 1)         # along the left / right border
    return y in (0, GRID_N - 1)             # E: along the bottom / top border


# ============================================================
# 2) Parametric policy class, Eq. (1):  pi(a|s) ~ exp(theta^T f(s,a))
# ============================================================
# Features shared by every cell:
#   progress    : 2 for NE, 1 for N or E           (preference for speed)
#   follow_road : the move continues the current road (route following)
#   enter_C     : the move enters the work zone    (attitude toward C)
#   stop        : stop at the coffee shop          (coffee preference)
FEATURE_NAMES = ("progress", "follow_road", "enter_C", "stop")
D = len(FEATURE_NAMES)

# Expert used to generate demonstrations.
#   * At the start every move follows a road, so the route choice is
#     softmax(2*th_prog, th_prog, th_prog): th_prog = ln(0.72/0.14) gives
#     (p_fast, p_top, p_right) = (0.72, 0.14, 0.14).
#   * th_road keeps drivers on their road (~98% per step on a border).
#   * th_C < 0: a mild dislike of the work zone; ~1.6% of shortcut drivers
#     swerve at (6,6). A larger dislike (e.g. -4, ~8% swerving) lowers
#     P(F C) so much that F T also wins nominally. With th_C much closer to 0
#     nobody swerves in the data and the MLE of th_C does not exist.
#   * th_stop = ln(0.6/0.4): stop with probability 0.6 at a shop.
THETA_TRUE = np.array([np.log(0.72 / 0.14), 5.7, -2.5, np.log(0.6 / 0.4)])


def pol_index(x, y, decided):
    """Policy state: the cell plus the 'shop decision made' flag."""
    return (x * GRID_N + y) * 2 + decided


N_POL = GRID_N * GRID_N * 2
POL_MASK = np.zeros((N_POL, N_SLOTS), dtype=bool)
POL_FEAT = np.zeros((N_POL, N_SLOTS, D))

for x in range(GRID_N):
    for y in range(GRID_N):
        for decided in (0, 1):
            j = pol_index(x, y, decided)
            if (x, y) == GOAL:
                POL_MASK[j, 0] = True        # dummy action, never used
                continue
            shop_choice = (x, y) in SHOPS and decided == 0
            for m, (name, dx, dy) in enumerate(MOVES):
                nx, ny = x + dx, y + dy
                if not in_grid(nx, ny):
                    continue
                f = np.array([
                    2.0 if name == "NE" else 1.0,
                    float(follows_road(x, y, name)),
                    float((nx, ny) in C_ZONE and (x, y) not in C_ZONE),
                    0.0,
                ])
                if shop_choice:
                    POL_MASK[j, m] = True            # move and pass
                    POL_FEAT[j, m] = f
                    POL_MASK[j, m + 3] = True        # move and stop
                    POL_FEAT[j, m + 3] = f + np.array([0, 0, 0, 1.0])
                else:
                    POL_MASK[j, m] = True
                    POL_FEAT[j, m] = f


def policy_probs(theta):
    """pi(a|s) for every policy state; invalid actions get probability 0."""
    logits = POL_FEAT @ np.asarray(theta, dtype=float)
    logits = np.where(POL_MASK, logits, -np.inf)
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logp), logp


# ============================================================
# 3) Product state space for the LTLf fragment
# ============================================================
# Product state (x, y, decided, stopped, hc) plus absorbing trap states.
#   decided : the stop/pass decision at the current shop cell was made
#   stopped : the driver has stopped at a shop      (label S)
#   hc      : the driver has visited C              (label C)
# For the fragment F p / G !p / conjunction, the "seen" bits are exactly the
# DFA memory, so this is the product MDP x DFA of Eqs. (8)-(10). The goal is
# absorbing, so F T holds iff the trace ends at the goal.
#
# Monotone moves increase x + y on every successful step, so there are no
# loops, C is entered at most once (one-time hazard needs no extra memory),
# and every shop is visited in one continuous stay.


def state_index(x, y, decided, stopped, hc):
    return (((x * GRID_N + y) * 2 + decided) * 2 + stopped) * 2 + hc


N_GRID_STATES = GRID_N * GRID_N * 8
TRAP = (N_GRID_STATES, N_GRID_STATES + 1)       # indexed by 'stopped'
N_STATES = N_GRID_STATES + 2
START_STATE = state_index(*START, 0, 0, 0)

PROD_POL = np.full(N_STATES, -1, dtype=int)
ABSORBING = np.zeros(N_STATES, dtype=bool)
NXT = np.zeros((N_STATES, N_SLOTS, 3), dtype=int)
PRB = np.zeros((N_STATES, N_SLOTS, 3))
OUTCOME = np.zeros((N_STATES, 3), dtype=int)    # (hit_target, hit_shop, hit_center)

for s in (0, 1):
    t = TRAP[s]
    ABSORBING[t] = True
    NXT[t, :, :] = t
    PRB[t, 0, 0] = 1.0
    OUTCOME[t] = (0, s, 1)

for x in range(GRID_N):
    for y in range(GRID_N):
        for decided in (0, 1):
            for stopped in (0, 1):
                for hc in (0, 1):
                    i = state_index(x, y, decided, stopped, hc)
                    OUTCOME[i] = (int((x, y) == GOAL), stopped, hc)
                    NXT[i, :, :] = i

                    if (x, y) == GOAL:
                        ABSORBING[i] = True
                        PRB[i, 0, 0] = 1.0
                        continue

                    j = pol_index(x, y, decided)
                    PROD_POL[i] = j
                    if (x, y) in C_ZONE:
                        adv = 1.0      # deterministic once inside C
                    elif on_road(x, y):
                        adv = P_ADV_ROAD
                    else:
                        adv = P_ADV_SIDE

                    for k in np.flatnonzero(POL_MASK[j]):
                        name, dx, dy = MOVES[k % 3]
                        nx, ny = x + dx, y + dy
                        stopped2 = stopped | int(k >= 3)

                        # Successful move
                        if (nx, ny) in C_ZONE and (x, y) not in C_ZONE:
                            NXT[i, k, 0] = state_index(nx, ny, 0, stopped2, 1)
                            PRB[i, k, 0] = adv * (1.0 - Q_STUCK)
                            NXT[i, k, 1] = TRAP[stopped2]
                            PRB[i, k, 1] = adv * Q_STUCK
                        else:
                            hc2 = 1 if (nx, ny) in C_ZONE else hc
                            NXT[i, k, 0] = state_index(nx, ny, 0, stopped2, hc2)
                            PRB[i, k, 0] = adv

                        # Failed move: stay; a shop decision is not repeated
                        decided2 = 1 if (x, y) in SHOPS else 0
                        NXT[i, k, 2] = state_index(x, y, decided2, stopped2, hc)
                        PRB[i, k, 2] = 1.0 - adv

NON_ABS = np.flatnonzero(~ABSORBING)
_ROWS = np.repeat(np.arange(N_STATES), N_SLOTS * 3)
_COLS = NXT.ravel()


def product_policy(pi_pol):
    """Lift pi(a|s) from policy states to product states."""
    pi = np.zeros((N_STATES, N_SLOTS))
    pi[NON_ABS] = pi_pol[PROD_POL[NON_ABS]]
    pi[ABSORBING, 0] = 1.0
    return pi


def transition_matrix(pi_prod):
    """Induced Markov chain P^{pi,x}, Eq. (10). Row-stochastic, sparse."""
    data = (pi_prod[:, :, None] * PRB).ravel()
    return sp.csr_matrix((data, (_ROWS, _COLS)), shape=(N_STATES, N_STATES))


def forward_distributions(pi_prod):
    """nu_0, ..., nu_H by forward propagation, Eq. (11)."""
    MT = transition_matrix(pi_prod).T.tocsr()
    nus = np.zeros((H + 1, N_STATES))
    nus[0, START_STATE] = 1.0
    for t in range(H):
        nus[t + 1] = MT @ nus[t]
    return nus


# ============================================================
# 4) Bounded grammar as a shared syntax DAG
# ============================================================
#   p   ::= T | C | S
#   psi ::= F p | G !p
#   phi ::= psi | (psi_1 ^ ... ^ psi_k), 1 <= k <= MAX_CLAUSES
ATOMIC_PROPOSITIONS = ("T", "C", "S")
MAX_CLAUSES = 3
REQUIRED_PROPOSITIONS = set()


@dataclass(frozen=True)
class SyntaxNode:
    operator: str
    children: tuple = ()
    proposition: str = None


class SyntaxDAG:
    """Hash-consed, topologically ordered syntax DAG."""

    def __init__(self):
        self.nodes = []
        self._node_index = {}

    def intern(self, operator, children=(), proposition=None):
        children = tuple(children)
        if operator == "AP":
            if proposition not in ATOMIC_PROPOSITIONS or children:
                raise ValueError("Invalid AP node.")
        elif operator in {"F", "G_NOT"}:
            if proposition is not None or len(children) != 1:
                raise ValueError(f"{operator} must have exactly one child.")
        elif operator == "AND":
            children = tuple(sorted(set(children)))
            if not children:
                raise ValueError("An AND node must have at least one child.")
            if len(children) == 1:
                return children[0]
        else:
            raise ValueError(f"Unknown syntax-DAG operator: {operator}")
        if any(c < 0 or c >= len(self.nodes) for c in children):
            raise ValueError("Every child must be an existing DAG node.")
        key = (operator, children, proposition)
        if key not in self._node_index:
            self._node_index[key] = len(self.nodes)
            self.nodes.append(SyntaxNode(operator, children, proposition))
        return self._node_index[key]

    def render(self, node_id):
        node = self.nodes[node_id]
        if node.operator == "AP":
            return node.proposition
        if node.operator == "F":
            return f"F {self.render(node.children[0])}"
        if node.operator == "G_NOT":
            return f"G !{self.render(node.children[0])}"
        return "(" + " ^ ".join(self.render(c) for c in node.children) + ")"


SYNTAX_DAG = SyntaxDAG()
AP_NODE_IDS = {ap: SYNTAX_DAG.intern("AP", proposition=ap)
               for ap in ATOMIC_PROPOSITIONS}
CLAUSE_NODE_IDS = {}
for ap in ATOMIC_PROPOSITIONS:
    CLAUSE_NODE_IDS[("F", ap)] = SYNTAX_DAG.intern("F", (AP_NODE_IDS[ap],))
for ap in ATOMIC_PROPOSITIONS:
    CLAUSE_NODE_IDS[("G_NOT", ap)] = SYNTAX_DAG.intern(
        "G_NOT", (AP_NODE_IDS[ap],))


def generate_candidate_roots(max_clauses=MAX_CLAUSES):
    roots, seen = [], set()
    for n in range(1, max_clauses + 1):
        for spec in combinations(list(CLAUSE_NODE_IDS), n):
            if any(("F", ap) in spec and ("G_NOT", ap) in spec
                   for ap in ATOMIC_PROPOSITIONS):
                continue
            if not REQUIRED_PROPOSITIONS.issubset({ap for _, ap in spec}):
                continue
            ids = tuple(CLAUSE_NODE_IDS[c] for c in spec)
            root = ids[0] if len(ids) == 1 else SYNTAX_DAG.intern("AND", ids)
            if root not in seen:
                seen.add(root)
                roots.append(root)
    return tuple(roots)


CANDIDATE_ROOT_IDS = generate_candidate_roots()
FORMULA_ROOT_IDS = {SYNTAX_DAG.render(r): r for r in CANDIDATE_ROOT_IDS}
FORMULAS = list(FORMULA_ROOT_IDS)


def evaluate_dag_on_outcome(outcome):
    """Evaluate every DAG node on (hit_target, hit_shop, hit_center)."""
    occurred = dict(zip(("T", "S", "C"), map(bool, outcome)))
    values = np.zeros(len(SYNTAX_DAG.nodes), dtype=bool)
    for node_id, node in enumerate(SYNTAX_DAG.nodes):
        if node.operator == "AP":
            values[node_id] = occurred[node.proposition]
        elif node.operator == "F":
            values[node_id] = values[node.children[0]]
        elif node.operator == "G_NOT":
            values[node_id] = not values[node.children[0]]
        else:
            values[node_id] = all(values[c] for c in node.children)
    return values


# Acceptance indicator I_F of Eq. (12) for every formula and product state.
_OUTCOME_VALUES = {o: evaluate_dag_on_outcome(o)
                   for o in {tuple(r) for r in OUTCOME}}
ACCEPT = {
    phi: np.array([_OUTCOME_VALUES[tuple(o)][root] for o in OUTCOME],
                  dtype=float)
    for phi, root in FORMULA_ROOT_IDS.items()
}


# ============================================================
# 5) Satisfaction probability P_H(phi | pi_theta) and its gradient
# ============================================================
class SatisfactionModel:
    """Caches the forward pass so objective and gradient share it."""

    def __init__(self):
        self._key = None

    def _prepare(self, theta):
        key = np.asarray(theta, dtype=float).tobytes()
        if key != self._key:
            self.pi_pol, _ = policy_probs(theta)
            self.pi_prod = product_policy(self.pi_pol)
            self.M = transition_matrix(self.pi_prod)
            MT = self.M.T.tocsr()
            nus = np.zeros((H + 1, N_STATES))
            nus[0, START_STATE] = 1.0
            for t in range(H):
                nus[t + 1] = MT @ nus[t]
            self.nus = nus
            self._key = key

    def value(self, theta, accept):
        """Eq. (12): P_H = sum_s nu_H(s) I_F(s)."""
        self._prepare(theta)
        return float(self.nus[H] @ accept)

    def gradient(self, theta, accept):
        """
        Exact gradient by one backward pass:
          V_H = I_F,   V_t = P^{pi,x} V_{t+1}
          dP_H / dpi(a|s) = sum_t nu_t(s) sum_s' P(s'|s,a) V_{t+1}(s')
        followed by the softmax chain rule of Eq. (1).
        """
        self._prepare(theta)
        V = np.zeros((H + 1, N_STATES))
        V[H] = accept
        for t in range(H - 1, -1, -1):
            V[t] = self.M @ V[t + 1]
        Q = np.einsum("nko,tnko->tnk", PRB, V[1:][:, NXT])
        G = np.einsum("tn,tnk->nk", self.nus[:H], Q)

        G_pol = np.zeros((N_POL, N_SLOTS))
        np.add.at(G_pol, PROD_POL[NON_ABS], G[NON_ABS])
        weight = G_pol * self.pi_pol
        f_bar = np.einsum("jk,jkd->jd", self.pi_pol, POL_FEAT)
        return (np.einsum("jk,jkd->d", weight, POL_FEAT)
                - np.einsum("j,jd->d", weight.sum(axis=1), f_bar))

    def all_values(self, theta):
        self._prepare(theta)
        return {phi: float(self.nus[H] @ ACCEPT[phi]) for phi in FORMULAS}


# ============================================================
# 6) Demonstrations, likelihood (Eq. 4) and MLE (Eq. 5)
# ============================================================
def generate_demonstrations(n_trajs=80, theta_true=THETA_TRUE, seed=18):
    """
    Simulate n_trajs length-H trajectories of the expert and return the
    state-action counts n[s, a] (the sufficient statistic of Eq. (4)).
    """
    rng = np.random.default_rng(seed)
    pi_true, _ = policy_probs(theta_true)
    counts = np.zeros((N_POL, N_SLOTS))
    for _ in range(n_trajs):
        i = START_STATE
        for _ in range(H):
            if ABSORBING[i]:
                break
            j = PROD_POL[i]
            k = rng.choice(N_SLOTS, p=pi_true[j])
            counts[j, k] += 1
            i = NXT[i, k, rng.choice(3, p=PRB[i, k])]
    return counts


def loglik(theta, counts):
    """log P(D | pi_theta) up to the constant C of Eq. (4)."""
    _, logp = policy_probs(theta)
    return float(np.sum(counts * np.where(counts > 0, logp, 0.0)))


def loglik_grad_hess(theta, counts):
    pi, _ = policy_probs(theta)
    n_state = counts.sum(axis=1)
    f_bar = np.einsum("jk,jkd->jd", pi, POL_FEAT)
    grad = (np.einsum("jk,jkd->d", counts, POL_FEAT)
            - np.einsum("j,jd->d", n_state, f_bar))
    second = np.einsum("jk,jkd,jke->jde", pi, POL_FEAT, POL_FEAT)
    cov = second - np.einsum("jd,je->jde", f_bar, f_bar)
    hess = -np.einsum("j,jde->de", n_state, cov)
    return grad, hess


def mle(counts, tol=1e-10, max_iter=200):
    """Newton's method on the concave log-likelihood."""
    theta = np.zeros(D)
    ll = loglik(theta, counts)
    for _ in range(max_iter):
        grad, hess = loglik_grad_hess(theta, counts)
        if np.max(np.abs(grad)) < tol:
            break
        step = np.linalg.solve(-hess, grad)
        t = 1.0
        while True:
            cand = theta + t * step
            ll_cand = loglik(cand, counts)
            if ll_cand >= ll or t < 1e-12:
                break
            t *= 0.5
        theta, ll = cand, ll_cand
    if np.max(np.abs(theta)) > 30:
        raise RuntimeError(
            "The MLE diverges: the demonstrations are separable for some "
            "feature (an action that is never deviated from). Use more "
            "demonstrations or a different expert / feature map.")
    return theta


# ============================================================
# 7) Worst case over Pi_delta(D), Eq. (13): multi-start IPOPT
# ============================================================
# min_theta P_H(phi | pi_theta)  s.t.  log P(D|pi_theta) >= log P(D|pi_hat) - delta
#
# The log-likelihood of a softmax policy with linear features is concave in
# theta, so the uncertainty set is CONVEX in theta. P_H is a polynomial of
# degree up to H in pi and is nonconvex, hence multi-start.
#
# Starts are placed on rays from the MLE (feasible by convexity):
#   * the MLE itself
#   * +/- each principal direction of the MLE covariance (inverse Fisher)
#   * the natural-gradient descent direction of the objective
#   * scrambled Sobol directions and ray lengths
# The search stops once the best value is reached by `min_hits` starts and
# the last batch did not improve it.
THETA_BOX = 25.0


def _ray_length(center, direction, counts, ll_min):
    with np.errstate(divide="ignore"):
        t_box = np.min(np.where(direction != 0,
                                (np.sign(direction) * THETA_BOX - center)
                                / direction, np.inf))
    if not np.isfinite(t_box):
        return 0.0
    if loglik(center + t_box * direction, counts) >= ll_min:
        return t_box
    lo, hi = 0.0, t_box
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if loglik(center + mid * direction, counts) >= ll_min:
            lo = mid
        else:
            hi = mid
    return lo


def _count_distinct(values, tol):
    values = np.sort(np.asarray(values))
    return 0 if values.size == 0 else int(1 + np.sum(np.diff(values) > tol))


def worst_case_multistart(phi, counts, theta_hat, delta, model,
                          batch_size=8, min_starts=24, max_starts=120,
                          min_hits=3, value_tol=1e-7, seed=0,
                          max_iter=300, print_level=0):
    accept = ACCEPT[phi]
    ll_hat = loglik(theta_hat, counts)
    ll_min = ll_hat - delta

    class NLP:
        def objective(self, theta):
            return model.value(theta, accept)

        def gradient(self, theta):
            return model.gradient(theta, accept)

        def constraints(self, theta):
            return np.array([loglik(theta, counts) - ll_min])

        def jacobian(self, theta):
            return loglik_grad_hess(theta, counts)[0]

    problem = cyipopt.Problem(
        n=D, m=1, problem_obj=NLP(),
        lb=np.full(D, -THETA_BOX), ub=np.full(D, THETA_BOX),
        cl=np.array([0.0]), cu=np.array([1e19]),
    )
    problem.add_option("sb", "yes")
    problem.add_option("print_level", int(print_level))
    problem.add_option("max_iter", int(max_iter))
    problem.add_option("tol", 1e-9)
    problem.add_option("acceptable_tol", 1e-7)
    problem.add_option("mu_strategy", "adaptive")
    problem.add_option("hessian_approximation", "limited-memory")

    # Starting points
    _, hess = loglik_grad_hess(theta_hat, counts)
    cov = np.linalg.inv(-hess)
    eigval, eigvec = np.linalg.eigh(cov)
    sqrt_cov = eigvec @ np.diag(np.sqrt(np.clip(eigval, 0, None)))

    def along(direction, frac):
        t = _ray_length(theta_hat, direction, counts, ll_min)
        return theta_hat + frac * t * direction

    directions = [s * sqrt_cov[:, i] for i in range(D) for s in (1, -1)]
    descent = -cov @ model.gradient(theta_hat, accept)
    if np.linalg.norm(descent) > 0:
        directions.append(descent)
    pending = [theta_hat.copy()]
    for d in directions:
        pending += [along(d, 0.5), along(d, 0.95)]
    sobol = qmc.Sobol(d=D + 1, scramble=True, seed=seed)

    values, solutions = [], []
    n_tried = n_failed = 0
    best_val = np.inf
    converged = False

    while n_tried < max_starts:
        if not pending:
            u = np.clip(sobol.random(batch_size), 1e-12, 1 - 1e-12)
            for row in u:
                d = sqrt_cov @ norm.ppf(row[:D])
                frac = 0.02 + 0.96 * row[D] ** (1.0 / D)
                pending.append(along(d, frac))
        batch, pending = pending[:batch_size], pending[batch_size:]
        best_before = best_val

        for theta0 in batch:
            if n_tried >= max_starts:
                break
            n_tried += 1
            try:
                theta_opt, info = problem.solve(np.asarray(theta0, float))
            except Exception:
                n_failed += 1
                continue
            feasible = loglik(theta_opt, counts) >= ll_min - 1e-6
            if info["status"] not in (0, 1) or not feasible:
                n_failed += 1
                continue
            val = model.value(theta_opt, accept)
            values.append(val)
            solutions.append(theta_opt)
            best_val = min(best_val, val)

        hits = int(np.sum(np.asarray(values) <= best_val + value_tol))
        improved = best_val < best_before - value_tol
        if n_tried >= min_starts and hits >= min_hits and not improved:
            converged = True
            break

    if not values:
        raise RuntimeError(f"IPOPT failed from every start for {phi}.")

    best = int(np.argmin(values))
    return {
        "val_wc": values[best],
        "theta_wc": solutions[best],
        "n_starts": n_tried,
        "n_failed": n_failed,
        "hits": int(np.sum(np.asarray(values) <= values[best] + value_tol)),
        "n_local_minima": _count_distinct(values, 1e-6),
        "converged": converged,
    }


# ============================================================
# 8) Diagnostics
# ============================================================
ROUTE_ACTIONS = {
    # cell -> move index (0 = N, 1 = E, 2 = NE) for a route-following driver
    "fast": {(k, k): 2 for k in range(GRID_N - 1)},
    "top": {**{(0, y): 0 for y in range(GRID_N - 1)},
            **{(x, GRID_N - 1): 1 for x in range(GRID_N - 1)}},
    "right": {**{(x, 0): 1 for x in range(GRID_N - 1)},
              **{(GRID_N - 1, y): 0 for y in range(GRID_N - 1)}},
}


def route_policy(route, stop_prob):
    """Deterministic route-following policy (pins the old three-route model)."""
    pi = np.zeros((N_POL, N_SLOTS))
    for x in range(GRID_N):
        for y in range(GRID_N):
            for decided in (0, 1):
                j = pol_index(x, y, decided)
                valid = np.flatnonzero(POL_MASK[j, :3])
                m = ROUTE_ACTIONS[route].get((x, y), valid[0])
                if POL_MASK[j, m + 3]:
                    pi[j, m], pi[j, m + 3] = 1 - stop_prob, stop_prob
                else:
                    pi[j, m] = 1.0
    return pi


def pinned_route_values(route, stop_prob=0.6):
    nus = forward_distributions(product_policy(route_policy(route, stop_prob)))
    return {phi: float(nus[H] @ ACCEPT[phi]) for phi in FORMULAS}


def describe_policy(theta):
    """A few interpretable behaviors of pi_theta."""
    pi, _ = policy_probs(theta)
    start = pi[pol_index(0, 0, 0)]
    border = pi[pol_index(0, 2, 0)]      # a left-border cell
    work_zone = pi[pol_index(6, 6, 0)]   # last diagonal cell before C
    shop = pi[pol_index(4, 4, 0)]
    return {
        "route choice at start (fast, top, right)":
            np.round([start[2], start[0], start[1]], 4),
        "stay on border road per step": round(float(border[0]), 4),
        "enter C at (6,6)": round(float(work_zone[2]), 4),
        "stop at shop": round(float(shop[3:].sum()), 4),
    }


# ============================================================
# 9) Run experiment
# ============================================================
def run(n_trajs=80, seed=18, delta=4.0):
    model = SatisfactionModel()
    counts = generate_demonstrations(n_trajs, THETA_TRUE, seed)
    theta_hat = mle(counts)

    print("=== 2D gridworld MDP, actions N/E/NE in every state ===")
    print(f"Grid {GRID_N}x{GRID_N}, start {START}, goal {GOAL}, H = {H}")
    print("C zone =", sorted(C_ZONE), " Q_stuck =", Q_STUCK)
    print("Shops =", SHOPS)
    print(f"Progress probability: roads {P_ADV_ROAD}, "
          f"side streets {P_ADV_SIDE}")
    print("Features:", FEATURE_NAMES)
    print("Candidate formulas =", len(FORMULAS),
          " Shared syntax-DAG nodes =", len(SYNTAX_DAG.nodes))
    print(f"Product states = {N_STATES}, policy states = {N_POL}")
    print()

    print("Pinned-route check (old three-route model, stop prob 0.6):")
    for route in ("fast", "top", "right"):
        v = pinned_route_values(route)
        print(f"  {route:6s} P(F T) = {v['F T']:.6f}  P(F C) = {v['F C']:.6f}"
              f"  P(F S) = {v['F S']:.6f}")
    print()

    print(f"Demonstrations: {n_trajs} trajectories, "
          f"{int(counts.sum())} state-action pairs")
    print("theta_true =", np.round(THETA_TRUE, 4))
    print("theta_hat  =", np.round(theta_hat, 4))
    _, hess = loglik_grad_hess(theta_hat, counts)
    print("std error  =", np.round(np.sqrt(np.diag(np.linalg.inv(-hess))), 4))
    for name, val in describe_policy(theta_hat).items():
        print(f"  MLE {name}: {val}")
    print()

    # Nominal
    nominal_start = time.perf_counter()
    nominal_vals = model.all_values(theta_hat)
    nominal_winner = max(FORMULAS, key=lambda phi: nominal_vals[phi])
    nominal_time = time.perf_counter() - nominal_start

    # Robust
    worstcase_start = time.perf_counter()
    wc = {}
    for i, phi in enumerate(FORMULAS):
        wc[phi] = worst_case_multistart(phi, counts, theta_hat, delta, model,
                                        seed=10 + i)
    worstcase_time = time.perf_counter() - worstcase_start
    wc_vals = {phi: wc[phi]["val_wc"] for phi in FORMULAS}
    robust_winner = max(FORMULAS, key=lambda phi: wc_vals[phi])

    print(f"=== Worst case over Pi_delta (multi-start IPOPT, delta={delta}) ===")
    print(f"  {'Formula':24s} {'Nominal':>9s} {'Worst':>9s} {'Gap':>8s} "
          f"{'starts':>7s} {'hits':>5s} {'minima':>7s} {'conv':>5s}")
    for phi in sorted(FORMULAS, key=lambda p: wc_vals[p], reverse=True):
        r = wc[phi]
        print(f"  {phi:24s} {nominal_vals[phi]:9.6f} {r['val_wc']:9.6f} "
              f"{nominal_vals[phi] - r['val_wc']:8.4f} {r['n_starts']:7d} "
              f"{r['hits']:5d} {r['n_local_minima']:7d} "
              f"{str(r['converged']):>5s}")
    print("  starts = IPOPT runs, hits = runs reaching the best value,")
    print("  minima = distinct local minimum values found")

    print("\n=== Ranking by nominal ===")
    for phi in sorted(FORMULAS, key=lambda p: nominal_vals[p], reverse=True)[:5]:
        print(f"  {phi:24s} {nominal_vals[phi]:.6f}")
    print("\n=== Ranking by worst case ===")
    for phi in sorted(FORMULAS, key=lambda p: wc_vals[p], reverse=True)[:5]:
        print(f"  {phi:24s} {wc_vals[phi]:.6f}")

    print("\nNominal winner:", nominal_winner,
          f"({nominal_vals[nominal_winner]:.6f})")
    print("Robust winner :", robust_winner,
          f"({wc_vals[robust_winner]:.6f})")
    print("Worst-case policy for the robust winner:")
    print("  theta =", np.round(wc[robust_winner]["theta_wc"], 4))
    for name, val in describe_policy(wc[robust_winner]["theta_wc"]).items():
        print(f"  {name}: {val}")

    print("\n=== Execution time summary ===")
    print(f"Nominal method     : {nominal_time:.6f} seconds")
    print(f"Worst-case method  : {worstcase_time:.6f} seconds")
    print(f"Number of formulas : {len(FORMULAS)}")
    print(f"Average WC/formula : {worstcase_time / len(FORMULAS):.6f} seconds")

    return nominal_vals, wc


if __name__ == "__main__":
    run()
