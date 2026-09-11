"""Human activity specification selection with IPOPT.
All model numbers are synthetic assumptions, not clinical advice.
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import dataclass, asdict, replace
from itertools import product
from pathlib import Path

import numpy as np

# 1) Settings, human actions, and environment dynamics
# ============================================================
BLOCKS = ('morning', 'afternoon', 'evening', 'night')
WEATHER = ('good', 'bad')
ACTIONS = ('usual_activity', 'indoor_walk', 'outdoor_walk')
ACTIVITY = ('low', 'medium', 'high')
STEPS = np.array([500, 1500, 3000], dtype=int)
# Rows: action; columns: realized low/medium/high block step volume.
# Usual activity includes incidental walking rather than compulsory inactivity.
STEP_KERNEL = np.array([
    [[.85, .15, .00], [.10, .65, .25], [.05, .20, .75]],  # good weather
    [[.85, .15, .00], [.10, .65, .25], [.45, .40, .15]],  # bad weather
])
WEATHER_KERNEL = np.array([[.8, .2], [.3, .7]])
INITIAL_WEATHER = np.array([.6, .4])
AP = BLOCKS + WEATHER + ACTIVITY + ('goal',)


@dataclass(frozen=True)
class Settings:
    days: int = 30
    seed: int = 16
    goal: int = 5500
    delta: float = 4.0
    # Underlying behavior is used "only as the reference / deployment policy".
    # This is intentionally separated from the demonstration-generating policy.
    true_weights: tuple = (.40, .35, .25)

    # One latent behavioral mode is sampled once at the beginning of each day
    # and then used for all four blocks of that day.
    # Components: 0=morning-oriented, 1=distributed, 2=goal-responsive.
    demo_weights: tuple = (.82, .12, .06)
    

    restarts: int = 15
    grid_resolution: int = 60
    max_iter: int = 500
    solver: str = 'auto'  # auto prefers cyipopt, otherwise casadi/IPOPT
    print_level: int = 0
    max_operators: int = 6
    max_depth: int = 4
    output_dir: str = 'general_activity_results'
    # Select families before examining results; None enables all families.
    families: tuple | None = None


SETTINGS = Settings()

@dataclass(frozen=True)
class State:
    block: int
    cumulative: int
    weather: int
    previous_activity: int = -1


def basis_policies(state, goal):
    """Three known behavioral components; rows sum to one over human actions.

    0: morning-oriented, 1: distributed, 2: goal-responsive.
    Each row is a state-dependent policy. In the trajectory-mixture model,
    one row is selected once per day and retained for the entire trajectory.
    """
    b, c, w, previous = state.block, state.cumulative, state.weather, state.previous_activity
    walk = np.array([[.90, .30, .25, .08][b],
                     [.65, .65, .65, .12][b],
                     [.35, .50, .65, .25][b]])
    if c < goal * (b + 1) / 4:        # This checks whether the participant is behind an expected cumulative target.
        walk[2] = [.45, .70, .90, .70][b]
    if c >= goal:
        walk *= .35                  # Less motivation for a planned walk after meeting the goal. Once the daily step goal has already been reached, all three policies reduce their probability of intentionally walking by 65%.
    if previous == 2:                # After a high-activity block
        walk *= np.array([.75, .85, .80])  # Behavioral persistence/recovery assumption.
    outdoor_share = np.array([.90, .65, .75] if w == 0 else [.35, .10, .05])
    return np.column_stack((1-walk, walk*(1-outdoor_share), walk*outdoor_share))


def label_completed_block(block, weather, level, cumulative, goal):
    labels = {BLOCKS[block], WEATHER[weather], ACTIVITY[level]}
    if cumulative >= goal:
        labels.add('goal')
    return frozenset(labels)


def transition_distribution(state, action, goal):
    """Explicit finite kernel: (probability, next decision state, emitted label).

    A label describes the block JUST completed. The next state's weather is
    for the NEXT decision and is never used to relabel the preceding block.
    None is terminal: after night there is no self-loop.
    Transition labels are equivalent to an ordinary labeled MDP by retaining
    the emitted label in an augmented successor state.
    """
    for level, py in enumerate(STEP_KERNEL[state.weather, action]):
        if py == 0: continue
        cumulative = min(goal, state.cumulative + int(STEPS[level]))
        label = label_completed_block(state.block, state.weather, level, cumulative, goal) # creates the APs associated with the completed block.
        if state.block == 3:                     # night is the final block/ terminal
            yield float(py), None, label         # no self-loop for LTL_f
        else:
            for next_weather, pw in enumerate(WEATHER_KERNEL[state.weather]):
                yield float(py*pw), State(state.block+1, cumulative, next_weather, level), label


# ============================================================
# 2) Shared syntax DAG and full finite-trace semantics
# ============================================================
class SyntaxDAG:
    def __init__(self):
        self.nodes = []
        self.index = {}

    def intern(self, op, *children):
        if op == 'AP':
            if len(children) != 1 or children[0] not in AP: raise ValueError(children)
        elif op in ('!', 'F', 'G', 'X'):
            if len(children) != 1: raise ValueError(op)
        elif op in ('&', '|', 'U'):
            if len(children) != 2: raise ValueError(op)
            if op in ('&', '|'):
                children = tuple(sorted(children))
                if children[0] == children[1]: return children[0]
        else: raise ValueError(op)
        if op != 'AP' and any(not isinstance(c, int) or not 0 <= c < len(self.nodes) for c in children):
            raise ValueError('Children must already exist')
        key = (op, tuple(children))
        if key not in self.index:
            self.index[key] = len(self.nodes)
            self.nodes.append(key)
        return self.index[key]

    def render(self, root):
        op, ch = self.nodes[root]
        if op == 'AP': return ch[0]
        if len(ch) == 1: return f'{op}({self.render(ch[0])})'
        return f'({self.render(ch[0])} {op} {self.render(ch[1])})'

    def evaluate_all(self, trace):
        """Evaluate each shared node at every position by backward recursion.

        Strong X is false at the last position; strong U requires its right
        operand before trace termination. No ever-visited shortcut is used.
        """
        if not trace: raise ValueError('Empty traces are not used')
        n = len(trace)
        v = np.zeros((len(self.nodes), n), dtype=bool)
        for root, (op, ch) in enumerate(self.nodes):
            if op == 'AP': v[root] = [ch[0] in label for label in trace]
            elif op == '!': v[root] = ~v[ch[0]]
            elif op == '&': v[root] = v[ch[0]] & v[ch[1]]
            elif op == '|': v[root] = v[ch[0]] | v[ch[1]]
            elif op == 'X': v[root, :-1] = v[ch[0], 1:]
            else:
                for t in range(n-1, -1, -1):
                    future = v[root, t+1] if t+1 < n else (op == 'G')
                    if op == 'F': v[root,t] = v[ch[0],t] or future
                    elif op == 'G': v[root,t] = v[ch[0],t] and future
                    elif op == 'U': v[root,t] = v[ch[1],t] or (v[ch[0],t] and future)
        return v[:, 0]


@dataclass(frozen=True)
class Candidate:
    name: str
    root: int
    description: str


def formula_complexity(dag, root):
    """Count operator OCCURRENCES in the expanded tree, not unique DAG nodes.

    AP leaves have depth zero and cost zero. Every !, &, |, F, G, X, U
    occurrence costs one. Repeated shared subformulas count each occurrence.
    """
    op, children = dag.nodes[root]
    if op == 'AP': return 0, 0
    child_costs = [formula_complexity(dag, child) for child in children]
    return 1 + sum(c[0] for c in child_costs), 1 + max(c[1] for c in child_costs)




def generate_templates(max_operators=6, max_depth=4):
    """Domain-expert-defined LTLf templates instantiated over AP combinations.

    Candidate families:
        F(block & level)
        F(block & weather & level)
        F(weather & goal)
        F(weather) & G(weather -> !low)
        F(goal)
        F(weather) & G(weather -> !low)
    Level is medium or high. Weather-only formulas are excluded.
    Implication is encoded as !weather | !low.
    Conditional candidates require the weather to occur at least once.
    They use 6 operators and a syntax depth of 4.

    Candidates use a shared syntax DAG and satisfy the specified
    operator-count and syntax-depth limits.
    """
    d = SyntaxDAG()
    a = {name: d.intern('AP', name) for name in AP}

    F = lambda x: d.intern('F', x)
    G = lambda x: d.intern('G', x)
    NOT = lambda x: d.intern('!', x)
    AND = lambda x, y: d.intern('&', x, y)
    OR = lambda x, y: d.intern('|', x, y)

    candidates = []

    def add(name, root, description):
        operators, depth = formula_complexity(d, root)
        if operators <= max_operators and depth <= max_depth:
            candidates.append(Candidate(name, root, description))

    # Activity during a particular time block.
    for block in BLOCKS:
        for level in ('medium', 'high'):
            add(
                f'{block}_{level}',
                F(AND(a[block], a[level])),
                f'Have {level} activity during {block}.'
            )

    for weather in WEATHER:
        # Activity combining time of day and weather.
        for block in BLOCKS:
            for level in ('medium', 'high'):
                add(
                    f'{block}_{weather}_{level}',
                    F(AND(AND(a[block], a[weather]), a[level])),
                    f'Have {level} activity during {block} '
                    f'with {weather} weather.'
                )

        # The goal has been reached by the end of this block,
        # but may have first been reached in an earlier block.
        add(
            f'{weather}_goal',
            F(AND(a[weather], a['goal'])),
            f'Have reached the daily goal by the end of a block '
            f'with {weather} weather.'
        )

        # F(weather) & G(weather -> !low): require weather occurrence.
        add(
            f'avoid_low_when_{weather}',
            AND(
                F(a[weather]),
                G(OR(NOT(a[weather]), NOT(a['low'])))
            ),
            f'Experience {weather} weather at least once and have '
            f'medium or high activity in every {weather}-weather block.'
        )

    goal_root = F(a['goal'])
    add(
        'reach_daily_goal',
        goal_root,
        'Reach the daily step goal.'
    )

    if not candidates:
        raise ValueError('No templates satisfy the complexity limits')

    return d, candidates, goal_root

# ============================================================
# 3) Full demonstration days and likelihood features
# ============================================================
def generate_demonstrations(cfg):
    """Generate biased demonstrations using one latent mode per day.

    A single discrete choice is made at the beginning of the trajectory.  Conditional on that mode, the
    corresponding state-dependent basis policy is used in every block.

    The latent mode is stored only for simulation diagnostics; estimation below
    does not use it.
    """
    rng = np.random.default_rng(cfg.seed)
    demonstrations = []
    for day in range(cfg.days):
        mode = int(rng.choice(3, p=np.asarray(cfg.demo_weights, dtype=float)))
        state = State(0, 0, int(rng.choice(2, p=INITIAL_WEATHER)), -1)
        observations = []
        while state is not None:
            action_probs = basis_policies(state, cfg.goal)[mode]
            action = int(rng.choice(3, p=action_probs))
            outcomes = list(transition_distribution(state, action, cfg.goal))
            chosen = int(rng.choice(len(outcomes), p=[x[0] for x in outcomes]))
            _, next_state, label = outcomes[chosen]
            level = next(i for i, name in enumerate(ACTIVITY) if name in label)
            observations.append(dict(day=day, latent_mode=mode,
                latent_mode_name=('morning_oriented','distributed','goal_responsive')[mode],
                block=state.block, time=BLOCKS[state.block],
                cumulative_before=state.cumulative, weather=WEATHER[state.weather],
                previous_activity=state.previous_activity, action=ACTIONS[action],
                steps=int(STEPS[level]), cumulative_after=min(cfg.goal, state.cumulative+int(STEPS[level])),
                labels=' '.join(sorted(label))))
            state = next_state
        demonstrations.append(observations)
    return demonstrations


def likelihood_features(demonstrations, cfg):
    """Per-day likelihood under each latent behavioral mode.

    For day d and mode j,
        L[d,j] = product_t pi_j(a_dt | s_dt).
    The environment transition factors are omitted because, conditional on the
    observed actions/states, they do not depend on the mixture weights p.

    Hence the mixture likelihood is
        sum_d log(sum_j p_j L[d,j]).
    """
    rows = []
    for day in demonstrations:
        day_like = np.ones(3, dtype=float)
        for r in day:
            state = State(r['block'], r['cumulative_before'],
                          WEATHER.index(r['weather']), r['previous_activity'])
            probs = basis_policies(state, cfg.goal)[:, ACTIONS.index(r['action'])]
            day_like *= probs
        rows.append(day_like)
    return np.asarray(rows)


# ============================================================
# 4) Exact model enumeration and satisfaction polynomials
# ============================================================
class Polynomial:
    """Polynomial in three mixture weights; degree one in this model."""
    def __init__(self, exponents, coefficients):
        self.exponents = np.asarray(exponents,dtype=int)
        self.coefficients = np.asarray(coefficients,dtype=float)

    def value(self,p):
        return float(self.coefficients @ np.prod(np.asarray(p)[None,:]**self.exponents,axis=1))

    def gradient(self,p):
        out = np.zeros(3)
        for j in range(3):
            mask = self.exponents[:,j] > 0
            e = self.exponents[mask].copy(); factors=e[:,j].copy(); e[:,j]-=1
            out[j] = (self.coefficients[mask]*factors) @ np.prod(np.asarray(p)[None,:]**e,axis=1)
        return out

    def hessian(self,p):
        h = np.zeros((3,3))
        for j in range(3):
            for k in range(3):
                mask = (self.exponents[:,j] >= (2 if j==k else 1)) & (self.exponents[:,k] >= 1)
                e=self.exponents[mask].copy()
                factor=e[:,j]*(e[:,k]-(j==k)); e[:,j]-=1; e[:,k]-=1
                h[j,k]=(self.coefficients[mask]*factor) @ np.prod(np.asarray(p)[None,:]**e,axis=1)
        return h


def compile_probabilities(cfg, dag, candidates, goal_root):
    """Compute exact satisfaction probabilities by finite enumeration.

    One behavioral mode is selected per day:
        P(phi | p) = sum_j p_j P(phi | mode=j).

    Enumerate all four-block weather and activity sequences.
    Actions are marginalized using the selected mode's policy.
    """
    exponents = np.eye(3, dtype=int)

    # Rows: candidate probabilities, goal diagnostic, total probability.
    coefficients = np.zeros((len(candidates) + 2, 3), dtype=float)
    truth_vectors = []
    roots = [candidate.root for candidate in candidates] + [goal_root]

    for mode in range(3):
        for ws in product(range(2), repeat=4):
            weather_prob = float(INITIAL_WEATHER[ws[0]])
            for b in range(3):
                weather_prob *= WEATHER_KERNEL[ws[b], ws[b + 1]]

            for levels in product(range(3), repeat=4):
                mass = weather_prob
                cumulative = 0
                previous = -1
                trace = []

                for b, (weather, level) in enumerate(zip(ws, levels)):
                    state = State(b, cumulative, weather, previous)
                    action_probs = basis_policies(state, cfg.goal)[mode]

                    # Sum over possible actions:
                    # P(level | state, mode)
                    # = sum_a pi_mode(a | state) P(level | weather, a).
                    level_prob = float(
                        action_probs @ STEP_KERNEL[weather, :, level]
                    )
                    mass *= level_prob

                    if mass == 0:
                        break

                    cumulative = min(
                        cfg.goal,
                        cumulative + int(STEPS[level])
                    )
                    previous = level
                    trace.append(
                        label_completed_block(
                            b, weather, level, cumulative, cfg.goal
                        )
                    )

                if mass == 0:
                    continue

                truth = dag.evaluate_all(trace)[roots]
                truth_vectors.append(truth)

                coefficients[:-1, mode] += truth * mass
                coefficients[-1, mode] += mass

    polynomials = [
        Polynomial(exponents, row)
        for row in coefficients
    ]

    return (
        polynomials[:-2],       # Candidate satisfaction probabilities
        polynomials[-2],        # Goal satisfaction probability
        polynomials[-1],        # Total probability, expected to equal one
        np.asarray(truth_vectors, dtype=bool).T
    )


def audit_templates(candidates, truth):
    audit=[]
    for i,c in enumerate(candidates):
        equal=[]; weaker=[]
        for j,other in enumerate(candidates):
            if i==j: continue
            if np.array_equal(truth[i],truth[j]): equal.append(other.name)
            elif np.all(~truth[i] | truth[j]): weaker.append(other.name)
        audit.append(dict(name=c.name,tautology=bool(truth[i].all()),
            unsatisfiable=bool(not truth[i].any()),equivalent_to='; '.join(equal),
            implies_weaker_candidate='; '.join(weaker)))
    return audit


# ============================================================
# 5) IPOPT problem with exact gradients, Jacobian, and Hessian
# ============================================================
class PolicyNLP:
    def __init__(self, features, polynomial=None, cutoff=None):
        self.features=features
        self.polynomial=polynomial
        self.cutoff=cutoff
        self.m=1 if cutoff is None else 2

    def loglik(self,p):
        return float(np.log(self.features @ p).sum())

    def ll_gradient(self,p):
        return (self.features/(self.features @ p)[:,None]).sum(axis=0)

    def ll_hessian(self,p):
        scaled=self.features/(self.features @ p)[:,None]
        return -(scaled.T @ scaled)

    def objective(self,p):
        return -self.loglik(p) if self.polynomial is None else self.polynomial.value(p)

    def gradient(self,p):
        return -self.ll_gradient(p) if self.polynomial is None else self.polynomial.gradient(p)

    def constraints(self,p):
        return np.array([p.sum()]) if self.cutoff is None else np.array([p.sum(),self.loglik(p)-self.cutoff])

    def jacobian(self,p):
        return np.ones(3) if self.cutoff is None else np.concatenate((np.ones(3),self.ll_gradient(p)))

    def jacobianstructure(self):
        return np.repeat(np.arange(self.m),3),np.tile(np.arange(3),self.m)

    def hessianstructure(self):
        return np.tril_indices(3)

    def hessian(self,p,lagrange,obj_factor):
        h=-self.ll_hessian(p) if self.polynomial is None else self.polynomial.hessian(p)
        h=obj_factor*h
        if self.cutoff is not None: h=h+lagrange[1]*self.ll_hessian(p)
        return h[np.tril_indices(3)]


def choose_backend(request):
    if request not in ('auto','cyipopt','casadi'): raise ValueError('Unknown IPOPT interface')
    if request in ('auto','cyipopt'):
        try:
            import cyipopt
            return 'cyipopt'
        except ImportError:
            if request=='cyipopt': raise ImportError('Install with: conda install -c conda-forge cyipopt')
    try:
        import casadi
        return 'casadi'
    except ImportError:
        raise ImportError('Install cyipopt (conda-forge), or run: pip install casadi. Both interfaces use IPOPT.')


class IpoptSolver:
    def __init__(self,nlp,cfg,backend):
        self.nlp=nlp; self.backend=backend
        self.lower=np.zeros(3); self.upper=np.ones(3)
        cl=np.array([1.] if nlp.cutoff is None else [1.,0.])
        cu=np.array([1.] if nlp.cutoff is None else [1.,1e19])
        self.cl,self.cu=cl,cu
        if backend=='cyipopt':
            import cyipopt
            self.problem=cyipopt.Problem(n=3,m=nlp.m,problem_obj=nlp,lb=self.lower,ub=self.upper,cl=cl,cu=cu)
            for key,value in {'tol':1e-9,'constr_viol_tol':1e-9,'max_iter':cfg.max_iter,
                    'print_level':cfg.print_level,'mu_strategy':'adaptive','bound_relax_factor':0.,'sb':'yes'}.items():
                self.problem.add_option(key,value)
        else:
            import casadi as ca
            p=ca.SX.sym('p',3)
            ll=ca.sum1(ca.log(ca.DM(nlp.features) @ p))
            if nlp.polynomial is None: obj=-ll
            else:
                obj=0
                for coefficient,e in zip(nlp.polynomial.coefficients,nlp.polynomial.exponents):
                    term=float(coefficient)
                    for j in range(3): term*=p[j]**int(e[j])
                    obj+=term
            g=ca.sum1(p) if nlp.cutoff is None else ca.vertcat(ca.sum1(p),ll-nlp.cutoff)
            self.problem=ca.nlpsol('activity_solver','ipopt',{'x':p,'f':obj,'g':g},
                {'print_time':False,'ipopt.print_level':cfg.print_level,'ipopt.sb':'yes',
                 'ipopt.tol':1e-9,'ipopt.constr_viol_tol':1e-9,'ipopt.max_iter':cfg.max_iter,
                 'ipopt.mu_strategy':'adaptive','ipopt.bound_relax_factor':0.})

    def solve(self,start):
        if self.backend=='cyipopt':
            p,info=self.problem.solve(np.asarray(start,dtype=float))
            return np.asarray(p),int(info['status']) in (0,1),str(info.get('status_msg',info['status']))
        result=self.problem(x0=start,lbx=self.lower,ubx=self.upper,lbg=self.cl,ubg=self.cu)
        status=self.problem.stats()
        return np.array(result['x']).ravel(),bool(status['success']),status['return_status']


def clean_simplex(p):
    p=np.maximum(np.asarray(p,dtype=float),0.)
    if not np.isfinite(p).all() or p.sum()==0: raise ValueError('Invalid optimizer result')
    return p/p.sum()


def estimate_policy(features,cfg,backend):
    nlp=PolicyNLP(features)
    solver=IpoptSolver(nlp,cfg,backend)
    p,success,status=solver.solve(np.full(3,1/3))
    if not success: raise RuntimeError('MLE IPOPT failed: '+status)
    return clean_simplex(p),nlp


def simplex_grid(resolution):
    return np.array([(i/resolution,j/resolution,(resolution-i-j)/resolution)
                     for i in range(resolution+1) for j in range(resolution+1-i)])


def feasible_segment(p,mle,nlp,cutoff):
    """Repair tiny constraint violations along a line toward feasible MLE."""
    p=clean_simplex(p)
    if nlp.loglik(p)>=cutoff: return p
    lo,hi=0.,1.
    for _ in range(55):
        t=(lo+hi)/2
        if nlp.loglik((1-t)*p+t*mle)>=cutoff: hi=t
        else: lo=t
    return (1-hi)*p+hi*mle


def worst_case_ipopt(polynomial,features,mle,cfg,backend,grid):
    base=PolicyNLP(features)
    cutoff=base.loglik(mle)-cfg.delta
    if cfg.delta==0:
        if np.linalg.matrix_rank(features[:,:2]-features[:,2,None])<2:
            raise ValueError('delta=0 shortcut needs identifiable mixture weights')
        return dict(value=polynomial.value(mle),p=mle.copy(),likelihood_drop=0.,
            successful_starts=0,grid_best=polynomial.value(mle),status_counts={'delta_zero':1})
    nlp=PolicyNLP(features,polynomial,cutoff)
    solver=IpoptSolver(nlp,cfg,backend)
    feasible=np.array([p for p in grid if base.loglik(p)>=cutoff])
    points=np.vstack((mle,feasible)) if len(feasible) else mle[None,:]
    scores=np.array([polynomial.value(p) for p in points])
    order=np.argsort(scores)
    # Include MLE, promising grid starts, and starts spread across the feasible set.
    indices=[0]+list(order[:cfg.restarts//2])+list(np.linspace(0,len(points)-1,cfg.restarts-cfg.restarts//2,dtype=int))
    indices=list(dict.fromkeys(indices))
    solutions=[]; statuses=Counter()
    for i in indices:
        try:
            p,success,status=solver.solve(points[i])
            statuses[status]+=1
            if not success: continue
            if abs(p.sum()-1)>1e-6 or np.min(p)<-1e-6: continue
            p=clean_simplex(p)
            if base.loglik(p)<cutoff-1e-6: continue
            p=feasible_segment(p,mle,base,cutoff)
            if base.loglik(p)<cutoff-1e-8: continue
            solutions.append((polynomial.value(p),p))
        except (RuntimeError,ValueError,FloatingPointError) as exc:
            statuses[type(exc).__name__]+=1
    if not solutions:
        raise RuntimeError('No converged, feasible IPOPT solution: '+str(statuses))
    value,p=min(solutions,key=lambda x:x[0])
    grid_best=float(scores.min())
    if value>grid_best+1e-6:
        raise RuntimeError('IPOPT minimum is worse than feasible grid: increase restarts')
    if value>polynomial.value(mle)+1e-6:
        raise RuntimeError('Worst-case estimate exceeds nominal value')
    return dict(value=value,p=p,likelihood_drop=base.loglik(mle)-base.loglik(p),
        successful_starts=len(solutions),grid_best=grid_best,status_counts=dict(statuses))


# ============================================================
# 6) Rankings, demonstration export, and fair timing
# ============================================================
def csv_write(path,rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(cfg=SETTINGS,save=True,verbose=True):
    if cfg.days<1 or cfg.goal<1 or cfg.delta<0 or cfg.restarts<2 or cfg.grid_resolution<2 or cfg.max_operators < 1 or cfg.max_depth < 1:
        raise ValueError('Invalid experiment settings')
    true=np.asarray(cfg.true_weights,dtype=float)
    demo=np.asarray(cfg.demo_weights,dtype=float)
    if true.shape!=(3,) or np.any(true<0) or not np.isclose(true.sum(),1):
        raise ValueError('true_weights must be three nonnegative values summing to one')
    if demo.shape!=(3,) or np.any(demo<0) or not np.isclose(demo.sum(),1):
        raise ValueError('demo_weights must be three nonnegative values summing to one')
    backend=choose_backend(cfg.solver)
    start=time.perf_counter()
    demonstrations=generate_demonstrations(cfg)
    features=likelihood_features(demonstrations,cfg)
    t=time.perf_counter(); mle,ll=estimate_policy(features,cfg,backend); mle_time=time.perf_counter()-t
    dag,candidates,goal_root=generate_templates(cfg.max_operators,cfg.max_depth)
    t=time.perf_counter()
    polynomials,goal_poly,total_poly,truth=compile_probabilities(cfg,dag,candidates,goal_root)
    compile_time=time.perf_counter()-t
    for p in [true,mle,*np.eye(3)]:
        if not np.isclose(total_poly.value(p),1,atol=1e-10): raise RuntimeError('Invalid total mass')
    audit=audit_templates(candidates,truth)
    if any(r['tautology'] or r['unsatisfiable'] for r in audit):
        raise ValueError('A template is trivial under these settings; inspect audit or change the fixed candidate class')
    # Match the urban-driving timing scope: score all candidates and
    # select one maximizer, excluding shared preprocessing and reporting.
    t = time.perf_counter()
    nominal = np.array([poly.value(mle) for poly in polynomials])
    nominal_winner = int(np.argmax(nominal))
    best_nom = float(nominal[nominal_winner])
    nominal_time = time.perf_counter() - t

    # Include all preparation specific to the worst-case search.
    t = time.perf_counter()
    grid = simplex_grid(cfg.grid_resolution)
    wc_results = []
    wc_durations = []
    for poly in polynomials:
        formula_start = time.perf_counter()
        result = worst_case_ipopt(poly, features, mle, cfg, backend, grid)
        wc_durations.append(time.perf_counter() - formula_start)
        wc_results.append(result)
    worst_case = np.array([result['value'] for result in wc_results])
    robust_winner = int(np.argmax(worst_case))
    best_rob = float(worst_case[robust_winner])
    worst_case_time = time.perf_counter() - t

    # Diagnostic evaluations and report construction are outside both timers.
    # ipopt_seconds retains the per-call duration (including setup/checks);
    # use worst_case_selection for the complete method duration.
    rows = []
    for i, (candidate, poly, result, duration) in enumerate(
        zip(candidates, polynomials, wc_results, wc_durations)
    ):
        rows.append(dict(name=candidate.name,formula=dag.render(candidate.root),
            meaning=candidate.description,operator_count=formula_complexity(dag,candidate.root)[0],
            syntax_depth=formula_complexity(dag,candidate.root)[1],nominal=float(nominal[i]),worst_case_ipopt=result['value'],
            gap=float(nominal[i]-result['value']),
            demo_policy_probability=poly.value(demo),
            true_policy_probability=poly.value(true),
            worst_p1=float(result['p'][0]),worst_p2=float(result['p'][1]),worst_p3=float(result['p'][2]),
            likelihood_drop=float(result['likelihood_drop']),ipopt_seconds=duration,
            successful_starts=result['successful_starts'],feasible_grid_minimum=result['grid_best'],
            solver_status_counts=json.dumps(result['status_counts'])))
    winners=lambda key,value:[r['name'] for r in rows if abs(r[key]-value)<1e-8]
    summary=dict(settings=asdict(cfg),backend=backend,mixture_type='trajectory_level_latent_mode',
        mle=mle.tolist(),
        true_weights=true.tolist(),demo_weights=demo.tolist(),
        biased_behavior='morning-oriented' if np.argmax(demo)==0 else ('distributed' if np.argmax(demo)==1 else 'goal-responsive'),
        demonstration_days=cfg.days,demonstration_decisions=4*cfg.days,
        syntax_dag_nodes=len(dag.nodes),candidate_count=len(candidates),
        nominal_winners=winners('nominal',best_nom),robust_winners=winners('worst_case_ipopt',best_rob),
        true_policy_in_set=bool(ll.loglik(true)>=ll.loglik(mle)-cfg.delta-1e-8),
        true_policy_likelihood_drop=ll.loglik(mle)-ll.loglik(true),
        goal_only_diagnostic=dict(nominal=goal_poly.value(mle),
                                  demo_policy=goal_poly.value(demo),
                                  true_policy=goal_poly.value(true)),
        times=dict(mle=mle_time,common_probability_compilation=compile_time,
            nominal_selection=nominal_time,worst_case_selection=worst_case_time,
            total=time.perf_counter()-start),
        interpretation='IPOPT multistart local minimum estimates; no certified global guarantee or hidden-intent recovery claim.')
    if save:
        out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
        csv_write(out/'rankings.csv',sorted(rows,key=lambda r:r['worst_case_ipopt'],reverse=True))
        csv_write(out/'demonstrations.csv',[r for d in demonstrations for r in d])
        csv_write(out/'template_audit.csv',audit)
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if verbose:
        print('=== Human physical activity: trajectory-mixture LTLf selection with IPOPT ===')
        print('IPOPT interface:',backend,'| Days:',cfg.days,'| Decisions:',4*cfg.days,'| Delta:',cfg.delta)
        print('Reference/true weights:',true)
        print('Biased demonstration weights:',demo,
              '| over-represented behavior:',summary['biased_behavior'])
        print('MLE estimated from biased demonstrations:',np.round(mle,6))
        print('True policy inside uncertainty set:',summary['true_policy_in_set'])
        print('Templates:',len(candidates),'| Shared DAG nodes:',len(dag.nodes))
        print('Maximum operators:',cfg.max_operators,'| Maximum syntax depth:',cfg.max_depth)
        print('\nAll candidate formulas:')
        for c in candidates: print(' ',c.name,':',dag.render(c.root))
        for key,title in [('nominal','NOMINAL RANKING'),('worst_case_ipopt','IPOPT WORST-CASE RANKING')]:
            print('\n===',title,'===')
            for rank,r in enumerate(sorted(rows,key=lambda r:r[key],reverse=True),1):
                print(f"{rank:2d}. {r['name']:42s} {r[key]:.6f}")
        print('\n=== Selected specifications ===')
        print('Nominal:',summary['nominal_winners'])
        print('Robust :',summary['robust_winners'])
        for r in rows:
            if r['name'] in summary['nominal_winners']+summary['robust_winners']:
                print(f"{r['name']}: nominal={r['nominal']:.6f}, WC={r['worst_case_ipopt']:.6f}, "
                      f"demo={r['demo_policy_probability']:.6f}, true={r['true_policy_probability']:.6f}")
        print('\nTiming (seconds):', {k:round(v,6) for k,v in summary['times'].items()})
        #print(summary['interpretation'])
        print('\n=== Execution time summary ===')
        print(f"Nominal method    : {nominal_time:.6f} seconds")
        print(f"Worst-case method : {worst_case_time:.6f} seconds")
        print(f"Number of formulas: {len(candidates)}")
    return dict(summary=summary,rankings=rows,template_audit=audit,demonstrations=demonstrations)


def sensitivity_study(base=SETTINGS,seeds=range(10),deltas=(0.,1.,2.,4.),days=(20,50),output='sensitivity.csv'):
    """Predeclare seeds/settings; paired samples for each (seed,day count).

    Both methods always use the same fixed templates and demonstrations.
    Reports every tie rather than choosing a favorable winner.
    """
    rows=[]
    for n,seed,delta in product(days,seeds,deltas):
        result=run(replace(base,days=n,seed=seed,delta=delta),save=False,verbose=False)
        by_name={r['name']:r for r in result['rankings']}
        for method,key in [('nominal','nominal_winners'),('robust','robust_winners')]:
            for name in result['summary'][key]:
                r=by_name[name]
                rows.append(dict(days=n,seed=seed,delta=delta,method=method,formula=name,
                    nominal=r['nominal'],worst_case=r['worst_case_ipopt'],true_probability=r['true_policy_probability'],
                    true_policy_in_set=result['summary']['true_policy_in_set']))
    csv_write(output,rows)
    return rows



if __name__=='__main__':
    results=run()
