"""
Student Exam Performance Prediction using Bayesian Networks

Techniques: Bayesian Network, Parameter Estimation (MLE), Variable Elimination
(exact inference), Likelihood Weighting (approximate inference), MAP inference.

Network structure:
    Study, Assign -> Prep
    Prep, Prev, Attend -> Perf
    Perf -> Result

The dataset is synthetic. Replace generate_dataset() to use real student data.
"""

import csv
import itertools
import random
from functools import reduce
from pathlib import Path

DOMAINS = {
    "Study":  ["Low", "High"],
    "Assign": ["Incomplete", "Complete"],
    "Prev":   ["Low", "Good"],
    "Attend": ["Low", "High"],
    "Prep":   ["Poor", "Good"],
    "Perf":   ["Weak", "Strong"],
    "Result": ["Fail", "Pass"],
}

PARENTS = {
    "Study":  [],
    "Assign": [],
    "Prev":   [],
    "Attend": [],
    "Prep":   ["Study", "Assign"],
    "Perf":   ["Prep", "Prev", "Attend"],
    "Result": ["Perf"],
}

# Topological order of the network
ORDER = ["Study", "Assign", "Prev", "Attend", "Prep", "Perf", "Result"]

INPUT_FIELDS = [
    ("Study", "Study hours"),
    ("Assign", "Assignment completion"),
    ("Prev", "Previous marks"),
    ("Attend", "Attendance"),
]

DATASET_PATH = Path(__file__).resolve().parent / "student_dataset.csv"


def generate_dataset(n=1000, seed=42):
    """Create a synthetic student dataset from assumed cause-effect rules."""
    rng = random.Random(seed)
    prep_prob = {
        ("High", "Complete"): 0.90, ("High", "Incomplete"): 0.60,
        ("Low", "Complete"): 0.45, ("Low", "Incomplete"): 0.10,
    }
    data = []
    for _ in range(n):
        study = "High" if rng.random() < 0.50 else "Low"
        assign = "Complete" if rng.random() < 0.60 else "Incomplete"
        prev = "Good" if rng.random() < 0.50 else "Low"
        attend = "High" if rng.random() < 0.60 else "Low"

        prep = "Good" if rng.random() < prep_prob[(study, assign)] else "Poor"

        p_strong = (0.05 + 0.45 * (prep == "Good")
                    + 0.25 * (prev == "Good") + 0.20 * (attend == "High"))
        perf = "Strong" if rng.random() < p_strong else "Weak"

        p_pass = 0.95 if perf == "Strong" else 0.12
        result = "Pass" if rng.random() < p_pass else "Fail"

        data.append({"Study": study, "Assign": assign, "Prev": prev,
                     "Attend": attend, "Prep": prep, "Perf": perf,
                     "Result": result})
    return data


def save_dataset(data, path=DATASET_PATH):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ORDER)
        writer.writeheader()
        writer.writerows(data)


class Factor:
    """A table of values over a set of discrete variables."""

    def __init__(self, variables, table):
        self.vars = tuple(variables)
        self.table = table

    def restrict(self, var, value):
        """Apply evidence var=value and drop var from the factor."""
        if var not in self.vars:
            return self
        i = self.vars.index(var)
        table = {k[:i] + k[i + 1:]: v for k, v in self.table.items() if k[i] == value}
        return Factor(self.vars[:i] + self.vars[i + 1:], table)

    def sum_out(self, var):
        """Marginalise var out of the factor."""
        i = self.vars.index(var)
        table = {}
        for k, v in self.table.items():
            key = k[:i] + k[i + 1:]
            table[key] = table.get(key, 0.0) + v
        return Factor(self.vars[:i] + self.vars[i + 1:], table)

    def normalize(self):
        total = sum(self.table.values())
        return Factor(self.vars, {k: v / total for k, v in self.table.items()})


def multiply(f, g):
    variables = f.vars + tuple(v for v in g.vars if v not in f.vars)
    table = {}
    for values in itertools.product(*(DOMAINS[v] for v in variables)):
        a = dict(zip(variables, values))
        table[values] = (f.table[tuple(a[v] for v in f.vars)]
                         * g.table[tuple(a[v] for v in g.vars)])
    return Factor(variables, table)


class BayesianNetwork:
    def __init__(self):
        self.cpt = {}  # variable -> {parent values: {value: probability}}

    def fit(self, data, alpha=1.0):
        """Learn CPTs by maximum likelihood with Laplace smoothing."""
        for var in ORDER:
            parents = PARENTS[var]
            counts = {pv: {val: 0 for val in DOMAINS[var]}
                      for pv in itertools.product(*(DOMAINS[p] for p in parents))}
            for row in data:
                counts[tuple(row[p] for p in parents)][row[var]] += 1
            self.cpt[var] = {}
            for pv, c in counts.items():
                total = sum(c.values()) + alpha * len(DOMAINS[var])
                self.cpt[var][pv] = {v: (c[v] + alpha) / total for v in DOMAINS[var]}

    def _as_factor(self, var):
        variables = (var,) + tuple(PARENTS[var])
        table = {(val,) + pv: p
                 for pv, dist in self.cpt[var].items()
                 for val, p in dist.items()}
        return Factor(variables, table)

    def variable_elimination(self, query_vars, evidence):
        """Exact inference: returns the factor P(query_vars | evidence)."""
        factors = [self._as_factor(v) for v in ORDER]
        for var, val in evidence.items():
            factors = [f.restrict(var, val) for f in factors]

        hidden = [v for v in ORDER if v not in query_vars and v not in evidence]
        while hidden:
            # Eliminate the variable that produces the smallest factor
            def scope_size(v):
                return len({u for f in factors if v in f.vars for u in f.vars})

            var = min(hidden, key=scope_size)
            hidden.remove(var)
            related = [f for f in factors if var in f.vars]
            factors = [f for f in factors if var not in f.vars]
            if related:
                factors.append(reduce(multiply, related).sum_out(var))
        return reduce(multiply, factors).normalize()

    def query(self, var, evidence):
        factor = self.variable_elimination([var], evidence)
        return {k[0]: p for k, p in factor.table.items()}

    def map_query(self, query_vars, evidence):
        """Most probable assignment of query_vars given the evidence."""
        factor = self.variable_elimination(query_vars, evidence)
        best = max(factor.table, key=factor.table.get)
        return dict(zip(factor.vars, best)), factor.table[best]

    def likelihood_weighting(self, var, evidence, n_samples=20000, seed=1):
        """Approximate inference by weighted forward sampling."""
        rng = random.Random(seed)
        weights = {val: 0.0 for val in DOMAINS[var]}
        for _ in range(n_samples):
            sample, weight = {}, 1.0
            for v in ORDER:
                dist = self.cpt[v][tuple(sample[p] for p in PARENTS[v])]
                if v in evidence:
                    sample[v] = evidence[v]
                    weight *= dist[evidence[v]]
                else:
                    sample[v] = rng.choices(list(dist), weights=list(dist.values()))[0]
            weights[sample[var]] += weight
        total = sum(weights.values())
        return {k: w / total for k, w in weights.items()}


def risk_label(p_pass):
    if p_pass >= 0.75:
        return "LIKELY TO PASS"
    if p_pass >= 0.45:
        return "BORDERLINE - NEEDS ATTENTION"
    return "HIGH RISK OF FAILURE"


def report(bn, evidence, title):
    exact = bn.query("Result", evidence)
    approx = bn.likelihood_weighting("Result", evidence)
    prep, prep_p = bn.map_query(["Prep"], evidence)

    print(f"\n--- {title} ---")
    print("Evidence:", evidence)
    print(f"Variable Elimination (exact):    Pass {exact['Pass']:.1%} | Fail {exact['Fail']:.1%}")
    print(f"Likelihood Weighting (approx.):  Pass {approx['Pass']:.1%} | Fail {approx['Fail']:.1%}")
    print(f"Prediction: {risk_label(exact['Pass'])}")
    print(f"MAP exam preparation: {prep['Prep']} ({prep_p:.1%})")


def read_student():
    """Read a student's details from the user; blank input means unknown."""
    print("\nEnter student details (press Enter to leave a factor unknown)")
    evidence = {}
    for var, label in INPUT_FIELDS:
        answer = input(f"{label} {DOMAINS[var]}: ").strip().capitalize()
        if answer in DOMAINS[var]:
            evidence[var] = answer
        elif answer:
            print("  Not recognised, treated as unknown.")
    return evidence


def main():
    data = generate_dataset()
    save_dataset(data)
    print(f"Dataset: {len(data)} students saved to {DATASET_PATH}")

    bn = BayesianNetwork()
    bn.fit(data)

    print("\nLearned P(Result | Perf):")
    for (perf,), dist in bn.cpt["Result"].items():
        print(f"  Perf={perf:<6} -> " + ", ".join(f"{k}: {v:.3f}" for k, v in dist.items()))

    report(bn, {"Study": "Low", "Attend": "Low", "Prev": "Low",
                "Assign": "Incomplete"}, "Weak student")
    report(bn, {"Study": "High", "Attend": "High", "Prev": "Good",
                "Assign": "Complete"}, "Strong student")
    report(bn, {"Study": "High", "Attend": "Low"}, "Partial evidence")

    cause, prob = bn.map_query(["Study", "Assign"], {"Result": "Fail"})
    print(f"\nMost likely cause of a Fail (MAP): {cause} ({prob:.1%})")

    while True:
        report(bn, read_student(), "Your student")
        if input("\nTry another student? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()
