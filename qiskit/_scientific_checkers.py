# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

"""Opt-in scientific runtime checks used by the SciBench pilot.

The module is inert unless ``SCIBENCH_TRIGGER_LOG`` names a log file. A hook is the
:func:`observe` decorator on a public function: it calls the original function, returns
its result unchanged and, only when logging is enabled, hands the result and the bound call
arguments to a checker. Checkers append stable IDs to the log, never raise into Qiskit,
never change a return value and never touch the global random state. Checkers that
re-call the API work on copies and run with checking suppressed (re-entrancy guard).

Tolerance convention: ``_C * eps(dtype) * size * scale`` with each factor justified at the
call site; ``sqrt(eps)`` is used where the checked quantity passes through a square root
of an eigenvalue that may be rounding-level small.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import os
import warnings

import numpy as np

_ACTIVE = False
_EPS = float(np.finfo(np.float64).eps)
_SQRT_EPS = float(np.sqrt(_EPS))
_C = 64.0
_ENV = "SCIBENCH_TRIGGER_LOG"
_MAX_DIM = 256  # largest Hilbert-space dimension a re-calling checker will handle
_MAX_QUBITS = 8
_SYN_TOL = 1.0e-7  # synthesis / transpilation equivalence (see QK-SYN-*, QK-TRN-001)
_WEYL_FID = 1.0e-9  # default specialisation infidelity of the Weyl decomposition
SWALLOWED = []
REACHED = {}


def enabled():
    return bool(os.environ.get(_ENV)) and not _ACTIVE


def trigger(checker_id):
    path = os.environ.get(_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"checker_id": checker_id}) + "\n")
    except Exception:  # pylint: disable=broad-exception-caught
        _swallowed()


def trigger_if(condition, checker_id):
    _reach(checker_id)
    if bool(condition):
        trigger(checker_id)


def _reach(checker_id):
    """Curator-only: count predicate evaluations (observation reachability)."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        REACHED[checker_id] = REACHED.get(checker_id, 0) + 1


def _swallowed():
    """Curator-only: record exceptions swallowed inside a checker."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        import traceback

        SWALLOWED.append(traceback.format_exc())


def _guarded(fn):
    """Run ``fn`` once, never nested, never raising into Qiskit."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _ACTIVE  # pylint: disable=global-statement
        if _ACTIVE or not os.environ.get(_ENV):
            return None
        _ACTIVE = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with np.errstate(all="ignore"), _quiet():
                    return fn(*args, **kwargs)
        except Exception:  # pylint: disable=broad-exception-caught
            _swallowed()
            return None
        finally:
            _ACTIVE = False

    return wrapper


@contextlib.contextmanager
def _quiet():
    import logging

    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def observe(*checkers):
    """Decorator: return the wrapped function's result unchanged, run ``checkers`` when enabled.

    Each checker is called as ``checker(result, arguments)`` where ``arguments`` maps the
    parameter names of the wrapped function to the values of the call (defaults applied).
    """

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if enabled():
                try:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    arguments = dict(bound.arguments)
                except Exception:  # pylint: disable=broad-exception-caught
                    arguments = None
                if arguments is not None:
                    for checker in checkers:
                        checker(result, arguments)
            return result

        return wrapper

    return deco


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _states():
    from qiskit.quantum_info import DensityMatrix, Statevector

    return Statevector, DensityMatrix


def _as_state(x):
    Statevector, DensityMatrix = _states()
    if isinstance(x, (Statevector, DensityMatrix)):
        return x
    arr = np.asarray(x, dtype=complex)
    if arr.ndim == 1:
        return Statevector(arr)
    if arr.ndim == 2:
        return DensityMatrix(arr)
    return None


def _norm_defect(state):
    Statevector, _ = _states()
    if isinstance(state, Statevector):
        return abs(float(np.vdot(state.data, state.data).real) - 1.0)
    return abs(float(np.trace(state.data).real) - 1.0)


def _usable(state, max_dim=_MAX_DIM):
    """A valid state, normalised to 1e-9, of bounded dimension."""
    if state is None or state.dim > max_dim:
        return False
    if _norm_defect(state) > 1e-9:
        return False
    return bool(state.is_valid())


def _matrix(state):
    Statevector, _ = _states()
    if isinstance(state, Statevector):
        return np.outer(state.data, state.data.conj())
    return np.asarray(state.data)


def _is_pure(state, tol=1e-14):
    Statevector, _ = _states()
    if isinstance(state, Statevector):
        return True
    return float(np.real(np.trace(state.data @ state.data))) >= 1.0 - tol


def _pure_vec(state):
    """Amplitude vector of a pure state, else None."""
    Statevector, _ = _states()
    if isinstance(state, Statevector):
        return np.asarray(state.data)
    if not _is_pure(state):
        return None
    w, v = np.linalg.eigh(np.asarray(state.data))
    return v[:, -1]


def _perm(d):
    return np.roll(np.arange(d), 1)


def _permuted(state):
    """The same state with its computational basis relabelled by a fixed permutation (exact)."""
    Statevector, DensityMatrix = _states()
    p = _perm(state.dim)
    if isinstance(state, Statevector):
        return Statevector(np.asarray(state.data)[p], dims=state.dims())
    return DensityMatrix(np.asarray(state.data)[np.ix_(p, p)], dims=state.dims())


def _num_qubits_of(state):
    return int(round(np.log2(state.dim))) if all(d == 2 for d in state.dims()) else None


def _unitary_defect(u):
    u = np.asarray(u)
    return float(np.abs(u.conj().T @ u - np.eye(u.shape[1])).max())


def _phase_error(u, v):
    """max |U - e^{i phi} V| with the best global phase phi."""
    t = np.trace(v.conj().T @ u)
    ph = t / abs(t) if abs(t) > 1e-14 else 1.0
    return float(np.abs(u - ph * v).max())


def _operator_data(obj):
    from qiskit.quantum_info import Operator

    return np.asarray(Operator(obj).data)


def _choi_data(obj):
    from qiskit.quantum_info import Choi

    return np.asarray(Choi(obj).data)


def _super_data(obj):
    from qiskit.quantum_info import SuperOp

    return np.asarray(SuperOp(obj).data)


def _test_state(d, seed=1234):
    """A fixed full-rank density matrix drawn with a private RNG (global state untouched)."""
    rng = np.random.RandomState(seed)
    g = rng.randn(d, d) + 1j * rng.randn(d, d)
    rho = g @ g.conj().T
    return rho / np.trace(rho).real


def _apply_super(s, rho):
    d_out = int(round(np.sqrt(s.shape[0])))
    return (s @ rho.ravel(order="F")).reshape(d_out, d_out, order="F")


# ---------------------------------------------------------------------------
# A. State measures
# ---------------------------------------------------------------------------


@_guarded
def check_state_fidelity(f, a):
    """QK-STM-001."""
    from qiskit.quantum_info import state_fidelity

    s1, s2 = _as_state(a["state1"]), _as_state(a["state2"])
    if not (_usable(s1) and _usable(s2)) or s1.dim != s2.dim:
        return
    d = s1.dim
    Statevector, _ = _states()
    both_dm = not (isinstance(s1, Statevector) or isinstance(s2, Statevector))
    # two DensityMatrix inputs always take the matrix-square-root route (whatever their rank),
    # which amplifies rounding-level eigenvalues to sqrt(eps)
    base = _C * (_SQRT_EPS * d if both_dm else _EPS * d * d)
    tol = base + 4 * (_norm_defect(s1) + _norm_defect(s2))
    f_swap = state_fidelity(s2, s1, validate=False)
    f_perm = state_fidelity(_permuted(s1), _permuted(s2), validate=False)
    trigger_if(
        abs(f - f_swap) > tol or abs(f - f_perm) > tol or f < -tol or f > 1 + tol,
        "QK-STM-001",
    )


@_guarded
def check_state_fidelity_monotone(f, a):
    """QK-STM-002."""
    from qiskit.quantum_info import partial_trace, state_fidelity

    s1, s2 = _as_state(a["state1"]), _as_state(a["state2"])
    if not (_usable(s1, 64) and _usable(s2, 64)) or s1.dim != s2.dim:
        return
    d = s1.dim
    Statevector, _ = _states()
    tol = _C * (_SQRT_EPS * d if not isinstance(s1, Statevector) else _EPS * d * d) + 4 * _norm_defect(s1)
    self_f = state_fidelity(s1, s1, validate=False)
    bad = abs(self_f - 1.0) > tol
    n = _num_qubits_of(s1)
    if n is not None and n >= 2 and s2.dims() == s1.dims():
        tol2 = _C * _SQRT_EPS * d + 4 * (_norm_defect(s1) + _norm_defect(s2))
        r1, r2 = partial_trace(s1, [0]), partial_trace(s2, [0])
        bad = bad or state_fidelity(r1, r2, validate=False) < f - tol2
    trigger_if(bad, "QK-STM-002")


def _entropy_tol(d, base):
    # eigenvalue errors ~ eps*d each; |dS| <= sum_i delta_i |log delta_i| ~ d * eps * d * 40
    return _C * _EPS * d * d * 40.0 / np.log(base)


@_guarded
def check_entropy(s, a):
    """QK-STM-003."""
    from qiskit.quantum_info import entropy

    st = _as_state(a["state"])
    base = a["base"]
    if not _usable(st) or base < 2:
        return
    d = st.dim
    tol = _entropy_tol(d, base) + 4 * _norm_defect(st)
    s2 = entropy(_permuted(st), base=base)
    bad = s < -tol or s > np.log(d) / np.log(base) + tol or abs(s - s2) > tol
    if float(np.real(np.trace(_matrix(st) @ _matrix(st)))) >= 1.0 - _EPS * d:
        bad = bad or s > tol
    trigger_if(bad, "QK-STM-003")


@_guarded
def check_mutual_information(mi, a):
    """QK-STM-004."""
    from qiskit.quantum_info import entropy, partial_trace

    st = _as_state(a["state"])
    base = a["base"]
    if not _usable(st, 64) or len(st.dims()) != 2 or base < 2:
        return
    da, db = st.dims()
    s_ab = entropy(st, base=base)
    s_a = entropy(partial_trace(st, [1]), base=base)
    s_b = entropy(partial_trace(st, [0]), base=base)
    tol = _entropy_tol(st.dim, base) + 4 * _norm_defect(st)
    top = 2 * min(np.log(da), np.log(db)) / np.log(base)
    bad = (
        mi < -tol
        or mi > top + tol
        or s_ab < abs(s_a - s_b) - tol
        or s_ab > s_a + s_b + tol
        or abs(mi - (s_a + s_b - s_ab)) > tol
    )
    trigger_if(bad, "QK-STM-004")


def _schmidt_two_qubit(vec):
    s = np.linalg.svd(np.asarray(vec).reshape(2, 2), compute_uv=False)
    return s


def _shannon2(p):
    p = np.asarray([x for x in p if x > 0])
    return float(-(p * np.log2(p)).sum())


@_guarded
def check_concurrence(c, a):
    """QK-STM-005."""
    st = _as_state(a["state"])
    if not _usable(st, 4) or st.dim != 4:
        return
    tol = _C * _SQRT_EPS + 4 * _norm_defect(st)
    bad = c < -tol or c > 1 + tol
    vec = _pure_vec(st)
    if vec is not None:
        s = _schmidt_two_qubit(vec)
        bad = bad or abs(c - 2 * s[0] * s[1]) > tol
    trigger_if(bad, "QK-STM-005")


@_guarded
def check_entanglement_of_formation(e, a):
    """QK-STM-006."""
    st = _as_state(a["state"])
    if not _usable(st, 4) or st.dim != 4:
        return
    tol = _C * _SQRT_EPS + 4 * _norm_defect(st)
    bad = e < -tol or e > 1 + tol
    vec = _pure_vec(st)
    if vec is not None:
        s = _schmidt_two_qubit(vec)
        bad = bad or abs(e - _shannon2([s[0] ** 2, s[1] ** 2])) > tol
    trigger_if(bad, "QK-STM-006")


@_guarded
def check_negativity(n_val, a):
    """QK-STM-007."""
    st = _as_state(a["state"])
    qargs = a["qargs"]
    if not _usable(st, 64):
        return
    n = _num_qubits_of(st)
    d = st.dim
    tol = _C * _EPS * d * d + 4 * _norm_defect(st)
    bad = n_val < -tol
    if n is not None and qargs is not None:
        vec = _pure_vec(st)
        if vec is not None:
            qs = sorted(int(q) for q in qargs)
            ax_a = [n - 1 - q for q in qs]
            ax_b = [i for i in range(n) if i not in ax_a]
            m = np.transpose(vec.reshape((2,) * n), ax_a + ax_b).reshape(2 ** len(qs), -1)
            s = np.linalg.svd(m, compute_uv=False)
            expected = (s.sum() ** 2 - 1.0) / 2.0
            bad = bad or abs(n_val - expected) > tol
    trigger_if(bad, "QK-STM-007")


@_guarded
def check_purity(p, a):
    """QK-STM-008."""
    from qiskit.quantum_info import partial_trace, purity

    st = _as_state(a["state"])
    if not _usable(st, 1024):
        return
    d = st.dim
    tol = _C * _EPS * d + 4 * _norm_defect(st)
    bad = p < 1.0 / d - tol or p > 1 + tol
    Statevector, _ = _states()
    n = _num_qubits_of(st)
    if isinstance(st, Statevector) and n is not None and n >= 2:
        pa = purity(partial_trace(st, [0]), validate=False)
        pb = purity(partial_trace(st, list(range(1, n))), validate=False)
        bad = bad or abs(pa - pb) > tol
    trigger_if(bad, "QK-STM-008")


# ---------------------------------------------------------------------------
# B. Process measures
# ---------------------------------------------------------------------------


def _unitary_or_none(obj):
    """Unitary matrix of ``obj``, else None (precondition not met)."""
    try:
        u = _operator_data(obj)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if u.shape[0] != u.shape[1] or _unitary_defect(u) > 1e-9:
        return None
    return u


@_guarded
def check_process_fidelity_routes(f, a):
    """QK-OPM-001."""
    from qiskit.quantum_info import Operator

    ch, tg = a["channel"], a["target"]
    sup = _super_data(ch)
    d = int(round(np.sqrt(sup.shape[0])))
    if d > 16 or sup.shape[0] != sup.shape[1]:
        return
    u = np.eye(d) if tg is None else _unitary_or_none(tg)
    if u is None or u.shape[0] != d:
        return
    # second route: overlap of the two Choi matrices, F = Tr[Lambda_E Lambda_U] / d^2
    lam_e = _choi_data(ch)
    lam_u = _choi_data(Operator(u))
    f2 = float(np.real(np.trace(lam_e @ lam_u))) / d**2
    tol = _C * _EPS * d**3 + 8 * _unitary_defect(u) * d
    trigger_if(abs(f - f2) > tol, "QK-OPM-001")


def _cptp_defects(chan_choi, d_in):
    d_out = chan_choi.shape[0] // d_in
    herm = (chan_choi + chan_choi.conj().T) / 2
    cp = max(0.0, -float(np.linalg.eigvalsh(herm).min()))
    tr_out = np.trace(chan_choi.reshape(d_in, d_out, d_in, d_out), axis1=1, axis2=3)
    tp = float(np.abs(tr_out - np.eye(d_in)).max())
    return cp, tp


@_guarded
def check_gate_error(e, a):
    """QK-OPM-002."""
    from qiskit.quantum_info import Operator

    ch, tg = a["channel"], a["target"]
    sup = _super_data(ch)
    d = int(round(np.sqrt(sup.shape[0])))
    if d > 16:
        return
    u = np.eye(d) if tg is None else _unitary_or_none(tg)
    if u is None or u.shape[0] != d:
        return
    s_u = _super_data(Operator(u))
    f_pro = float(np.real(np.trace(s_u.conj().T @ sup))) / d**2
    f_ave = (d * f_pro + 1.0) / (d + 1.0)
    tol = _C * _EPS * d**3 + 8 * _unitary_defect(u) * d
    bad = abs(e - (1.0 - f_ave)) > tol
    cp, tp = _cptp_defects(_choi_data(ch), d)
    if cp <= 1e-9 and tp <= 1e-9:
        bad = bad or e < -tol - 4 * (cp + tp) or e > d / (d + 1.0) + tol + 4 * (cp + tp)
    trigger_if(bad, "QK-OPM-002")


@_guarded
def check_process_fidelity_invariance(f, a):
    """QK-OPM-003."""
    from qiskit.quantum_info import Operator, process_fidelity

    ch, tg = a["channel"], a["target"]
    u = _unitary_or_none(ch)
    if u is None or u.shape[0] > 16:
        return
    d = u.shape[0]
    v = np.eye(d) if tg is None else _unitary_or_none(tg)
    if v is None or v.shape[0] != d:
        return
    p = _perm(d)
    w = np.eye(d)[p]
    f_conj = process_fidelity(Operator(w @ u @ w.T), Operator(w @ v @ w.T))
    f_self = process_fidelity(Operator(u), Operator(u))
    tol = _C * _EPS * d**2 + 8 * (_unitary_defect(u) + _unitary_defect(v))
    trigger_if(abs(f - f_conj) > tol or abs(f_self - 1.0) > tol, "QK-OPM-003")


# ---------------------------------------------------------------------------
# C. Quantum channels
# ---------------------------------------------------------------------------


def _pauli_basis(n):
    mats = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    basis = [np.eye(1, dtype=complex)]
    for _ in range(n):
        basis = [np.kron(b, m) for b in basis for m in mats]
    return basis


def _action(rep, data, d_in, d_out, rho):
    """Action of a channel stored in representation ``rep`` on ``rho`` (documented conventions)."""
    if rep == "Operator":
        u = np.asarray(data)
        return u @ rho @ u.conj().T
    if rep == "Kraus":
        left, right = data
        right = left if right is None else right
        return sum(np.asarray(a) @ rho @ np.asarray(b).conj().T for a, b in zip(left, right))
    if rep == "SuperOp":
        return (np.asarray(data) @ rho.ravel(order="F")).reshape(d_out, d_out, order="F")
    if rep == "Choi":
        lam = np.asarray(data).reshape(d_in, d_out, d_in, d_out)
        return np.einsum("ikml,im->kl", lam, rho)
    if rep in ("PTM", "Chi"):
        n = int(round(np.log2(d_in)))
        if d_in != d_out or 2**n != d_in:
            return None
        basis = _pauli_basis(n)
        mat = np.asarray(data)
        out = np.zeros((d_out, d_out), dtype=complex)
        if rep == "PTM":
            coef = np.array([np.trace(p @ rho) for p in basis])
            for i, p in enumerate(basis):
                out += (mat[i] @ coef) * p
            return out / d_in
        for i, pi in enumerate(basis):
            for j, pj in enumerate(basis):
                out += mat[i, j] * (pi @ rho @ pj)
        return out / d_in
    return None


def _source_isometry_defect(rep, data, d_in):
    """Trace-preservation defect of a Kraus/Operator source (None if not a plain CP source)."""
    if rep == "Operator":
        return _unitary_defect(data)
    if rep == "Kraus":
        left, right = data
        if right is not None:
            return None
        tot = sum(np.asarray(k).conj().T @ np.asarray(k) for k in left)
        return float(np.abs(tot - np.eye(d_in)).max())
    return None


@_guarded
def check_transform_cptp(res, a):
    """QK-CHN-001."""
    rep_in, rep_out, data = a["input_rep"], a["output_rep"], a["data"]
    d_in, d_out = a["input_dim"], a["output_dim"]
    if rep_out != "Choi" or rep_in == rep_out or d_in * d_out > 256:
        return
    src = _source_isometry_defect(rep_in, data, d_in)
    if src is None or src > 1e-9:
        return
    lam = np.asarray(res)
    cp, tp = _cptp_defects(lam, d_in)
    scale = 1.0 + float(np.abs(lam).max())
    tol = _C * _EPS * (d_in * d_out) ** 2 * scale + 4 * src * d_in
    trigger_if(cp > tol or tp > tol, "QK-CHN-001")


@_guarded
def check_transform_action(res, a):
    """QK-CHN-002."""
    rep_in, rep_out, data = a["input_rep"], a["output_rep"], a["data"]
    d_in, d_out = a["input_dim"], a["output_dim"]
    if rep_in == rep_out or d_in != d_out or d_in > 8:
        return
    rho = _test_state(d_in)
    src = _action(rep_in, data, d_in, d_out, rho)
    dst = _action(rep_out, res, d_in, d_out, rho)
    if src is None or dst is None:
        return
    scale = 1.0 + max(
        float(np.abs(np.asarray(x)).max()) for x in _flat(data) + _flat(res)
    )
    tol = _C * _EPS * d_in**3 * scale
    if rep_out in ("Kraus", "Operator") and rep_in != "Operator":
        # eigenvalues of the Choi matrix below the library tolerance (ATOL_DEFAULT = 1e-8)
        # are discarded when Kraus operators are extracted
        tol += d_in**2 * 1e-8
    trigger_if(float(np.abs(src - dst).max()) > tol, "QK-CHN-002")


def _flat(data):
    if isinstance(data, (tuple, list)):
        out = []
        for x in data:
            out += _flat(x)
        return out
    return [] if data is None else [np.asarray(data)]


def _adapt_to(checker, out_rep):
    """Adapt a ``_transform_rep`` checker to a ``_to_<rep>(rep, data, input_dim, output_dim)`` call."""

    def wrapper(res, a):
        return checker(res, {**a, "input_rep": a["rep"], "output_rep": out_rep})

    return wrapper


for _rep in ("Choi", "SuperOp", "Kraus", "Chi", "PTM", "Stinespring", "Operator"):
    globals()["check_to_%s_action" % _rep.lower()] = _adapt_to(check_transform_action, _rep)
check_to_choi_cptp = _adapt_to(check_transform_cptp, "Choi")


def _channel_square(obj):
    s = _super_data(obj)
    d = int(round(np.sqrt(s.shape[0])))
    return s, d


@_guarded
def check_channel_compose(res, a):
    """QK-CHN-003 (compose)."""
    me, other, qargs, front = a["self"], a["other"], a["qargs"], a["front"]
    if qargs is not None:
        return
    s_a, d = _channel_square(me)
    s_b, d_b = _channel_square(other)
    s_r, d_r = _channel_square(res)
    if not (d == d_b == d_r) or d > 8:
        return
    rho = _test_state(d)
    first, second = (s_b, s_a) if front else (s_a, s_b)
    expected = _apply_super(second, _apply_super(first, rho))
    got = _apply_super(s_r, rho)
    tol = _C * _EPS * d**3 * (1.0 + float(np.abs(s_a).max() * np.abs(s_b).max()))
    trigger_if(float(np.abs(expected - got).max()) > tol, "QK-CHN-003")


@_guarded
def check_channel_tensor(res, a, swap=False):
    """QK-CHN-003 (tensor / expand)."""
    me, other = a["self"], a["other"]
    s_a, d_a = _channel_square(me)
    s_b, d_b = _channel_square(other)
    s_r, d_r = _channel_square(res)
    if d_r != d_a * d_b or d_r > 16:
        return
    rho, sig = _test_state(d_a, 11), _test_state(d_b, 12)
    left, right = (s_b, s_a) if swap else (s_a, s_b)
    r_left, r_right = (sig, rho) if swap else (rho, sig)
    expected = np.kron(_apply_super(left, r_left), _apply_super(right, r_right))
    got = _apply_super(s_r, np.kron(r_left, r_right))
    tol = _C * _EPS * d_r**3 * (1.0 + float(np.abs(s_a).max() * np.abs(s_b).max()))
    trigger_if(float(np.abs(expected - got).max()) > tol, "QK-CHN-003")


def check_channel_expand(res, a):
    return check_channel_tensor(res, a, swap=True)


@_guarded
def check_channel_evolve_state(res, a):
    """QK-CHN-004."""
    from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel

    st, other = a["self"], a["other"]
    if not isinstance(other, QuantumChannel) or not _usable(st, 64):
        return
    lam = _choi_data(other)
    d_in = other.dim[0]
    cp, tp = _cptp_defects(lam, d_in)
    if cp > 1e-9 or tp > 1e-9:
        return
    out = np.asarray(res.data)
    d = out.shape[0]
    tol = _C * _EPS * d**2 * (1.0 + float(np.abs(lam).max())) + 4 * (cp + tp + _norm_defect(st)) * d
    herm = float(np.abs(out - out.conj().T).max())
    tr = abs(float(np.trace(out).real) - 1.0)
    lmin = float(np.linalg.eigvalsh((out + out.conj().T) / 2).min())
    trigger_if(herm > tol or tr > tol or lmin < -tol, "QK-CHN-004")


# ---------------------------------------------------------------------------
# D. States and operators
# ---------------------------------------------------------------------------


@_guarded
def check_evolve_unitary(res, a):
    """QK-STA-001."""
    st, other = a["self"], a["other"]
    if not _usable(st, 4096):
        return
    u = _unitary_or_none(other)
    if u is None:
        return
    ud = _unitary_defect(u)
    d = st.dim
    tol = _C * _EPS * d + 4 * (ud + _norm_defect(st)) * d
    Statevector, _ = _states()
    if isinstance(st, Statevector):
        bad = abs(float(np.vdot(res.data, res.data).real) - float(np.vdot(st.data, st.data).real)) > tol
    else:
        r, s = np.asarray(res.data), np.asarray(st.data)
        bad = (
            abs(np.trace(r).real - np.trace(s).real) > tol
            or abs(np.trace(r @ r).real - np.trace(s @ s).real) > tol
        )
    trigger_if(bad, "QK-STA-001")


@_guarded
def check_evolve_circuit(res, a):
    """QK-STA-002."""
    from qiskit.circuit import QuantumCircuit

    st, other, qargs = a["self"], a["other"], a["qargs"]
    n = _num_qubits_of(st)
    if qargs is not None or not isinstance(other, QuantumCircuit) or n is None or n > 6:
        return
    if other.num_qubits != n or not _usable(st, 64):
        return
    u = _operator_data(other)
    tol = _C * _EPS * st.dim * (1 + other.size()) + 4 * _norm_defect(st) * st.dim
    Statevector, _ = _states()
    if isinstance(st, Statevector):
        expected = u @ np.asarray(st.data)
    else:
        expected = u @ np.asarray(st.data) @ u.conj().T
    trigger_if(float(np.abs(np.asarray(res.data) - expected).max()) > tol, "QK-STA-002")


@_guarded
def check_probabilities(res, a):
    """QK-STA-003."""
    st, qargs, decimals = a["self"], a["qargs"], a["decimals"]
    n = _num_qubits_of(st)
    if decimals is not None or n is None or not _usable(st, 4096):
        return
    Statevector, _ = _states()
    if isinstance(st, Statevector):
        p = np.abs(np.asarray(st.data)) ** 2
    else:
        p = np.real(np.diag(np.asarray(st.data)))
    res = np.asarray(res)
    d = st.dim
    tol = _C * _EPS * d + 4 * _norm_defect(st)
    if qargs is None:
        expected = p
    else:
        qs = [int(q) for q in qargs]
        axes = [n - 1 - q for q in qs]
        sum_axes = tuple(ax for ax in range(n) if ax not in axes)
        m = p.reshape((2,) * n).sum(axis=sum_axes)
        kept = sorted(axes)
        target = [n - 1 - q for q in reversed(qs)]
        expected = np.transpose(m, [kept.index(ax) for ax in target]).ravel()
    bad = res.shape != expected.shape or float(np.abs(res - expected).max()) > tol
    bad = bad or abs(float(res.sum()) - 1.0) > tol
    trigger_if(bad, "QK-STA-003")


def _local_hermitians(n):
    rng = np.random.RandomState(7)
    out = []
    for _ in range(n):
        g = rng.randn(2, 2) + 1j * rng.randn(2, 2)
        out.append((g + g.conj().T) / 2)
    return out


@_guarded
def check_partial_trace(res, a):
    """QK-STA-004."""
    st, qargs = _as_state(a["state"]), a["qargs"]
    n = _num_qubits_of(st) if st is not None else None
    if n is None or not _usable(st, 1024):
        return
    traced = {int(q) for q in qargs}
    kept = [q for q in range(n) if q not in traced]
    if not kept or len(kept) == n:
        return
    ops = _local_hermitians(n)
    full = np.array([[1.0 + 0j]])
    for q in range(n - 1, -1, -1):
        full = np.kron(full, ops[q] if q in kept else np.eye(2))
    red_op = np.array([[1.0 + 0j]])
    for q in reversed(kept):
        red_op = np.kron(red_op, ops[q])
    rho = _matrix(st)
    r = np.asarray(res.data)
    d = st.dim
    scale = float(np.abs(ops[0]).max()) ** n + 1.0
    tol = _C * _EPS * d * d * scale + 4 * _norm_defect(st) * scale
    exp_full = np.trace(full @ rho)
    exp_red = np.trace(red_op @ r)
    herm = float(np.abs(r - r.conj().T).max())
    lmin = float(np.linalg.eigvalsh((r + r.conj().T) / 2).min())
    bad = abs(exp_full - exp_red) > tol or abs(np.trace(r).real - np.trace(rho).real) > tol
    bad = bad or herm > tol or lmin < -tol
    trigger_if(bad, "QK-STA-004")


def _dense_observable(oper, dim):
    if hasattr(oper, "to_matrix"):
        m = oper.to_matrix()
    else:
        m = _operator_data(oper)
    m = np.asarray(m)
    return m if m.shape == (dim, dim) else None


@_guarded
def check_expectation_value(res, a):
    """QK-STA-005."""
    st, oper, qargs = a["self"], a["oper"], a["qargs"]
    if qargs is not None or not _usable(st, 1024):
        return
    h = _dense_observable(oper, st.dim)
    if h is None:
        return
    hnorm = float(np.abs(h).sum(axis=1).max())
    hdef = float(np.abs(h - h.conj().T).max())
    if hdef > 1e-10 * (1.0 + hnorm):
        return
    d = st.dim
    tol = (_C * _EPS * d + 4 * _norm_defect(st) + 4 * hdef) * (1.0 + hnorm)
    Statevector, _ = _states()
    if isinstance(st, Statevector):
        exact = np.vdot(st.data, h @ np.asarray(st.data))
    else:
        exact = np.trace(h @ np.asarray(st.data))
    bad = abs(res - exact) > tol or abs(np.imag(res)) > tol
    if d <= 256:
        ev = np.linalg.eigvalsh((h + h.conj().T) / 2)
        bad = bad or np.real(res) < ev.min() - tol or np.real(res) > ev.max() + tol
    trigger_if(bad, "QK-STA-005")


@_guarded
def check_operator_algebra(res, a, kind="compose"):
    """QK-OPR-001."""
    from qiskit.quantum_info import Operator

    me, other = a["self"], a["other"]
    if isinstance(me, Operator) and me.dim[0] > 64:
        return
    x = np.asarray(me.data)
    y = _operator_data(other)
    if kind == "compose":
        if a["qargs"] is not None:
            return
        expected = x @ y if a["front"] else y @ x
    elif kind == "tensor":
        expected = np.kron(x, y)
    else:
        expected = np.kron(y, x)
    got = np.asarray(res.data)
    if got.shape != expected.shape or max(x.shape + y.shape) > 64:
        return
    inner = max(x.shape[1], y.shape[0], 1)
    tol = _C * _EPS * inner * float(np.abs(x).max()) * float(np.abs(y).max()) * 4
    trigger_if(float(np.abs(got - expected).max()) > tol, "QK-OPR-001")


def check_operator_compose(res, a):
    return check_operator_algebra(res, a, "compose")


def check_operator_tensor(res, a):
    return check_operator_algebra(res, a, "tensor")


def check_operator_expand(res, a):
    return check_operator_algebra(res, a, "expand")


@_guarded
def check_circuit_operator(res, a):
    """QK-OPR-002."""
    from qiskit.quantum_info import Operator

    qc = a["circuit"]
    if qc.num_qubits > _MAX_QUBITS or qc._layout is not None or a["layout"] is not None:
        return
    if a["final_layout"] is not None:
        return
    u = np.asarray(res.data)
    d = u.shape[0]
    tol = _C * _EPS * d * (1 + qc.size())
    bad = _unitary_defect(u) > tol
    inv = Operator.from_circuit(qc.inverse()).data
    bad = bad or float(np.abs(inv - u.conj().T).max()) > tol
    trigger_if(bad, "QK-OPR-002")


# ---------------------------------------------------------------------------
# E. Pauli and Clifford algebra
# ---------------------------------------------------------------------------


@_guarded
def check_clifford_symplectic(res, a):
    """QK-CLF-001."""
    s = np.asarray(res.symplectic_matrix).astype(int)
    n = s.shape[0] // 2
    omega = np.block(
        [[np.zeros((n, n), int), np.eye(n, dtype=int)], [np.eye(n, dtype=int), np.zeros((n, n), int)]]
    )
    trigger_if(((s @ omega @ s.T) % 2 != omega).any(), "QK-CLF-001")


@_guarded
def check_clifford_unitary_from_circuit(res, a):
    """QK-CLF-002 (from_circuit)."""
    from qiskit.quantum_info import Operator

    qc = a["circuit"]
    if getattr(qc, "num_qubits", 99) > 6:
        return
    u = _operator_data(qc)
    v = np.asarray(res.to_operator().data)
    d = u.shape[0]
    tol = _C * _EPS * d * (1 + getattr(qc, "size", lambda: 1)())
    trigger_if(_phase_error(u, v) > tol, "QK-CLF-002")
    del Operator


@_guarded
def check_clifford_unitary_compose(res, a):
    """QK-CLF-002 (compose)."""
    from qiskit.quantum_info import Operator

    me, other, qargs, front = a["self"], a["other"], a["qargs"], a["front"]
    if me.num_qubits > 6 or (qargs is not None and getattr(other, "qargs", None) is None and qargs):
        return
    if qargs is not None:
        return
    x = np.asarray(me.to_operator().data)
    y = _operator_data(other)
    if y.shape != x.shape:
        return
    expected = x @ y if front else y @ x
    got = np.asarray(res.to_operator().data)
    tol = _C * _EPS * x.shape[0] * 8
    trigger_if(_phase_error(expected, got) > tol, "QK-CLF-002")
    del Operator


@_guarded
def check_pauli_compose(res, a):
    """QK-PAU-001 (compose)."""
    me, other = a["self"], a["other"]
    if a["inplace"] or a["qargs"] is not None or me.num_qubits > 6:
        return
    x, y = me.to_matrix(), other.to_matrix()
    expected = x @ y if a["front"] else y @ x
    trigger_if(float(np.abs(res.to_matrix() - expected).max()) > 1e-12, "QK-PAU-001")


@_guarded
def check_pauli_commutes(res, a):
    """QK-PAU-001 (commutes)."""
    from qiskit.quantum_info import Pauli

    me, other = a["self"], a["other"]
    if a["qargs"] is not None or not isinstance(other, Pauli) or me.num_qubits > 6:
        return
    x, y = me.to_matrix(), other.to_matrix()
    expected = bool(np.allclose(x @ y, y @ x, atol=1e-12))
    trigger_if(bool(res) != expected, "QK-PAU-001")


def _spo_matrix(op):
    return np.asarray(op.to_matrix())


@_guarded
def check_spo_simplify(res, a):
    """QK-SPO-001 (simplify)."""
    me = a["self"]
    if me.num_qubits > 6 or me.size > 200:
        return
    atol = me.atol if a["atol"] is None else a["atol"]
    rtol = me.rtol if a["rtol"] is None else a["rtol"]
    coeffs = np.abs(me.coeffs)
    drop = len(coeffs) * (atol + rtol * float(coeffs.max(initial=0.0)))
    tol = drop + _C * _EPS * float(coeffs.sum()) * 4 + 1e-15
    trigger_if(float(np.abs(_spo_matrix(res) - _spo_matrix(me)).max()) > tol, "QK-SPO-001")


@_guarded
def check_spo_compose(res, a):
    """QK-SPO-001 (compose)."""
    from qiskit.quantum_info import SparsePauliOp

    me, other = a["self"], a["other"]
    if a["qargs"] is not None or not isinstance(other, SparsePauliOp):
        return
    if me.num_qubits > 6 or me.size * other.size > 400:
        return
    x, y = _spo_matrix(me), _spo_matrix(other)
    expected = x @ y if a["front"] else y @ x
    scale = float(np.abs(me.coeffs).sum() * np.abs(other.coeffs).sum())
    trigger_if(float(np.abs(_spo_matrix(res) - expected).max()) > _C * _EPS * scale * 4, "QK-SPO-001")


@_guarded
def check_spo_tensor(res, a):
    """QK-SPO-001 (tensor)."""
    me, other = a["self"], a["other"]
    if me.num_qubits + other.num_qubits > 6 or me.size * other.size > 400:
        return
    expected = np.kron(_spo_matrix(me), _spo_matrix(other))
    scale = float(np.abs(me.coeffs).sum() * np.abs(other.coeffs).sum())
    trigger_if(float(np.abs(_spo_matrix(res) - expected).max()) > _C * _EPS * scale * 4, "QK-SPO-001")


@_guarded
def check_spo_adjoint(res, a):
    """QK-SPO-001 (adjoint)."""
    me = a["self"]
    if me.num_qubits > 6 or me.size > 200:
        return
    expected = _spo_matrix(me).conj().T
    scale = float(np.abs(me.coeffs).sum())
    trigger_if(float(np.abs(_spo_matrix(res) - expected).max()) > _C * _EPS * scale * 4, "QK-SPO-001")


# ---------------------------------------------------------------------------
# F. Random objects
# ---------------------------------------------------------------------------


@_guarded
def check_random_unitary(res, a):
    """QK-RND-001 (unitary)."""
    u = np.asarray(res.data)
    d = u.shape[0]
    trigger_if(d <= 1024 and _unitary_defect(u) > _C * _EPS * d * d, "QK-RND-001")


@_guarded
def check_random_density(res, a):
    """QK-RND-001 (density matrix)."""
    r = np.asarray(res.data)
    d = r.shape[0]
    if d > 1024:
        return
    tol = _C * _EPS * d * d
    herm = float(np.abs(r - r.conj().T).max())
    lmin = float(np.linalg.eigvalsh((r + r.conj().T) / 2).min())
    trigger_if(herm > tol or abs(np.trace(r).real - 1) > tol or lmin < -tol, "QK-RND-001")


@_guarded
def check_random_statevector(res, a):
    """QK-RND-001 (statevector)."""
    v = np.asarray(res.data)
    trigger_if(abs(float(np.vdot(v, v).real) - 1) > _C * _EPS * len(v), "QK-RND-001")


@_guarded
def check_random_hermitian(res, a):
    """QK-RND-001 (hermitian)."""
    h = np.asarray(res.data)
    d = h.shape[0]
    tol = _C * _EPS * d * (1.0 + float(np.abs(h).max()))
    bad = float(np.abs(h - h.conj().T).max()) > tol
    if a.get("traceless"):
        bad = bad or abs(np.trace(h)) > tol * d
    trigger_if(bad, "QK-RND-001")


@_guarded
def check_random_channel(res, a):
    """QK-RND-001 (channel)."""
    lam = _choi_data(res)
    d = res.dim[0]
    if d > 8:
        return
    cp, tp = _cptp_defects(lam, d)
    tol = _C * _EPS * d**3
    trigger_if(cp > tol or tp > tol, "QK-RND-001")


# ---------------------------------------------------------------------------
# G. Synthesis and compilation
# ---------------------------------------------------------------------------


@_guarded
def check_one_qubit_synthesis(res, a):
    """QK-SYN-001."""
    from qiskit.circuit import QuantumCircuit

    if getattr(a["self"], "use_dag", False) or not isinstance(res, QuantumCircuit):
        return
    u = a["unitary"]
    u = u.to_operator().data if hasattr(u, "to_operator") else (
        u.to_matrix() if hasattr(u, "to_matrix") else u
    )
    u = np.asarray(u, dtype=complex)
    if u.shape != (2, 2):
        return
    atol = a["atol"]
    tol = _C * _EPS + (8 * atol if a["simplify"] else 0.0) + 4 * _unitary_defect(u)
    v = _operator_data(res)
    trigger_if(_phase_error(u, v) > tol, "QK-SYN-001")


@_guarded
def check_weyl(res, a):
    """QK-SYN-002."""
    me = a["self"]
    u = np.asarray(me.unitary_matrix)
    defect = _unitary_defect(u)
    if defect > 1e-9:
        return
    fid = me.requested_fidelity
    tol = _C * _EPS + 8 * float(np.sqrt(max(1.0 - fid, 0.0)) if fid is not None else 0.0) + 4 * defect
    v = _operator_data(me.circuit())
    chamber = 1e-12
    bad = float(np.abs(u - v).max()) > tol
    bad = bad or me.a > np.pi / 4 + chamber or me.a < me.b - chamber or me.b < abs(me.c) - chamber
    trigger_if(bad, "QK-SYN-002")


@_guarded
def check_two_qubit_basis(res, a):
    """QK-SYN-003."""
    from qiskit.circuit import QuantumCircuit

    me = a["self"]
    if a["use_dag"] or not isinstance(res, QuantumCircuit):
        return
    fid = me.basis_fidelity if a["basis_fidelity"] is None else a["basis_fidelity"]
    if fid != 1.0 or me.gate_name != "cx":
        return
    u = np.asarray(a["unitary"], dtype=complex)
    if u.shape != (4, 4):
        return
    defect = _unitary_defect(u)
    # the library's internal Weyl step is specialised within _WEYL_FID (operator error ~2*sqrt(1.25*_WEYL_FID))
    tol = _SYN_TOL + 8 * float(np.sqrt(_WEYL_FID)) + 8 * defect
    v = _operator_data(res)
    n_cx = res.count_ops().get("cx", 0)
    trigger_if(_phase_error(u, v) > tol or n_cx > 3, "QK-SYN-003")


@_guarded
def check_qsd(res, a):
    """QK-SYN-004 (quantum Shannon decomposition)."""
    mat = np.asarray(a["mat"], dtype=complex)
    if mat.shape[0] > 16 or mat.shape[0] != mat.shape[1]:
        return
    defect = _unitary_defect(mat)
    if defect > 1e-9:
        return
    v = _operator_data(res)
    n = int(round(np.log2(mat.shape[0])))
    # 4**(n-2) two-qubit base-case blocks, each a Weyl step specialised within _WEYL_FID
    tol = _SYN_TOL + 8 * float(np.sqrt(_WEYL_FID)) * max(1, 4 ** max(n - 2, 0)) + 8 * defect
    trigger_if(_phase_error(mat, v) > tol, "QK-SYN-004")


@_guarded
def check_clifford_synthesis(res, a):
    """QK-SYN-004 (Clifford synthesis)."""
    from qiskit.quantum_info import Clifford

    cliff = a["clifford"]
    if cliff.num_qubits > 8:
        return
    trigger_if(Clifford(res) != cliff, "QK-SYN-004")


@_guarded
def check_qft(res, a):
    """QK-SYN-005."""
    n = a["num_qubits"]
    if a["approximation_degree"] != 0 or n > 8 or n < 1:
        return
    dim = 2**n
    j = np.arange(dim)
    f = np.exp(2j * np.pi * np.outer(j, j) / dim) / np.sqrt(dim)
    if a["inverse"]:
        f = f.conj()
    rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(dim)])
    u = _operator_data(res)
    tol = _C * _EPS * dim * n
    if a["do_swaps"]:
        bad = float(np.abs(u - f).max()) > tol
    else:
        # without the final swaps the output register is bit-reversed
        bad = min(float(np.abs(u - f[rev]).max()), float(np.abs(u - f[:, rev]).max())) > tol
    trigger_if(bad, "QK-SYN-005")


@_guarded
def check_product_formula(res, a):
    """QK-SYN-006."""
    from scipy.linalg import expm

    from qiskit.quantum_info import SparsePauliOp

    me, evo = a["self"], a["evolution"]
    op, t = evo.operator, evo.time
    if not isinstance(op, SparsePauliOp) or isinstance(t, bool):
        return
    try:
        t = float(t)
    except TypeError:
        return
    if op.num_qubits > 5 or op.size > 14:
        return
    coeffs = np.asarray(op.coeffs)
    if np.abs(coeffs.imag).max(initial=0.0) > 1e-12:
        return
    c = coeffs.real
    h = _spo_matrix(op)
    d = h.shape[0]
    exact = expm(-1j * t * h)
    got = _operator_data(res)
    err = float(np.linalg.norm(got - exact, 2))
    norm1 = float(np.abs(c).sum())
    tol = _C * _EPS * d * (1.0 + norm1 * abs(t)) * max(op.size, 1)
    paulis = op.paulis
    anti = 0.0
    for i in range(op.size):
        for k in range(i + 1, op.size):
            if not paulis[i].commutes(paulis[k]):
                anti += abs(c[i] * c[k])
    if anti == 0.0:
        bad = err > tol
    elif getattr(me, "order", None) == 1:
        # first-order Lie-Trotter bound: (t^2 / 2r) * sum_{j<k} ||[H_j, H_k]||, ||[P, Q]|| = 2
        bad = err > t * t * anti / me.reps + tol
    else:
        return
    trigger_if(bad, "QK-SYN-006")


@_guarded
def check_transpile(res, a):
    """QK-TRN-001."""
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import Operator

    if a["approximation_degree"] not in (None, 1.0) or a["unitary_synthesis_method"] != "default":
        return
    ins, outs = a["circuits"], res
    if isinstance(ins, QuantumCircuit):
        ins, outs = [ins], [outs]
    elif not isinstance(ins, (list, tuple)) or not isinstance(outs, (list, tuple)):
        return
    bad = False
    for inp, out in zip(ins, outs):
        if not isinstance(inp, QuantumCircuit) or not isinstance(out, QuantumCircuit):
            continue
        if inp.num_qubits > 6 or out.num_qubits != inp.num_qubits:
            continue
        try:
            u_in = Operator(inp).data
            u_out = Operator.from_circuit(out).data
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        # each re-synthesised two-qubit block is a Weyl decomposition specialised within the
        # library's default infidelity (_WEYL_FID), i.e. an operator error of ~2*sqrt(1.25*_WEYL_FID)
        n_blocks = max(1, sum(1 for i in inp.data if len(i.qubits) >= 2))
        tol = _SYN_TOL * (1 + out.size() / 100.0) + 8 * float(np.sqrt(_WEYL_FID)) * n_blocks
        bad = bad or _phase_error(u_in, u_out) > tol
    trigger_if(bad, "QK-TRN-001")
