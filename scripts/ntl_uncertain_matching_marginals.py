"""Uncertain prompt matching: exact marginalization over partial matchings.

Research-only CPU primitive. Input pair odds are externally supplied, not
learned or calibrated here. A cut distribution over matchings depends only
on those odds, not on classifier labels. For a matched pair, condition
independent categorical scores on different labels; for unmatched nodes,
retain their original scores. Average over all possible disjoint matchings.

This is not a scored candidate and does not modify a submission. Existing
v29 already contains grouped joint marginals; the proposed distinction is
explicit uncertainty over the matching itself rather than one fixed group.
"""
from functools import lru_cache
import math
import numpy as np

MAX_COMPONENT = 12
MAX_ODDS = 1e6

def marginalize(probabilities, edges, *, max_component=MAX_COMPONENT):
    """Return proposed probability rows and aggregate diagnostics.

    edges: iterable of (row index, row index, nonnegative finite odds).
    Matching prior mass is proportional to product of included edge odds,
    with unmatched nodes of weight 1. Not a guarantee of calibrated odds.
    Oversized components are skipped as a whole. No class labels consumed.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not len(p) or p.shape[1] != 18:
        raise ValueError("expected nonempty N x 18 probability rows")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-10, rtol=0):
        raise ValueError("invalid categorical probabilities")
    if not isinstance(max_component, int) or not 2 <= max_component <= MAX_COMPONENT:
        raise ValueError("component cap invalid")
    n = len(p)
    adj = [dict() for _ in range(n)]
    seen = set()
    for i, j, odds in edges:
        if not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)):
            raise ValueError("noninteger edge index")
        if not 0 <= i < n or not 0 <= j < n or i == j:
            raise ValueError("invalid edge endpoints")
        key = tuple(sorted((int(i), int(j))))
        if key in seen:
            raise ValueError("duplicate edge")
        seen.add(key)
        w = float(odds)
        if not math.isfinite(w) or not 0 <= w <= MAX_ODDS:
            raise ValueError("invalid edge odds")
        if w:
            adj[i][j] = adj[j][i] = w
    out = p.copy()
    visited = set()
    stats = {"components_processed": 0, "components_skipped": 0,
             "rows_skipped": 0, "edges_processed": 0, "degenerate_edges": 0,
             "expected_matched_pairs": 0.0, "max_match_probability_sum": 0.0,
             "promotion": False, "actual_scorer_evaluated": False}
    for start in range(n):
        if start in visited or not adj[start]:
            continue
        comp = []
        todo = [start]
        visited.add(start)
        while todo:
            i = todo.pop()
            comp.append(i)
            for j in adj[i]:
                if j not in visited:
                    visited.add(j)
                    todo.append(j)
        if len(comp) > max_component:
            stats["components_skipped"] += 1
            stats["rows_skipped"] += len(comp)
            continue
        comp.sort()
        loc = {i: k for k, i in enumerate(comp)}
        m = len(comp)
        weights = [{loc[j]: w for j, w in adj[i].items()} for i in comp]
        @lru_cache(maxsize=None)
        def Z(mask):
            if not mask:
                return 1.0
            bit = mask & -mask
            i = bit.bit_length()-1
            rest = mask ^ bit
            v = Z(rest)
            for j, w in weights[i].items():
                bj = 1 << j
                if rest & bj:
                    v += w * Z(rest ^ bj)
            return v
        full = (1 << m)-1
        z = Z(full)
        assert math.isfinite(z) and z >= 1
        match = np.zeros(m)
        for u, i in enumerate(comp):
            for v, w in weights[u].items():
                if v <= u:
                    continue
                j = comp[v]
                q = w * Z(full ^ (1 << u) ^ (1 << v)) / z
                match[u] += q
                match[v] += q
                stats["expected_matched_pairs"] += q
                stats["edges_processed"] += 1
                den = 1.0 - float(p[i] @ p[j])
                if den <= 1e-12:
                    # Conditioning is undefined or numerically unstable:
                    # retain the original rows for this matching branch.
                    stats["degenerate_edges"] += 1
                    continue
                pi = p[i] * (1.0 - p[j]) / den
                pj = p[j] * (1.0 - p[i]) / den
                out[i] += q * (pi - p[i])
                out[j] += q * (pj - p[j])
        if np.any(match > 1+1e-10):
            raise ArithmeticError("invalid matching marginal mass")
        stats["max_match_probability_sum"] = max(stats["max_match_probability_sum"], float(match.max()))
        stats["components_processed"] += 1
    if np.any(out < -1e-10) or not np.isfinite(out).all() or not np.allclose(out.sum(1), 1, atol=1e-10, rtol=0):
        raise ArithmeticError("invalid posterior")
    # Preserve untouched rows exactly; remove only tiny roundoff elsewhere.
    changed = np.any(out != p, axis=1)
    if changed.any():
        v = np.maximum(out[changed], 0)
        out[changed] = v / v.sum(1, keepdims=True)
    stats["argmax_changes"] = int((p.argmax(1) != out.argmax(1)).sum())
    return out, stats

def propose(probabilities, edges, incumbent):
    """Abstain on baseline/score disagreement; no mutation of incumbent."""
    p = np.asarray(probabilities, dtype=float)
    old = np.asarray(incumbent)
    if old.shape != (len(p),) or old.dtype.kind not in "iu" or np.any(old < 0) or np.any(old >= 18):
        raise ValueError("incumbent must be ID-aligned class indices")
    post, stats = marginalize(p, edges)
    agreed = old == p.argmax(1)
    proposal = np.where(agreed, post.argmax(1), old)
    stats["score_decision_disagreements"] = int((~agreed).sum())
    stats["proposed_label_changes"] = int((proposal != old).sum())
    return proposal, stats
