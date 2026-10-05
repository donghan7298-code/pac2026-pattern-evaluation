"""Small NumPy MLP: future outcome regression plus listwise ranking loss."""
import json
from pathlib import Path

import numpy as np

from .rollout import OUTPUT_NAMES
from .scoring import FEATURE_NAMES, efficiency_score, feature_vector, priority


class Ranker:
    def __init__(self, value):
        if value.get("format") != "ahead-future-mlp-v2" or value.get("feature_names") != FEATURE_NAMES or value.get("output_names") != OUTPUT_NAMES:
            raise ValueError("Incompatible model feature/output schema")
        self.value = value
        for name in ("x_mean", "x_std", "y_mean", "y_std", "w1", "b1", "w2", "b2"):
            setattr(self, name, np.asarray(value[name], dtype=float))
        f = len(FEATURE_NAMES); o = len(OUTPUT_NAMES)
        if self.w1.ndim != 2 or self.w1.shape[0] != f or self.w2.shape != (self.w1.shape[1], o) or self.b1.shape != (self.w1.shape[1],) or self.b2.shape != (o,) or self.x_mean.shape != (f,) or self.x_std.shape != (f,) or self.y_mean.shape != (o,) or self.y_std.shape != (o,):
            raise ValueError("Incompatible model array shapes")
        if any(not np.all(np.isfinite(getattr(self, n))) for n in ("x_mean", "x_std", "y_mean", "y_std", "w1", "b1", "w2", "b2")) or np.any(self.x_std <= 0) or np.any(self.y_std <= 0):
            raise ValueError("Invalid model values")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.value, separators=(",", ":"), allow_nan=False), encoding="utf-8")

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid model features")
        y = (np.tanh((x-self.x_mean)/self.x_std @ self.w1 + self.b1) @ self.w2 + self.b2)*self.y_std+self.y_mean
        y = np.clip(y, 0., 1.)
        y[:, 2] = np.minimum(y[:, 2], y[:, 0])
        return [dict(zip(OUTPUT_NAMES, map(float, row)), risk=float(min(1., row[0]-row[2]+.5*row[3]))) for row in y]


def arrays(groups):
    x, y, offsets = [], [], [0]
    for group in groups:
        for row in group["rows"]:
            x.append(row["features"]); y.append([row["future"][n] for n in OUTPUT_NAMES])
        offsets.append(len(x))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), offsets


def ranking_metrics(model, groups, weights, top_k=4):
    x, y, offsets = arrays(groups); predicted = model.predict(x)
    hits, exact, regrets, overlap = [], [], [], []
    for gi, group in enumerate(groups):
        rows = group["rows"]; start, end = offsets[gi:gi+2]
        actual = [efficiency_score(r["static"], r["future"], weights) for r in rows]
        estimated = [efficiency_score(r["static"], f, weights) for r, f in zip(rows, predicted[start:end])]
        truth = sorted(range(len(rows)), key=lambda i: priority(rows[i]["static"], actual[i]), reverse=True)
        order = sorted(range(len(rows)), key=lambda i: priority(rows[i]["static"], estimated[i]), reverse=True)
        best_key = priority(rows[truth[0]]["static"], actual[truth[0]])
        tied = {i for i in truth if priority(rows[i]["static"], actual[i])[:2] == best_key[:2] and abs(actual[i]-actual[truth[0]]) <= 1e-7}
        hits.append(bool(tied.intersection(order[:top_k]))); exact.append(order[0] in tied)
        regrets.append(max(0., actual[truth[0]]-actual[order[0]]))
        overlap.append(len(set(order[:top_k]).intersection(truth[:top_k]))/min(top_k, len(rows)))
    raw = np.asarray([[f[n] for n in OUTPUT_NAMES] for f in predicted])
    return dict(groups=len(groups), candidates=len(x), top_k=top_k,
                teacher_best_top_k_recall=float(np.mean(hits)), teacher_top1_match=float(np.mean(exact)),
                mean_teacher_score_regret=float(np.mean(regrets)), mean_top_k_overlap=float(np.mean(overlap)),
                output_mse={n: float(v) for n, v in zip(OUTPUT_NAMES, np.mean((raw-y)**2, axis=0))})


def train(train_groups, validation_groups, weights, seed=20261005, epochs=80, hidden=48):
    if not train_groups or not validation_groups:
        raise ValueError("Separate nonempty train and validation groups are required")
    train_ids = sorted({g["base_group"] for g in train_groups})
    validation_ids = sorted({g["base_group"] for g in validation_groups})
    if set(train_ids).intersection(validation_ids):
        raise ValueError("Training and validation inventory groups must be disjoint")
    x, y, offsets = arrays(train_groups); vx, vy, _ = arrays(validation_groups)
    rng = np.random.default_rng(seed)
    xm = x.mean(axis=0); xs = np.maximum(x.std(axis=0), .05)
    ym = y.mean(axis=0); ys = np.maximum(y.std(axis=0), [.05, .02, .05, .05])
    z = (x-xm)/xs; target = (y-ym)/ys; vz = (vx-xm)/xs
    params = [rng.normal(0, np.sqrt(1/z.shape[1]), (z.shape[1], hidden)), np.zeros(hidden),
              rng.normal(0, np.sqrt(1/hidden), (hidden, 4)), np.zeros(4)]
    first = [np.zeros_like(v) for v in params]; second = [np.zeros_like(v) for v in params]
    step = 0; best = None; best_loss = float("inf"); history = []
    coef = np.asarray([weights["future_fit"]-weights["risk"], weights["future_volume"],
                       weights["risk"], -.5*weights["risk"]])

    def update(batch, derivative, activations):
        nonlocal step
        w1, b1, w2, b2 = params
        dh = derivative @ w2.T * (1-activations**2)
        grads = [batch.T @ dh + 1e-5*w1, dh.sum(axis=0),
                 activations.T @ derivative + 1e-5*w2, derivative.sum(axis=0)]
        step += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            g = np.clip(g, -5., 5.)
            first[i] = .9*first[i]+.1*g; second[i] = .999*second[i]+.001*g*g
            p -= .002 * (first[i]/(1-.9**step)) / (np.sqrt(second[i]/(1-.999**step))+1e-8)

    for epoch in range(epochs):
        for start in range(0, len(z), 256):
            # Shuffle every epoch, retaining group boundaries for the ranking pass.
            if start == 0: order = rng.permutation(len(z))
            ids = order[start:start+256]; batch = z[ids]
            h = np.tanh(batch @ params[0]+params[1]); output = h @ params[2]+params[3]
            update(batch, 2*(output-target[ids])/output.size, h)
        # Learn relative ordering over entire candidate groups, rather than isolated scores.
        for gi in rng.permutation(len(train_groups))[::3]:
            group = train_groups[gi]; lo, hi = offsets[gi:gi+2]; rows = group["rows"]
            best_safety = max(priority(r["static"], 0.)[:2] for r in rows)
            ids = [j for j, r in enumerate(rows) if priority(r["static"], 0.)[:2] == best_safety]
            if len(ids) < 2: continue
            batch = z[lo+np.asarray(ids)]; h = np.tanh(batch @ params[0]+params[1])
            output = (h @ params[2]+params[3])*ys+ym
            static = np.asarray([efficiency_score(rows[j]["static"], dict(mean_fit=0., mean_volume=0., risk=0.), weights) for j in ids])
            truth = np.asarray([efficiency_score(rows[j]["static"], rows[j]["future"], weights) for j in ids])
            estimated = static + output @ coef
            def softmax(v):
                v = np.exp((v-v.max())/.04); return v/v.sum()
            derivative = .08*(softmax(estimated)-softmax(truth))[:, None]/.04 * (coef*ys)[None, :]
            update(batch, derivative, h)
        prediction = (np.tanh(vz @ params[0]+params[1]) @ params[2]+params[3])*ys+ym
        loss = float(np.mean(((prediction-vy)/ys)**2))
        history.append(dict(epoch=epoch+1, validation_normalized_mse=loss))
        if loss < best_loss:
            best_loss = loss; best = ([v.copy() for v in params], epoch+1)
        if (epoch+1) % 20 == 0: print(f"ranker epoch {epoch+1}/{epochs}: validation loss={loss:.5f}", flush=True)
    p, chosen_epoch = best
    value = dict(format="ahead-future-mlp-v2", feature_names=FEATURE_NAMES, output_names=OUTPUT_NAMES,
                 x_mean=xm.tolist(), x_std=xs.tolist(), y_mean=ym.tolist(), y_std=ys.tolist(),
                 w1=p[0].tolist(), b1=p[1].tolist(), w2=p[2].tolist(), b2=p[3].tolist(),
                 training=dict(seed=seed, epochs=epochs, selected_epoch=chosen_epoch,
                               objective="normalized outcome MSE + listwise candidate ranking", hidden=hidden,
                               train_base_groups=train_ids, validation_base_groups=validation_ids))
    return Ranker(value), dict(history=history, selected_epoch=chosen_epoch,
        train=ranking_metrics(Ranker(value), train_groups, weights),
        validation=ranking_metrics(Ranker(value), validation_groups, weights))
