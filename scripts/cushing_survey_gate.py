"""Evolve a small nonlinear network that weights the independent weekly survey against the incumbent."""
import numpy as np
import pandas as pd

HIDDEN = 8
INPUTS = 6
SIZE = INPUTS * HIDDEN + 2 * HIDDEN + 1
POPULATION = 64


def features(rows):
    # Fixed units let the same genome be applied to an unlabeled release without a future scaler.
    d = rows.survey.to_numpy() - rows.strong_reference.to_numpy()
    return np.clip(np.column_stack([d, np.abs(d), rows.survey, rows.vol13 - 1, rows.persistence13, rows.basis_move5]), -4., 4.)


def output(genomes, x, survey, base):
    w = genomes[:, :INPUTS * HIDDEN].reshape(-1, INPUTS, HIDDEN)
    b = genomes[:, INPUTS * HIDDEN : -HIDDEN-1]
    v = genomes[:, -HIDDEN-1:-1]
    hidden = np.tanh(x @ w + b[:, None, :])
    logits = 2.0 + genomes[:, -1, None] + (hidden * v[:, None, :]).sum(axis=2) / np.sqrt(HIDDEN)
    weight = 1 / (1 + np.exp(-np.clip(logits, -12, 12)))
    return base + weight * (survey - base)


def fit_gate(ledger, cutoff):
    observed = ledger[(ledger.known < cutoff) & ledger.survey.notna() & ledger.strong_reference.notna()].sort_values("cutoff")
    if len(observed) < 40:
        raise ValueError("Insufficient previously published survey comparisons")
    stop = max(12, min(39, len(observed) // 4))
    boundary = observed.cutoff.iloc[-stop]
    train = observed[observed.known < boundary]
    valid = observed.tail(stop)
    if len(train) < 24 or train.known.max() >= valid.cutoff.min():
        raise ValueError("Invalid chronological early stopping split")
    def loss(genes, sample):
        p = output(genes, features(sample), sample.survey.to_numpy(), sample.strong_reference.to_numpy())
        return np.abs(p - sample.target.to_numpy()).mean(axis=1)
    rng = np.random.default_rng(17915)
    genes = rng.normal(0, .32, (POPULATION, SIZE))
    genes[0] = 0.
    best_genome = genes[0].copy()
    best_validation = loss(genes[:1], valid)[0]
    last_improvement = 0
    log = []
    for generation in range(1, 193):
        train_mae = loss(genes, train)
        # Choose parents on training only. Check one training winner on the later stopping block.
        score = train_mae + .005 * (genes * genes).mean(axis=1)
        elite = genes[np.argsort(score)[:8]]
        probe = loss(elite[:1], valid)[0]
        if probe < best_validation - .0001:
            best_genome, best_validation, last_improvement = elite[0].copy(), probe, generation
        log.append(dict(generation=generation, best_training_mae_kbbl=float(train_mae.min() * 1000),
                        stopping_block_mae_kbbl=float(probe * 1000), best_stopping_mae_kbbl=float(1000 * best_validation)))
        if generation >= 48 and generation - last_improvement >= 32:
            break
        left, right = elite[rng.integers(len(elite), size=(2, POPULATION - len(elite)))]
        children = np.where(rng.random(left.shape) < .5, left, right)
        mutate = rng.random(children.shape) < .3
        children += mutate * rng.normal(0, max(.035, .35 * .985 ** generation), children.shape)
        genes = np.vstack([elite, np.clip(children, -4.0, 4.0)])
    return dict(weights=best_genome.tolist(), training_rows=len(train), stopping_rows=len(valid),
                latest_label_utc=observed.known.max().isoformat(), chosen_generation=last_improvement), log


def forecast(gate, rows):
    if len(rows) and pd.Timestamp(gate["latest_label_utc"]) >= rows.cutoff.min():
        raise ValueError("Survey network used a label unavailable at prediction time")
    return output(np.array([gate["weights"]]), features(rows), rows.survey.to_numpy(), rows.strong_reference.to_numpy())[0]
