"""Test whether lagged nationwide survey errors transfer to Cushing, using only previously published data."""

import json

import fit_cushing_survey_balance as local
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

FAMILIES = {
    "four_week": ["national4"],
    "short_and_long": ["national1", "national12", "surprise"],
    "conditional": ["national4", "difference", "current_available", "local_sign"],
}


def released_national_history():
    sources = pd.read_csv(local.v5.MEASUREMENTS).set_index("week_ending")
    if not sources.index.is_unique:
        raise ValueError("Duplicate national survey week")
    reports = pd.read_csv(local.EXTERNAL, usecols=["regional_week", "regional_time", "national_prior"]).dropna()
    reports["regional_time"] = pd.to_datetime(reports.regional_time, utc=True)
    first = reports.sort_values("regional_time").drop_duplicates("regional_week").set_index("regional_week")
    history = first.join(sources[["national_kbbl", "national_time"]]).dropna()
    history["national_time"] = pd.to_datetime(history.national_time, utc=True)
    return history[history.national_time < history.regional_time]


def national_features(rows, history):
    features = []
    for r in rows.itertuples():
        earlier = history[(history.index < r.week_ending) & (history.regional_time < r.cutoff)]
        age = (pd.Timestamp(r.week_ending) - pd.to_datetime(earlier.index)).days.to_numpy() / 7
        gap = np.clip((earlier.national_prior - earlier.national_kbbl / 1000).to_numpy() / 4, -2, 2)
        item = {f"national{n}": float(np.exp2(-(age - 1) / n) @ gap) / np.sqrt(n) for n in (1, 4, 12)}
        have = pd.notna(r.national_kbbl)
        shift = (
            (r.national_kbbl / 1000 - earlier.national_prior.tail(8).median()) / 4
            if have and len(earlier)
            else 0
        )
        item.update(
            surprise=np.clip(shift, -2, 2),
            difference=np.clip(shift - r.survey, -2, 2) if have and pd.notna(r.survey) else 0,
            current_available=item["national4"] * float(have),
            local_sign=item["national4"] * np.sign(r.state16),
        )
        features.append(item)
    return pd.DataFrame(features, index=rows.index)


def adjusted(train, dates, fields):
    train = train[(train.known < dates.cutoff.min()) & train.survey.notna()]
    x = train[fields].to_numpy()
    y = (train.target - train.estimate).to_numpy()

    def loss(beta):
        e = x @ beta - y
        d = np.hypot(0.09, e)
        return d.sum() + 4 * (beta @ beta), x.T @ (e / d) + 8 * beta

    fit = minimize(loss, np.zeros(len(fields)), jac=True, bounds=[(-0.2, 0.2)] * len(fields), method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(fit.message)
    return np.where(dates.survey.notna(), dates.estimate + dates[fields].to_numpy() @ fit.x, dates.estimate)


def audit():
    prior, data = local.setup()
    _, val = local.calibration_replay(prior, data)
    val["estimate"] = local.apply(val, json.loads(local.WEIGHTS.read_bytes())["mix"])
    history = released_national_history()
    val = val.join(national_features(val, history))
    val = val[val.survey.notna()]
    trials = {}
    for name, cols in FAMILIES.items():
        deltas = {}
        for year in ("2022", "2023"):
            train = val[val.release_date.str[:4] < year]
            group = val[val.release_date.str[:4] == year]
            pred = adjusted(train, group, cols)
            deltas[year] = float(1000 * (abs(pred - group.target) - abs(group.estimate - group.target)).mean())
        trials[name] = deltas
    winner = min(trials, key=lambda name: np.mean(list(trials[name].values())))
    test = data[data.phase == "holdout"].copy()
    published = pd.read_csv(local.MODEL_DIR / "balance_forecasts.csv").set_index("release_date")
    if list(published.index) != list(test.release_date):
        raise ValueError("Unexpected evaluation calendar")
    test["estimate"] = test.release_date.map(published.estimate)
    test = test.join(national_features(test, history))
    settled = pd.concat([val, test]).sort_values("cutoff")
    for _, group in test.groupby(test.release_date.str[:4]):
        test.loc[group.index, "national_adjusted"] = adjusted(settled, group, FAMILIES[winner])
    cohort = test[test.release_date.isin(published[published.original_quote.notna()].index)]
    return {
        "pre_2024_yearly_delta_kbbl": trials,
        "selected": winner,
        "scores": {
            name: local.metrics(frame, "national_adjusted")
            for name, frame in (("all", test), ("original_quoted", cohort))
        },
    }


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        print(json.dumps(audit(), indent=2))
