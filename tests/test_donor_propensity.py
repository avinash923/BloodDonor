"""UCI dataset integrity, portable XGBoost propensity model and API exposure."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.api.server as api
import src.data.uci_loader as uci
from src.api.donor_accounts import donation_history
from src.api.propensity import PropensityService
from src.models.donor_propensity import (DEFAULT_MODEL_PATH, DonorPropensityModel, RFMTFeatures,
                                         engineer_row, rfmt_from_history)
from tests.test_api_server import api_server, request, login, create
from tests.test_donor_registration import register

DATA = Path(__file__).resolve().parents[1] / "data/raw/uci/transfusion.data"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------- data

def test_committed_dataset_is_canonical_uci():
    df = uci.load_uci_dataframe(DATA, verify=True)
    assert df.shape == (748, 5) and int(df["donated"].sum()) == 178
    assert (df["monetary"] == 250 * df["frequency"]).all()


def test_modified_dataset_fails_checksum():
    df = pd.read_csv(DATA)
    df.iloc[0, 0] += 1
    with pytest.raises(ValueError, match="checksum"):
        uci.verify_uci_content(df)


def test_download_falls_back_to_verified_mirror(tmp_path, monkeypatch):
    corrupt = tmp_path / "corrupt.csv"
    pd.read_csv(DATA).iloc[:-1].to_csv(corrupt, index=False)
    monkeypatch.setattr(uci, "_MIRROR_URLS", (
        (tmp_path / "missing.csv").as_uri(), corrupt.as_uri(), DATA.resolve().as_uri()))
    saved = uci.download_uci_dataset(str(tmp_path / "out"))
    uci.load_uci_dataframe(saved, verify=True)


def test_download_fails_when_no_source_verifies(tmp_path, monkeypatch):
    monkeypatch.setattr(uci, "_MIRROR_URLS", ((tmp_path / "missing.csv").as_uri(),))
    with pytest.raises(RuntimeError, match="verified copy"):
        uci.download_uci_dataset(str(tmp_path / "out"))


# ---------------------------------------------------------------- features

def test_engineered_features():
    assert engineer_row(2, 5, 20) == [2.0, 5.0, 20.0, 0.25, 0.1, 4.5]
    assert engineer_row(4, 1, 4)[-1] == 0.0          # single donation: no gap
    assert engineer_row(2, 5, 20, "raw") == [2.0, 5.0, 20.0]


def test_rfmt_from_history_combines_sources():
    assert rfmt_from_history([], NOW) is None
    recorded = (NOW - timedelta(days=30.4375 * m) for m in (2, 1))  # generator consumed once
    features = rfmt_from_history(recorded, NOW, prior_donations=3,
                                 prior_first_at=NOW - timedelta(days=30.4375 * 24),
                                 prior_last_at=NOW - timedelta(days=30.4375 * 6))
    assert features.frequency == 5
    assert features.recency_months == pytest.approx(1.0)
    assert features.tenure_months == pytest.approx(24.0)


def test_self_reported_history_validation():
    assert donation_history({}) is None
    assert donation_history({"prior_donation_count": 0}) is None
    single = donation_history({"prior_donation_count": 1, "months_since_last_donation": 5}, NOW)
    assert single["first_donation_at"] == single["last_donation_at"]
    for bad in ({"prior_donation_count": 2, "months_since_last_donation": 9, "months_since_first_donation": 3},
                {"prior_donation_count": -1}, {"prior_donation_count": True},
                {"prior_donation_count": 2, "months_since_last_donation": float("nan"), "months_since_first_donation": 4},
                {"prior_donation_count": 0, "months_since_last_donation": 3},
                {"prior_donation_count": 1, "months_since_last_donation": 3, "months_since_first_donation": 9}):
        with pytest.raises(ValueError):
            donation_history(bad, NOW)


# ---------------------------------------------------------------- model

@pytest.fixture(scope="module")
def model():
    return DonorPropensityModel.load(DEFAULT_MODEL_PATH)


def test_shipped_model_metadata_and_quality(model):
    art = model.artefact
    assert art["training_data"]["content_sha256"] == uci.EXPECTED_CONTENT_SHA256
    assert art["export_check"]["max_abs_margin_diff"] < 1e-4
    df = uci.load_uci_dataframe(DATA)
    p = np.array(model.predict_many(df[["recency", "frequency", "time"]].to_numpy(float)))
    assert ((p > 0) & (p < 1)).all()
    assert abs(p.mean() - df["donated"].mean()) < 0.03          # calibration-in-the-large
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(df["donated"], p) > 0.7
    assert art["nested_cv_estimate"]["roc_auc_mean"] > 0.7      # honest out-of-sample estimate


def test_out_of_range_inputs_are_clipped_and_flagged(model):
    result = model.estimate(RFMTFeatures(recency_months=200, frequency=80, tenure_months=300))
    assert set(result["features"]["out_of_training_range"]) == {"recency_months", "frequency", "tenure_months"}
    assert 0 < result["probability"] < 1


def test_portable_export_matches_xgboost():
    xgb_training = pytest.importorskip("src.models.donor_propensity_training")
    df = uci.load_uci_dataframe(DATA)
    X, y = df[["recency", "frequency", "time"]].to_numpy(float), df["donated"].to_numpy()
    est = xgb_training.XGBPropensityEstimator(tune=False, params={"n_estimators": 40, "max_depth": 3},
                                              calibrate=True).fit(X, y)
    meta = {"training_ranges": {"recency_months": [0, 74], "frequency": [1, 50], "tenure_months": [2, 98]},
            "training_data": {"positive_rate": float(y.mean())}}
    portable = DonorPropensityModel(json.loads(json.dumps(est.export_portable(X, meta))))
    grid = np.array([[r, f, t] for r in (0, 1, 4, 11.5, 30) for f in (1, 2, 7, 20) for t in (2, 15.5, 40, 98) if t >= r])
    rows = np.vstack([X, grid])
    assert np.max(np.abs(np.array(portable.predict_many(rows)) - est.predict_proba(rows)[:, 1])) < 1e-5


def test_service_reports_why_no_prediction(model):
    assert PropensityService(None).estimate({}, None, NOW)["reason"] == "MODEL_NOT_LOADED"
    assert PropensityService(model).estimate({}, None, NOW)["reason"] == "NO_DONATION_HISTORY"
    assert PropensityService.load("/nonexistent/model.json").model is None


# ---------------------------------------------------------------- API

def test_profile_and_recommendations_expose_propensity(api_server):
    donor = login(api_server, "donor_001")
    estimate = request(api_server, "donor/profile", token=donor)[1]["donation_propensity"]
    assert estimate["available"] and 0 < estimate["probability"] < 1
    assert estimate["history_sources"] == ["synthetic_demo"]
    assert estimate["features"]["frequency"] == 6
    staff = login(api_server, "staff_001")
    recs = request(api_server, "emergency/recommendations?request_id=REQ_TRAUMA_101", token=staff)[1]
    assert recs["recommendations"]
    assert all(0 < r["donation_propensity"] < 1 for r in recs["recommendations"])


def test_registration_and_history_update(api_server):
    status, account = register(api_server, prior_donation_count=4, months_since_last_donation=2,
                                months_since_first_donation=20)
    assert status == 201
    profile = request(api_server, "donor/profile", token=account["token"])[1]
    assert profile["donation_propensity"]["available"]
    assert profile["donation_propensity"]["features"]["frequency"] == 4
    assert request(api_server, "donor/history", "POST", account["token"],
                   {"prior_donation_count": 2, "months_since_last_donation": 9,
                    "months_since_first_donation": 3})[0] == 400
    status, body = request(api_server, "donor/history", "POST", account["token"], {"prior_donation_count": 0})
    assert status == 200 and body["donation_propensity"]["reason"] == "NO_DONATION_HISTORY"
    assert request(api_server, "donor/history", "POST", login(api_server, "staff_001"), {})[0] == 401
    assert register(api_server, username="other_donor", prior_donation_count=-3)[0] == 400
