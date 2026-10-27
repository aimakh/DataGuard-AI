import os
import pytest
import requests
from dotenv import load_dotenv

from integrations.zetaris_client import ZetarisClient
from integrations.meterless_nightshift import MeterlessNightshiftClient
from integrations.meterless_engine import MarkovianEngine
from integrations.nemotron import NemotronReasoner
from models.config import get_settings

load_dotenv("/home/aima/open-agent-hack/.env")


def test_meterless_markovian_engine_real_state():
    """Verify that the Meterless Markovian O(1) engine works with real chunked execution."""
    engine = MarkovianEngine()

    def multi_chunk_task(inp, carry):
        if carry == "":
            return "Analyzed null values. [STATE_CHECKPOINT] Step 1: found 3 nulls."
        elif "Step 1" in carry:
            return "Analyzed duplicates. [STATE_CHECKPOINT] Step 2: found 1 duplicate."
        else:
            return "Final synthesis complete. [TASK_COMPLETE]"

    run = engine.run_chain("Data quality audit", multi_chunk_task)
    assert run.completed is True
    assert len(run.chunks) == 3
    assert "Step 1" in run.chunks[0].carryover
    assert "Step 2" in run.chunks[1].carryover
    assert run.duration_ms >= 0


def test_nvidia_endpoint_live_reachability():
    """Verify that NVIDIA's API endpoint is reachable over the network."""
    settings = get_settings()
    try:
        r = requests.get(f"{settings.nvidia_api_url}/models", timeout=5)
        endpoint_reachable = (r.status_code == 200)
    except Exception:
        endpoint_reachable = False

    assert endpoint_reachable is True, f"NVIDIA endpoint {settings.nvidia_api_url} should be reachable"


def test_zetaris_live_execution():
    """Verify real Zetaris connectivity, authentication, and live query execution."""
    zc = ZetarisClient()
    zh = zc.check_health()
    assert zh["connected"] is True
    assert zh["ui_online"] is True

    # Real authentication
    login_ok = zc.login()
    assert login_ok is True
    assert zc.token is not None

    # Real live query
    success, df, msg = zc.execute_query("select 7 as seven")
    assert success is True
    assert df is not None
    assert int(df["seven"].iloc[0]) == 7

    # Real table query from TPCDS_DB
    success_tbl, df_tbl, _ = zc.execute_query("SELECT * FROM TPCDS_DB.customer_address LIMIT 3")
    assert success_tbl is True
    assert df_tbl is not None
    assert len(df_tbl) == 3


def test_meterless_nightshift_and_nemotron_statuses():
    """Verify accurate status reporting for Meterless Nightshift and Nemotron."""
    # Meterless Nightshift
    mc = MeterlessNightshiftClient()
    mh = mc.probe_runtime()
    assert mh["status"] in ("ONLINE", "STANDBY")

    # Nemotron API Key Status
    key = os.getenv("NVIDIA_API_KEY")
    reasoner = NemotronReasoner()
    res = reasoner.synthesize_investigation([], "test_dataset.csv")
    assert res["confidence_score"] == 1.0
