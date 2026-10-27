import pytest
from integrations.meterless_engine import MarkovianEngine, MarkovianRun


def test_markovian_chunked_execution():
    engine = MarkovianEngine(chunk_size=8000, max_chunks=3, carryover_tokens=256)
    
    def mock_generator(prompt, carryover):
        if not carryover:
            return "Analyzed step 1. [STATE_CHECKPOINT] Key insight: anomaly in date formats."
        elif "date formats" in carryover:
            return "Analyzed step 2. [STATE_CHECKPOINT] Correlated with mobile app client."
        else:
            return "Analyzed step 3. Solved. [TASK_COMPLETE]"

    run = engine.run_chain("Test task", mock_generator)
    assert run.completed
    assert len(run.chunks) == 3
    assert run.total_tokens_saved > 0
    assert run.efficiency_percent > 50.0
    assert "[TASK_COMPLETE]" not in run.chunks[-1].content
