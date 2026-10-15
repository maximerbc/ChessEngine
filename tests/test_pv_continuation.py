from chessengine.common import write_json, read_json
import chessengine.pv_continue as continuation


def test_continuation_records_forecast_before_full_launch(tmp_path, monkeypatch):
    write_json(tmp_path / "status.json", {"status": "pilot_complete"})
    write_json(tmp_path / "pilot-decision.json", {"approved": True})
    write_json(tmp_path / "forecast.json", {"seconds": 3600})
    events = []
    monkeypatch.setattr(continuation, "publish", lambda: events.append("published"))

    def run(config, stage):
        assert stage == "all"
        assert read_json(tmp_path / "launch-estimate.json")["estimated_hours"] == 1
        assert events == ["published"]
        return {"status": "complete"}

    monkeypatch.setattr(continuation, "run", run)
    assert continuation.continue_study({"output": str(tmp_path)})["status"] == "complete"
    assert read_json(tmp_path / "continuation-status.json")["status"] == "complete"


def test_rejected_pilot_never_launches_training(tmp_path, monkeypatch):
    write_json(tmp_path / "status.json", {"status": "pilot_rejected"})
    write_json(tmp_path / "pilot-decision.json", {"approved": False})
    monkeypatch.setattr(continuation, "publish", lambda: None)

    def forbidden(*args):
        raise AssertionError("Rejected pilot cannot train")

    monkeypatch.setattr(continuation, "run", forbidden)
    assert continuation.continue_study({"output": str(tmp_path)})["approved"] is False
    assert read_json(tmp_path / "continuation-status.json")["status"] == "pilot_rejected"


def test_environment_change_blocks_cpu_resume(tmp_path, monkeypatch):
    import pytest

    write_json(tmp_path / "status.json", {"status": "pilot_complete"})
    write_json(tmp_path / "pilot-decision.json", {"approved": True})
    write_json(tmp_path / "forecast.json", {"seconds": 3600})
    write_json(
        tmp_path / "environment.json", {"packages": {"torch": "incompatible"}, "pyproject_sha256": "wrong"}
    )
    monkeypatch.setattr(continuation, "publish", lambda: None)
    with pytest.raises(RuntimeError, match="environment changed"):
        continuation.continue_study({"output": str(tmp_path)})
    assert read_json(tmp_path / "continuation-status.json")["status"] == "failed"
