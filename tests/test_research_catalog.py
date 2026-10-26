from html.parser import HTMLParser
from pathlib import Path
import json
import pytest
from chessengine.common import digest
from chessengine.research_catalog import publish, sanitize


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for key, value in attrs if key in ("href", "src"))


def test_report_publication_is_independent_of_failed_model_gate(tmp_path):
    original = {
        str(p): digest(p)
        for directory in ["artifacts/models", "web/public/models"]
        for p in Path(directory).glob("*")
        if p.is_file()
    }
    rows = publish(tmp_path / "research")
    assert len(rows) == 11
    rank = next(r for r in rows if r["id"] == "move-ranking")
    assert rank["result"] == "inconclusive" and rank["availability"] == "local preview"
    assert json.load(open("docs/experiments/move-ranking/final-audit.json"))["release_eligible"] is False
    pilot = next(r for r in rows if r["id"] == "policy-value")
    assert pilot["level"] == "Completed feasibility pilot"
    assert pilot["result"] == "pilot"
    assert "deferred" in pilot["takeaway"]
    assert (tmp_path / "research/policy-value/release-scope.json").is_file()
    for p, sha in original.items():
        assert digest(p) == sha
    for document in (tmp_path / "research").glob("*/index.html"):
        text = document.read_text()
        assert "/Users/" not in text and "runs/" not in text and "data/" not in text
        links = Links()
        links.feed(text)
        for link in links.targets:
            if link.startswith(("https://", "#", "/research")):
                continue
            assert (document.parent / link).is_file(), link


def test_private_links_are_removed_and_unlisted_images_rejected():
    doc = sanitize('<a href="../../runs/private.json">Evidence</a><img src="curves.png">', {"curves.png"})
    assert "../../runs/" not in doc and "Evidence" in doc
    with pytest.raises(ValueError, match="non-allowlisted"):
        sanitize('<img src="../../runs/private.png">', set())


def test_public_evidence_keeps_distinct_artifact_hashes_and_omits_paths(tmp_path):
    from chessengine.pv_reporting import public, build

    raw = {"runs/a/one.npz": "a" * 64, "runs/b/two.npz": "b" * 64}
    clean = public(raw)
    assert len(clean) == 2 and set(clean.values()) == set(raw.values())
    root = tmp_path / "run"
    root.mkdir()
    from chessengine.common import write_json

    write_json(
        root / "protocol.json",
        {
            "identity": {
                "inputs": {"runs/private.npz": "f" * 64},
                "config": {
                    "pilot_simulations": [128, 256, 512],
                    "anchor_weights": {"baseline": 0.0, "anchored": 0.1},
                    "seeds": [42, 43, 44],
                    "initializer": "runs/private.npz",
                    "policy_seed": 20261015,
                },
            }
        },
    )
    evidence = build(root, tmp_path / "report")
    assert evidence["status"]["status"] == "pilot_in_progress"
    text = (tmp_path / "report/index.html").read_text()
    assert "No completed final comparison is claimed" in text
    assert "<svg" not in text
    assert "runs/" not in (tmp_path / "report/protocol.json").read_text()


def test_measured_learning_figure_is_a_standalone_svg(tmp_path):
    from chessengine.pv_reporting import learning_plot

    training = {
        "baseline-42": {
            "history": [
                {"batch": 1, "frozen_checkpoint_outcome_mse": 0.4, "frozen_checkpoint_anchor_mse": 0.01},
                {"batch": 2, "frozen_checkpoint_outcome_mse": 0.3, "frozen_checkpoint_anchor_mse": 0.02},
            ]
        }
    }
    path = learning_plot(training, tmp_path, tmp_path)
    text = path.read_text()
    assert path.name == "learning.svg" and "<svg" in text
    assert "Changing self-play data" in text
    assert learning_plot({}, tmp_path, tmp_path) is None


def test_scope_amendment_preserves_interrupted_study_without_claiming_full_results(tmp_path):
    from chessengine.common import write_json
    from chessengine.pv_reporting import build

    source = {
        "identity": {
            "inputs": {"runs/private.npz": "f" * 64},
            "config": {
                "pilot_simulations": [128, 256, 512],
                "anchor_weights": {"baseline": 0.0, "anchored": 0.1},
                "seeds": [42, 43, 44],
                "initializer": "runs/private.npz",
                "policy_seed": 20261015,
            },
        }
    }
    # Closure must match the exact private protocol bytes, before publication strips paths.
    root, out = tmp_path / "run", tmp_path / "report"
    write_json(root / "protocol.json", source)
    write_json(root / "status.json", {"status": "training", "seed": 42})
    write_json(root / "continuation-status.json", {"status": "interrupted"})
    scope = json.loads(Path("docs/experiments/policy-value/release-scope.json").read_text())
    scope["original_protocol_sha256"] = digest(root / "protocol.json")
    write_json(out / "release-scope.json", scope)
    evidence = build(root, out)
    assert evidence["status"]["status"] == "feasibility_complete"
    assert evidence["execution_status"]["status"] == "training"
    assert evidence["continuation-status"]["status"] == "interrupted"
    assert "results" not in evidence
    text = (out / "index.html").read_text()
    assert "Comparison deferred" in text and "remains untested" in text
    assert "All six fits and selections" not in text
    scope["original_protocol_sha256"] = "0" * 64
    write_json(out / "release-scope.json", scope)
    with pytest.raises(ValueError, match="does not match"):
        build(root, out)
