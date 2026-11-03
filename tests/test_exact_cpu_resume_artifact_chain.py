from pathlib import Path

from scripts.exact_cpu_resume_artifact_chain import finish_chain, start_chain


def test_artifact_chain_round_trip_preserves_exact_training_trajectory(tmp_path: Path):
    produced = tmp_path / "produced"
    resumed = tmp_path / "resumed"

    start = start_chain(produced)
    assert start["cursor"] == 7
    assert start["accumulation_position"] == 1
    assert start["state_sha256"]
    assert (produced / "resume.pt").is_file()
    assert (produced / "manifest.json").is_file()

    result = finish_chain(
        state_path=produced / "resume.pt",
        manifest_path=produced / "manifest.json",
        output_dir=resumed,
    )
    assert result["input_state_sha256"] == start["state_sha256"]
    assert result["final_cursor"] == 13
    assert result["final_accumulation_position"] == 0
    assert result["parameters_equal"] is True
    assert result["optimizer_equal"] is True
    assert result["scheduler_equal"] is True
    assert result["next_python_rng_equal"] is True
    assert result["next_numpy_rng_equal"] is True
    assert result["next_torch_rng_equal"] is True
    assert result["all_equal"] is True
    assert (resumed / "result.json").is_file()
