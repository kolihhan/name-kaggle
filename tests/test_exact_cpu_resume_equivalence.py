from scripts.exact_cpu_resume_equivalence import run_equivalence_probe


def test_uninterrupted_and_mid_accumulation_resumed_training_are_exactly_equivalent(tmp_path):
    result = run_equivalence_probe(tmp_path)

    assert result["microbatches"] == 13
    assert result["accumulation_size"] == 6
    assert result["interrupted_after_microbatch"] == 7
    assert result["final_cursor"] == 13
    assert result["final_accumulation_position"] == 0
    assert result["parameters_equal"] is True
    assert result["optimizer_equal"] is True
    assert result["scheduler_equal"] is True
    assert result["next_python_rng_equal"] is True
    assert result["next_numpy_rng_equal"] is True
    assert result["next_torch_rng_equal"] is True
    assert result["all_equal"] is True
