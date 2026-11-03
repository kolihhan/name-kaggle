from pathlib import Path


def test_ephemeral_gpu_runner_bootstrap_is_one_shot_and_gpu_guarded() -> None:
    script = Path("scripts/bootstrap_ephemeral_gpu_runner.sh").read_text(encoding="utf-8")

    assert "--ephemeral" in script
    assert "--labels gpu" in script
    assert "RUNNER_TOKEN" in script
    assert 'need >=15 GiB VRAM' in script
    assert "nvidia-smi" in script
    assert "torch.cuda.is_available()" in script
    assert "./run.sh" in script


def test_bootstrap_does_not_print_registration_token() -> None:
    script = Path("scripts/bootstrap_ephemeral_gpu_runner.sh").read_text(encoding="utf-8")

    assert 'echo "$RUNNER_TOKEN"' not in script
    assert 'printf "%s" "$RUNNER_TOKEN"' not in script
