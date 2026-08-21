from fastwam.trainer import Wan22Trainer


def test_checkpoint_retention_keeps_only_newest_states_and_weights(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.state_dir = str(tmp_path / "state")
    trainer.weights_dir = str(tmp_path / "weights")
    trainer.max_checkpoints = 2
    (tmp_path / "state").mkdir()
    (tmp_path / "weights").mkdir()
    for step in (1, 2, 3):
        (tmp_path / "state" / f"step_{step:06d}").mkdir()
        (tmp_path / "weights" / f"step_{step:06d}.pt").touch()

    trainer._prune_checkpoints()

    assert sorted(path.name for path in (tmp_path / "state").iterdir()) == [
        "step_000002",
        "step_000003",
    ]
    assert sorted(path.name for path in (tmp_path / "weights").iterdir()) == [
        "step_000002.pt",
        "step_000003.pt",
    ]
