import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_native_qwen_dataset_payload_does_not_read_legacy_context_before_cache_branch():
    source = (
        ROOT / "src/fastwam/datasets/lerobot/robot_video_dataset.py"
    ).read_text()
    module = ast.parse(source)
    dataset_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "RobotVideoDataset"
    )
    get_method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get"
    )
    initial_payload = next(
        node.value
        for node in get_method.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "data"
            for target in node.targets
        )
        and isinstance(node.value, ast.Dict)
    )
    keys = {
        key.value
        for key in initial_payload.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert "context" not in keys
    assert "context_mask" not in keys
