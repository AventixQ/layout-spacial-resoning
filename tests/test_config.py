import os

from layout_spatial_reasoning.config import env_float, env_int, env_str, load_env


def test_load_env_does_not_override_existing_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "EXAMPLE_KEY=from_file\n"
        "NUMBER_VALUE=7\n"
        "FLOAT_VALUE=0.25\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXAMPLE_KEY", "from_process")

    load_env(env_path)

    assert os.environ["EXAMPLE_KEY"] == "from_process"
    assert env_str("NUMBER_VALUE") == "7"
    assert env_int("NUMBER_VALUE", 0) == 7
    assert env_float("FLOAT_VALUE", 0.0) == 0.25
