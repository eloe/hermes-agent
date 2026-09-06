"""Security policy reads reject broken inputs even after a successful cached read."""

import os

import pytest


@pytest.fixture
def policy_files(tmp_path, monkeypatch):
    from hermes_cli import config, managed_scope

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    config._LOAD_CONFIG_CACHE.clear()
    config._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    managed_scope.invalidate_managed_cache()
    return home / "config.yaml", managed / "config.yaml"


@pytest.mark.parametrize("layer", [0, 1], ids=["user", "managed"])
@pytest.mark.parametrize("failure", ["malformed", "nonmapping", "unreadable", "same_signature"])
def test_strict_rejects_broken_policy_after_warm_cache(policy_files, layer, failure):
    from hermes_cli.config import load_config

    for path in policy_files:
        path.write_text("memory: {write_approval: true}\n", encoding="utf-8")
    assert load_config()["memory"]["write_approval"] is True
    path = policy_files[layer]
    before = path.stat()
    if failure == "unreadable":
        path.unlink()
        path.mkdir()  # Deterministic read error, including privileged test runners.
    elif failure == "nonmapping":
        path.write_text("- not-a-mapping\n", encoding="utf-8")
    else:
        body = "memory: ["
        if failure == "same_signature":
            body = body.ljust(before.st_size)
        path.write_text(body, encoding="utf-8")
        if failure == "same_signature":
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError):
        load_config(strict=True)


@pytest.mark.parametrize("configured", [False, True], ids=["missing-defaults", "overlay"])
def test_strict_matches_normal_resolution_without_mutating_cache(policy_files, monkeypatch, configured):
    from hermes_cli import config

    if configured:
        monkeypatch.setenv("STRICT_TEST_MODEL", "managed/model")
        policy_files[0].write_text(
            "max_turns: 17\nmodel: {name: user/model}\nmemory: {write_approval: false}\n",
            encoding="utf-8")
        policy_files[1].write_text(
            "model: {default: {provider: nous, model: '${STRICT_TEST_MODEL}'}}\n"
            "memory: {write_approval: true}\n", encoding="utf-8")
    normal = config.load_config()
    cache_before = dict(config._LOAD_CONFIG_CACHE)
    strict = config.load_config(strict=True)
    assert strict == normal
    assert config._LOAD_CONFIG_CACHE == cache_before
    if configured:
        assert strict["agent"]["max_turns"] == 17
        assert strict["model"]["default"] == "managed/model"
        assert strict["model"]["provider"] == "nous"
        assert strict["memory"]["write_approval"] is True
