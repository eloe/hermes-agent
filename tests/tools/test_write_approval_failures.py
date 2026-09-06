"""Real-import write boundaries: failures cannot authorize, lose, or replay writes."""

import builtins
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest


@pytest.fixture
def approval_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  write_approval: true\nskills:\n  write_approval: true\n")
    return tmp_path


def _tool_write(subsystem):
    if subsystem == "memory":
        from tools.memory_tool import MemoryStore, memory_tool
        store = MemoryStore()
        store.load_from_disk()
        return json.loads(memory_tool("add", "user", "approval probe", store=store))
    from tools.skill_manager_tool import skill_manage
    return json.loads(skill_manage(
        action="create", name="approval-probe",
        content="---\nname: approval-probe\ndescription: Use for an approval probe.\n---\n# Probe\n"))


def _target(home, subsystem):
    return (home / "memories" / "USER.md" if subsystem == "memory"
            else home / "skills" / "approval-probe" / "SKILL.md")


@pytest.mark.parametrize("subsystem", ["memory", "skills"])
@pytest.mark.parametrize("fault", ["config", "invalid", "import", "staging", "directory_sync"])
def test_gate_failures_do_not_mutate_or_report_staged(approval_home, monkeypatch, subsystem, fault):
    from tools import write_approval as wa

    secret = "private-fault-detail-must-not-appear"

    def fail(*args, **kwargs):
        raise OSError(secret)

    if fault == "config":
        (approval_home / "config.yaml").write_text("memory: [broken\n")
    elif fault == "invalid":
        (approval_home / "config.yaml").write_text(f"{subsystem}:\n  write_approval: maybe\n")
    elif fault == "import":
        original_import = builtins.__import__

        def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tools" and "write_approval" in (fromlist or ()):
                raise ImportError(secret)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", failing_import)
    elif fault == "staging":
        monkeypatch.setattr(wa, "atomic_json_write", fail)
    else:
        monkeypatch.setattr(wa, "_sync_directory", fail)

    result = _tool_write(subsystem)
    assert result["success"] is False, result
    assert not result.get("staged") and not result.get("pending_id")
    assert secret not in json.dumps(result)
    target = _target(approval_home, subsystem)
    assert not target.exists() or not target.read_text()


@pytest.mark.parametrize("subsystem", ["memory", "skills"])
def test_durable_proposal_can_be_approved_after_reload_only_once(approval_home, subsystem):
    from hermes_cli.write_approval_commands import handle_pending_subcommand
    from tools import write_approval as wa
    from tools.memory_tool import load_on_disk_store

    staged = _tool_write(subsystem)
    assert staged["success"] and staged["staged"], staged
    pending_id = staged["pending_id"]
    assert wa.get_pending(subsystem, pending_id)["subsystem"] == subsystem
    assert not _target(approval_home, subsystem).exists() or not _target(approval_home, subsystem).read_text()
    result = handle_pending_subcommand(subsystem, ["approve", pending_id],
                                       memory_store=load_on_disk_store())
    assert "Approved 1" in result, result
    before = _target(approval_home, subsystem).read_bytes()
    result = handle_pending_subcommand(subsystem, ["approve", pending_id],
                                       memory_store=load_on_disk_store())
    assert "Approved 1" not in result
    assert _target(approval_home, subsystem).read_bytes() == before
    assert wa.pending_count(subsystem) == 0
    assert wa._claim_path(subsystem, pending_id).exists()


@pytest.mark.parametrize("fault", ["after_claim", "partial_apply", "cleanup"])
def test_uncertain_attempt_is_retained_and_never_replayed(approval_home, monkeypatch, fault):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa
    from tools.memory_tool import load_on_disk_store

    staged = _tool_write("memory")
    pending_id = staged["pending_id"]
    original_apply = commands._apply_one
    calls = []

    def apply_once(subsystem, rec, store):
        calls.append(rec["id"])
        result = original_apply(subsystem, rec, store)
        if fault == "partial_apply":
            return False, "A later operation failed."
        return result

    monkeypatch.setattr(commands, "_apply_one", apply_once)
    if fault == "after_claim":
        wa.claim_pending("memory", pending_id, operation="approve")
    elif fault == "cleanup":
        original_unlink = Path.unlink

        def fail_pending_unlink(path, *args, **kwargs):
            if path == wa._pending_path("memory", pending_id):
                raise OSError("cleanup failure")
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_pending_unlink)

    first = commands.handle_pending_subcommand("memory", ["approve", pending_id],
                                                memory_store=load_on_disk_store())
    second = commands.handle_pending_subcommand("memory", ["approve", pending_id],
                                                 memory_store=load_on_disk_store())
    assert "Already claimed" in second, (first, second)
    assert len(calls) == (0 if fault == "after_claim" else 1)
    assert wa.pending_count("memory") == 1
    assert wa.list_pending("memory")[0]["claimed"] is True
    assert "claimed" in commands.handle_pending_subcommand("memory", ["pending"])


@pytest.mark.parametrize("other_action", ["approve", "reject"])
def test_concurrent_actions_share_one_durable_claim(approval_home, monkeypatch, other_action):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa
    from tools.memory_tool import load_on_disk_store

    pending_id = _tool_write("memory")["pending_id"]
    entered, release = threading.Event(), threading.Event()
    original_apply = commands._apply_one
    calls, results = [], []

    def blocking_apply(subsystem, record, store):
        calls.append(record["id"])
        entered.set()
        assert release.wait(10), "test failed to release the applying thread"
        return original_apply(subsystem, record, store)

    monkeypatch.setattr(commands, "_apply_one", blocking_apply)
    thread = threading.Thread(target=lambda: results.append(commands.handle_pending_subcommand(
        "memory", ["approve", pending_id], memory_store=load_on_disk_store())))
    thread.start()
    try:
        assert entered.wait(5)
        other = commands.handle_pending_subcommand("memory", [other_action, pending_id],
                                                    memory_store=load_on_disk_store())
        assert "Already claimed" in other, other
        # A fresh interpreter must observe the same claim, not a process-local lock.
        probe = subprocess.run([sys.executable, "-c", (
            "import sys\nfrom tools import write_approval as w\n"
            "try: w.claim_pending('memory', sys.argv[1], operation='approve')\n"
            "except w.PendingWriteClaimed: sys.exit(0)\n"
            "sys.exit(1)\n"), pending_id], capture_output=True, text=True, timeout=10)
        assert probe.returncode == 0, probe.stderr
        assert wa.pending_count("memory") == 1
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert calls == [pending_id]
    assert len(results) == 1 and "Approved 1" in results[0]
    assert wa.pending_count("memory") == 0


@pytest.mark.parametrize("identifier", ["../elsewhere", "", "a" * 33])
def test_claim_identifier_cannot_escape_pending_store(approval_home, identifier):
    from tools import write_approval as wa
    with pytest.raises(ValueError):
        wa.claim_pending("memory", identifier, operation="approve")


@pytest.mark.parametrize("fault", ["claim_file_sync", "claim_directory_sync", "crash_after_claim"])
def test_claim_durability_failure_or_crash_never_reaches_replay(approval_home, monkeypatch, fault):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa
    from tools.memory_tool import load_on_disk_store

    pending_id = _tool_write("memory")["pending_id"]
    claim_path = wa._claim_path("memory", pending_id)
    calls = []
    monkeypatch.setattr(commands, "_apply_one", lambda *args: calls.append(args) or (True, ""))
    with monkeypatch.context() as faults:
        if fault == "crash_after_claim":
            child = subprocess.run([sys.executable, "-c", (
                "import os, sys\nfrom tools import write_approval as w\n"
                "w.claim_pending('memory', sys.argv[1], operation='approve')\n"
                "os._exit(23)\n"), pending_id], capture_output=True, text=True, timeout=10)
            assert child.returncode == 23, child.stderr
        elif fault == "claim_file_sync":
            original_sync = wa.os.fsync

            def fail_claim_sync(fd):
                if claim_path.exists() and os.fstat(fd).st_ino == claim_path.stat().st_ino:
                    raise OSError("claim sync failed")
                return original_sync(fd)

            faults.setattr(wa.os, "fsync", fail_claim_sync)
        else:
            original_directory_sync = wa._sync_directory

            def fail_claim_directory_sync(path):
                if path == claim_path.parent and claim_path.exists():
                    raise OSError("claim directory sync failed")
                return original_directory_sync(path)

            faults.setattr(wa, "_sync_directory", fail_claim_directory_sync)
        commands.handle_pending_subcommand("memory", ["approve", pending_id],
                                           memory_store=load_on_disk_store())
    assert claim_path.exists()
    assert not calls
    for action in ("approve", "reject"):
        result = commands.handle_pending_subcommand("memory", [action, pending_id],
                                                     memory_store=load_on_disk_store())
        assert "Already claimed" in result, result
    assert not calls and wa.pending_count("memory") == 1


def test_apply_reply_does_not_echo_tool_error_content(approval_home, monkeypatch):
    from hermes_cli import write_approval_commands as commands
    from tools import write_approval as wa
    from tools import skill_manager_tool

    secret = "private-parser-source-line"
    record = wa.stage_write("skills", {"action": "create", "name": "probe"},
                            summary="probe", origin="foreground")
    monkeypatch.setattr(skill_manager_tool, "apply_skill_pending", lambda payload:
                        json.dumps({"success": False, "error": secret}))
    result = commands.handle_pending_subcommand("skills", ["approve", record["id"]])
    assert secret not in result
    assert "Approved 0" in result and "Claim retained" in result
