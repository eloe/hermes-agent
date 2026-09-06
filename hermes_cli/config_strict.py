"""Uncached policy reads for writes that must not use last-known-good configuration."""

import copy


def load_strict_config() -> dict:
    """Read each input once, then use the normal loader's merge/normalization pipeline.

    This does not lock out external editors or transact the two files together. It
    guarantees that a failed read cannot become defaults or a cached policy decision.
    """
    from hermes_cli import config as cfg

    with cfg._CONFIG_LOCK:
        user = cfg.require_readable_config_before_write(cfg.get_config_path())
        managed_dir = cfg.managed_scope.get_managed_dir()
        managed = (cfg.require_readable_config_before_write(managed_dir / "config.yaml")
                   if managed_dir is not None else {})
        merged = cfg._merge_user_config(copy.deepcopy(cfg.DEFAULT_CONFIG), user)
        normalized = cfg._canonicalize_config(merged)
        expanded, _ = cfg._merge_managed_overlay(
            cfg._expand_env_vars(normalized), managed_config=managed)
        return expanded
