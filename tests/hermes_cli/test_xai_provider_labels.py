"""Regression tests for xAI provider label disambiguation."""

from hermes_cli.models import provider_label
from hermes_cli.providers import get_label
from agent.models_dev import ProviderInfo


def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label(monkeypatch):
    """The model picker must distinguish xAI API-key and OAuth providers."""
    def provider_info(provider):
        assert provider == "xai", f"unexpected provider lookup: {provider}"
        return ProviderInfo(id="xai", name="xAI", env=("XAI_API_KEY",), api="https://api.x.ai/v1")

    monkeypatch.setattr("agent.models_dev.get_provider_info", provider_info)
    assert get_label("xai") == "xAI"
    assert get_label("xai-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"
    assert get_label("grok-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"

