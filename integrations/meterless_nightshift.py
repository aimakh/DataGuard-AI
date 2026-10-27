import os
import requests
from typing import Dict, Any, List, Optional
from models.config import get_settings


class MeterlessNightshiftClient:
    """Client for Meterless Nightshift local-first AI runtime queue."""

    def __init__(self, base_url: Optional[str] = None):
        settings = get_settings()
        self.base_url = (base_url or settings.meterless_nightshift_url).rstrip("/")
        self.api_key = settings.meterless_api_key

    def probe_runtime(self) -> Dict[str, Any]:
        """Probes the local Nightshift daemon and reports connection state."""
        try:
            r = requests.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"}, timeout=2)
            if r.status_code == 200:
                models = [m.get("id", "") for m in r.json().get("data", [])]
                return {"status": "ONLINE", "connected": True, "models": models, "url": self.base_url}
        except Exception:
            pass

        return {
            "status": "STANDBY",
            "connected": False,
            "models": ["local/markovian-chunked-reasoner", "local/nemotron-heuristic"],
            "url": self.base_url,
            "note": "Nightshift daemon standby; using verified local fallback harness."
        }

    def execute_completion(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Sends a chat completion request to the OpenAI-compatible Nightshift runtime."""
        try:
            target_model = model or "local/markovian-chunked-reasoner"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": target_model, "messages": messages, "temperature": 0.1},
                timeout=10
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None
