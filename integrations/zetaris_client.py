import os
import uuid
import requests
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from models.config import get_settings


class ZetarisClient:
    """Enterprise client for authenticating and querying the Zetaris Network Data Platform."""

    def __init__(self, ui_url: Optional[str] = None, api_url: Optional[str] = None):
        settings = get_settings()
        self.ui_url = (ui_url or settings.zetaris_ui_url).rstrip("/")
        self.api_url = (api_url or settings.zetaris_api_url).rstrip("/")
        self.email = settings.zetaris_admin_email
        self.password = settings.zetaris_admin_password
        self.default_timeout = getattr(settings, "zetaris_query_timeout", 45)
        self.token: Optional[str] = None
        self.org_id: Optional[str] = "1"

    def login(self) -> bool:
        """Authenticates with the Zetaris platform via the UI auth proxy."""
        endpoint = f"{self.ui_url}/api/auth/login"
        payload = {"username": self.email, "password": self.password}
        try:
            resp = requests.post(endpoint, json=payload, timeout=10)
            if resp.status_code in (200, 201):
                data = resp.json()
                self.token = data.get("idToken") or data.get("accessToken") or data.get("token")
                self.org_id = str(data.get("orgId", "1"))
                return bool(self.token)
            return False
        except Exception:
            return False

    def check_health(self) -> Dict[str, Any]:
        """Probes the live Zetaris UI and API health endpoints."""
        ui_ok = False
        api_ok = False
        try:
            r = requests.get(f"{self.ui_url}/api/config", timeout=5)
            ui_ok = (r.status_code == 200)
        except Exception:
            pass

        try:
            r = requests.get(f"{self.api_url}/health", timeout=5)
            api_ok = (r.status_code in (200, 404))
        except Exception:
            pass

        return {
            "ui_online": ui_ok,
            "api_online": api_ok,
            "connected": ui_ok or api_ok,
            "ui_url": self.ui_url,
            "api_url": self.api_url
        }

    def execute_query(self, sql: str, limit: int = 100, timeout: Optional[int] = None) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """Executes a SQL query through the Zetaris Spark Query Proxy."""
        if not self.token:
            if not self.login():
                return False, None, "Zetaris login failed. Verify platform is running."

        endpoint = f"{self.ui_url}/api/proxy/query/sql/start"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Org-ID": self.org_id or "1",
            "X-Request-ID": str(uuid.uuid4()),
            "Content-Type": "application/json"
        }
        
        sql_clean = sql.strip().rstrip(";")
        if "limit" not in sql_clean.lower():
            sql_clean += f" LIMIT {limit}"

        payload = {"select": sql_clean, "pageLimit": limit}
        query_timeout = timeout if timeout is not None else self.default_timeout

        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=query_timeout)
            
            # Re-authenticate once if token expired
            if resp.status_code == 401:
                if self.login():
                    headers["Authorization"] = f"Bearer {self.token}"
                    resp = requests.post(endpoint, json=payload, headers=headers, timeout=query_timeout)

            if resp.status_code == 200:
                data = resp.json()
                rows = data.get("records") or data.get("data") or []
                if isinstance(rows, list):
                    df = pd.DataFrame(rows)
                    return True, df, f"Successfully retrieved {len(df)} rows from Zetaris Spark engine."
                return True, pd.DataFrame(), "Query executed; 0 rows returned."
            return False, None, f"Query proxy returned status {resp.status_code}: {resp.text[:120]}"
        except requests.exceptions.Timeout:
            return False, None, f"Zetaris Spark query timed out after {query_timeout}s. Distributed cluster scans need more time or a smaller LIMIT."
        except Exception as e:
            return False, None, f"Zetaris query error: {str(e)}"

    # Alias for health check
    health_check = check_health
