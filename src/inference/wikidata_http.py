# coding: utf-8
"""Small HTTP helper for Wikidata requests.

We keep it dependency-free (stdlib only) to avoid adding requirements.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


class WikidataHTTPError(RuntimeError):
    pass


def _build_opener() -> urllib.request.OpenerDirector:
    """Build an opener honoring proxy env vars.

    Many environments in CN need an HTTP(S) proxy to reach wikidata.org.
    We support standard env vars: HTTPS_PROXY / HTTP_PROXY / ALL_PROXY.
    """

    proxies: Dict[str, str] = {}
    https_p = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    http_p = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    all_p = os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")

    if all_p:
        proxies.setdefault("http", all_p)
        proxies.setdefault("https", all_p)
    if http_p:
        proxies["http"] = http_p
    if https_p:
        proxies["https"] = https_p

    if proxies:
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    return urllib.request.build_opener()


def get_json(
    url: str,
    *,
    timeout_s: float = 10.0,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 2,
    retry_backoff_s: float = 0.6,
) -> Any:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    # Wikidata strongly prefers a real UA; some networks/proxies also block empty UA.
    req.add_header("User-Agent", "ConvQA-new/0.1 (local inference; contact: none)")
    for k, v in (headers or {}).items():
        req.add_header(str(k), str(v))

    opener = _build_opener()

    last_err: Optional[Exception] = None
    for i in range(max(1, int(retries) + 1)):
        try:
            with opener.open(req, timeout=float(timeout_s)) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except Exception as e:  # pragma: no cover
            last_err = e
            # small backoff then retry
            if i < max(1, int(retries) + 1) - 1:
                time.sleep(float(retry_backoff_s) * (1.5**i))
                continue

            code = getattr(e, "code", None)
            reason = getattr(e, "reason", None)
            body = ""
            try:
                if hasattr(e, "read"):
                    body = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                body = ""

            extra = ""
            if code is not None:
                extra += f" status={code}"
            if reason is not None:
                extra += f" reason={reason}"
            if body:
                extra += f" body={body!r}"

            raise WikidataHTTPError(f"HTTP GET failed: {url} ({e}){extra}") from e

    raise WikidataHTTPError(f"HTTP GET failed: {url} ({last_err})")


def build_url(base: str, params: Dict[str, Any]) -> str:
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return f"{base}?{q}"
