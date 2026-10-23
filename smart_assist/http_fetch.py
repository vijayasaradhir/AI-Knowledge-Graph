from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Sequence
from urllib import error, request


@dataclass
class FetchResult:
    stdout: str
    stderr: str
    returncode: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-assist-http-fetch",
        description="Fetch an HTTP URL and print the response body.",
    )
    parser.add_argument("url", help="HTTP or HTTPS URL to fetch")
    parser.add_argument(
        "--method",
        default="GET",
        help="HTTP method to use (default: GET)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)",
    )
    return parser


def _decode_body(body: bytes, headers: object | None = None) -> str:
    charset = "utf-8"
    if headers is not None:
        get_content_charset = getattr(headers, "get_content_charset", None)
        if callable(get_content_charset):
            detected = get_content_charset()
            if detected:
                charset = detected
    return body.decode(charset, errors="replace")


def _fetch_with_urllib(url: str, method: str, timeout: int) -> FetchResult:
    req = request.Request(url, method=method)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            return FetchResult(
                stdout=_decode_body(body, response.headers),
                stderr="",
                returncode=0,
            )
    except error.HTTPError as exc:
        body = exc.read() if hasattr(exc, "read") else b""
        return FetchResult(
            stdout=_decode_body(body, exc.headers if hasattr(exc, "headers") else None),
            stderr=f"HTTP {exc.code} {exc.reason}",
            returncode=1,
        )
    except error.URLError as exc:
        reason = exc.reason if getattr(exc, "reason", None) else str(exc)
        return FetchResult(stdout="", stderr=str(reason), returncode=1)


def _fetch_with_winhttp(url: str, method: str, timeout: int) -> FetchResult | None:
    if os.name != "nt":
        return None

    try:
        import win32com.client  # type: ignore[import-not-found]
    except Exception:
        return None

    client = None

    try:
        client = win32com.client.Dispatch("WinHttp.WinHttpRequest.5.1")
        set_timeouts = getattr(client, "SetTimeouts", None)
        if callable(set_timeouts):
            timeout_ms = max(timeout, 1) * 1000
            set_timeouts(timeout_ms, timeout_ms, timeout_ms, timeout_ms)

        set_auto_logon_policy = getattr(client, "SetAutoLogonPolicy", None)
        if callable(set_auto_logon_policy):
            # 0 == always use the current user's default logon credentials.
            set_auto_logon_policy(0)

        client.Open(method, url, False)
        client.Send()

        body = getattr(client, "ResponseText", "") or ""
        return FetchResult(stdout=body, stderr="", returncode=0)
    except Exception as exc:
        # Best effort: if WinHTTP returned an HTTP error, preserve the response body
        # and surface the status code instead of failing silently.
        status = None
        status_text = None
        body = ""

        if client is not None:
            try:
                status = int(getattr(client, "Status"))
            except Exception:
                pass

            try:
                status_text = str(getattr(client, "StatusText"))
            except Exception:
                pass

            try:
                body = getattr(client, "ResponseText", "") or ""
            except Exception:
                pass

        if status is not None:
            return FetchResult(
                stdout=body,
                stderr=f"HTTP {status} {status_text or ''}".rstrip(),
                returncode=1,
            )

        return FetchResult(stdout=body, stderr=str(exc), returncode=1)


def fetch(url: str, method: str = "GET", timeout: int = 30) -> FetchResult:
    method = method.upper()

    # Prefer Windows integrated authentication when available, then fall back to the
    # standard library HTTP client.
    winhttp_result = _fetch_with_winhttp(url, method, timeout)
    if winhttp_result is not None:
        return winhttp_result

    return _fetch_with_urllib(url, method, timeout)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = fetch(args.url, method=args.method, timeout=args.timeout)

    if result.stdout:
        sys.stdout.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stdout.write("\n")

    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
