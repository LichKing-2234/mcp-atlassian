"""Agora OAuth token refresh utility.

Reuses OAuthConfig for token exchange and refresh logic where possible,
adding Agora-specific features:
- Password grant (headless, via username/password)
- Authorization code grant (browser-based SSO via redirect)
- Background auto-refresh thread with configurable interval

The public API (get_token, start_auto_refresh, create_agora_token_refresher)
is kept stable so callers (jira/client.py, confluence/client.py) need no changes.
"""

import http.server
import logging
import os
import secrets
import socketserver
import threading
import time
import urllib.parse
import webbrowser

import requests

from .oauth import OAuthConfig

logger = logging.getLogger("mcp-atlassian.agora-oauth")

# Default auto-refresh interval in seconds (~2 hours, well within typical
# 2-hour token lifetime).
_DEFAULT_REFRESH_INTERVAL = 7000


class _AutoRefreshMixin:
    """Mixin that adds a background auto-refresh thread.

    Subclasses must implement ``_do_refresh(self) -> None``.
    """

    refresh_interval: int
    _lock: threading.Lock
    _refresh_thread: threading.Thread | None
    _stop_refresh: bool

    def _init_refresh_state(self) -> None:
        self._lock = threading.Lock()
        self._refresh_thread = None
        self._stop_refresh = False

    # -- to be implemented by concrete classes ------------------------------

    def _do_refresh(self) -> None:
        """Refresh the token. Must be overridden by subclasses."""
        return  # pragma: no cover

    # -- public API ---------------------------------------------------------

    def start_auto_refresh(self) -> None:
        """Start background thread to auto-refresh token."""
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
        self._stop_refresh = False
        self._refresh_thread = threading.Thread(
            target=self._auto_refresh_loop, daemon=True
        )
        self._refresh_thread.start()
        logger.info(
            "Started auto-refresh thread (interval: %ds)",
            self.refresh_interval,
        )

    def stop(self) -> None:
        """Stop auto-refresh thread."""
        self._stop_refresh = True
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)

    # -- internal -----------------------------------------------------------

    def _auto_refresh_loop(self) -> None:
        """Background loop to refresh token periodically."""
        while True:
            time.sleep(self.refresh_interval)
            if self._stop_refresh:
                break
            with self._lock:
                try:
                    self._do_refresh()
                except (requests.RequestException, KeyError):
                    logger.exception("Auto-refresh failed")


class AgoraTokenRefresher(_AutoRefreshMixin):
    """Auto-refresh OAuth tokens for Agora services (password grant).

    Delegates token exchange to an internal ``OAuthConfig`` instance,
    adding password-grant support and a background refresh loop.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        username: str,
        password: str,
        refresh_interval: int = _DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        self.username = username
        self.password = password
        self.refresh_interval = refresh_interval

        # Build an OAuthConfig pointing at the Agora token endpoint.
        # We set base_url to a non-Atlassian URL so ``is_data_center``
        # returns True and ``token_url`` resolves via the DC path.
        # However, OAuthConfig hard-codes the DC path suffix, so we
        # override the property below.
        self._oauth = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri="",
            scope="",
            base_url=None,
        )
        # Store the raw Agora token URL — OAuthConfig's property logic
        # is Atlassian-specific, so we keep our own.
        self._token_url = token_url

        self._init_refresh_state()

    def get_token(self) -> str:
        """Get current valid token, refresh if needed."""
        with self._lock:
            if not self._oauth.access_token:
                self._do_refresh()
            return self._oauth.access_token or ""

    def _do_refresh(self) -> None:
        """Refresh the OAuth token."""
        # Try refresh_token grant first if available
        if self._oauth.refresh_token:
            try:
                self._token_request(
                    grant_type="refresh_token",
                    refresh_token=self._oauth.refresh_token,
                )
                return
            except (requests.RequestException, KeyError):
                logger.warning(
                    "Refresh token failed, falling back to password grant"
                )

        # Fallback to password grant
        self._token_request(
            grant_type="password",
            username=self.username,
            password=self.password,
        )

    def _token_request(self, **params: str) -> None:
        """Send a token request and store the result in _oauth."""
        response = requests.post(
            self._token_url,
            auth=(self._oauth.client_id, self._oauth.client_secret),
            data=params,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        self._oauth.access_token = data["access_token"]
        self._oauth.refresh_token = data.get(
            "refresh_token", self._oauth.refresh_token
        )
        self._oauth.expires_at = time.time() + data.get(
            "expires_in", 3600
        )
        logger.info(
            "Refreshed Agora OAuth token via %s grant",
            params.get("grant_type", "unknown"),
        )


# ---------------------------------------------------------------------------
# Browser-based Authorization Code flow
# ---------------------------------------------------------------------------


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Captures the OAuth authorization code from the redirect."""

    authorization_code: str | None = None
    received_state: str | None = None
    error: str | None = None
    done = threading.Event()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/favicon.ico":
            self.send_error(404)
            return

        params = urllib.parse.parse_qs(parsed.query)

        if "error" in params:
            _CallbackHandler.error = params["error"][0]
            self._respond(
                f"Authorization failed: {_CallbackHandler.error}",
                status=400,
            )
        elif "code" in params:
            _CallbackHandler.authorization_code = params["code"][0]
            _CallbackHandler.received_state = params.get(
                "state", [None]
            )[0]
            self._respond(
                "Authorization successful! You can close this window."
            )
        else:
            self._respond("Missing authorization code.", status=400)

        _CallbackHandler.done.set()

    def _respond(self, message: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = (
            "<!DOCTYPE html><html><body style='font-family:sans-serif;"
            "text-align:center;padding:40px'>"
            f"<p>{message}</p>"
            "<script>setTimeout(()=>window.close(),3000)</script>"
            "</body></html>"
        )
        self.wfile.write(html.encode())

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default request logging."""


class AgoraBrowserOAuthRefresher(_AutoRefreshMixin):
    """OAuth token refresher using browser-based authorization code flow.

    Delegates token storage to an internal ``OAuthConfig``, reusing its
    ``exchange_code_for_tokens`` pattern while keeping the local callback
    server for capturing the authorization code.
    """

    def __init__(
        self,
        authorize_url: str,
        token_url: str,
        client_id: str,
        client_secret: str = "",
        redirect_uri: str = "http://localhost:18082",
        scope: str = "read",
        refresh_interval: int = _DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        self.refresh_interval = refresh_interval
        self._authorize_url = authorize_url
        self._token_url = token_url

        self._oauth = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            base_url=None,
        )

        self._init_refresh_state()

    # -- public API ---------------------------------------------------------

    def authorize(self, timeout: int = 300) -> str:
        """Run the full browser authorization flow (blocking).

        Args:
            timeout: Max seconds to wait for the user to complete login.

        Returns:
            The access token.

        Raises:
            RuntimeError: If authorization fails or times out.
        """
        _CallbackHandler.authorization_code = None
        _CallbackHandler.received_state = None
        _CallbackHandler.error = None
        _CallbackHandler.done.clear()

        parsed = urllib.parse.urlparse(self._oauth.redirect_uri)
        port = parsed.port or 18082

        httpd = socketserver.TCPServer(("", port), _CallbackHandler)
        server_thread = threading.Thread(
            target=httpd.serve_forever, daemon=True
        )
        server_thread.start()

        state = secrets.token_urlsafe(16)
        auth_params = urllib.parse.urlencode({
            "response_type": "code",
            "client_id": self._oauth.client_id,
            "redirect_uri": self._oauth.redirect_uri,
            "scope": self._oauth.scope,
            "state": state,
        })
        url = f"{self._authorize_url}?{auth_params}"

        logger.info("Opening browser for Agora SSO authorization...")
        webbrowser.open(url)
        logger.info("If the browser did not open, visit: %s", url)

        try:
            self._wait_for_callback(timeout, state)
            return self._oauth.access_token or ""
        finally:
            httpd.shutdown()

    def _wait_for_callback(
        self, timeout: int, expected_state: str
    ) -> None:
        """Block until the callback arrives and exchange the code.

        Raises:
            RuntimeError: On timeout, error, state mismatch, or
                missing code.
        """
        if not _CallbackHandler.done.wait(timeout=timeout):
            msg = "Timed out waiting for authorization callback"
            raise RuntimeError(msg)

        error = _CallbackHandler.error
        if error:
            msg = f"Authorization error: {error}"
            raise RuntimeError(msg)

        if _CallbackHandler.received_state != expected_state:
            msg = "State mismatch — possible CSRF attack"
            raise RuntimeError(msg)

        code = _CallbackHandler.authorization_code
        if not code:
            msg = "No authorization code received"
            raise RuntimeError(msg)

        self._exchange_code(code)

    def get_token(self) -> str:
        """Get current valid token."""
        with self._lock:
            if not self._oauth.access_token:
                msg = "No token available. Call authorize() first."
                raise RuntimeError(msg)
            return self._oauth.access_token

    # -- internal -----------------------------------------------------------

    def _exchange_code(self, code: str) -> None:
        """Exchange authorization code for tokens."""
        payload = {
            "grant_type": "authorization_code",
            "client_id": self._oauth.client_id,
            "client_secret": self._oauth.client_secret,
            "code": code,
            "redirect_uri": self._oauth.redirect_uri,
        }
        logger.debug(
            "Token exchange request to %s with params: %s",
            self._token_url,
            {k: v for k, v in payload.items() if k != "code"},
        )
        response = requests.post(
            self._token_url,
            data=payload,
            timeout=10,
        )
        if not response.ok:
            logger.error(
                "Token exchange failed (%s): %s",
                response.status_code,
                response.text,
            )
        response.raise_for_status()
        data = response.json()
        self._oauth.access_token = data["access_token"]
        self._oauth.refresh_token = data.get("refresh_token")
        self._oauth.expires_at = time.time() + data.get(
            "expires_in", 3600
        )
        logger.info(
            "Obtained Agora OAuth token via authorization_code grant"
        )

    def _do_refresh(self) -> None:
        """Refresh the token using the stored refresh token."""
        if not self._oauth.refresh_token:
            logger.warning(
                "No refresh token available, cannot auto-refresh"
            )
            return
        response = requests.post(
            self._token_url,
            data={
                "grant_type": "refresh_token",
                "client_id": self._oauth.client_id,
                "client_secret": self._oauth.client_secret,
                "refresh_token": self._oauth.refresh_token,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        self._oauth.access_token = data["access_token"]
        self._oauth.refresh_token = data.get(
            "refresh_token", self._oauth.refresh_token
        )
        self._oauth.expires_at = time.time() + data.get(
            "expires_in", 3600
        )
        logger.info("Refreshed Agora OAuth token via refresh_token")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

_RefresherType = AgoraTokenRefresher | AgoraBrowserOAuthRefresher


def create_agora_token_refresher() -> _RefresherType | None:
    """Create token refresher from environment variables.

    Checks AGORA_OAUTH_GRANT_TYPE (default: "password"):
      - "password"            → AgoraTokenRefresher
      - "authorization_code"  → AgoraBrowserOAuthRefresher
    """
    grant_type = os.getenv(
        "AGORA_OAUTH_GRANT_TYPE", "password"
    ).lower()
    token_url = os.getenv("AGORA_OAUTH_TOKEN_URL")
    client_id = os.getenv("AGORA_OAUTH_CLIENT_ID")
    client_secret = os.getenv("AGORA_OAUTH_CLIENT_SECRET")
    # Agora OAuth base — derive authorize/token URLs if not set
    oauth_base = os.getenv("AGORA_OAUTH_BASE_URL", "").rstrip("/")
    if not token_url and oauth_base:
        token_url = f"{oauth_base}/token"

    if grant_type == "authorization_code":
        authorize_url = os.getenv("AGORA_OAUTH_AUTHORIZE_URL")
        if not authorize_url and oauth_base:
            authorize_url = f"{oauth_base}/authorize"
        redirect_uri = os.getenv(
            "AGORA_OAUTH_REDIRECT_URI",
            "http://localhost:18082",
        )
        scope = os.getenv("AGORA_OAUTH_SCOPE", "read")

        if not client_id:
            logger.warning(
                "Missing AGORA_OAUTH_CLIENT_ID for "
                "Agora browser OAuth"
            )
            return None

        if not authorize_url or not token_url:
            logger.warning(
                "Missing authorize/token URL for "
                "Agora browser OAuth"
            )
            return None

        browser_refresher = AgoraBrowserOAuthRefresher(
            authorize_url=authorize_url,
            token_url=token_url,
            client_id=str(client_id),
            client_secret=str(client_secret or ""),
            redirect_uri=redirect_uri,
            scope=scope,
        )
        browser_refresher.authorize()
        browser_refresher.start_auto_refresh()
        return browser_refresher

    # Default: password grant
    username = os.getenv("AGORA_OAUTH_USERNAME")
    password = os.getenv("AGORA_OAUTH_PASSWORD")

    if not all(
        [client_id, client_secret, username, password, token_url]
    ):
        return None

    refresher = AgoraTokenRefresher(
        token_url=str(token_url),
        client_id=str(client_id),
        client_secret=str(client_secret),
        username=str(username),
        password=str(password),
    )
    refresher.get_token()
    refresher.start_auto_refresh()
    return refresher


# Global token refresher instance
_agora_token_refresher: _RefresherType | None = None


def get_agora_token() -> str | None:
    """Get current Agora OAuth token."""
    global _agora_token_refresher

    if _agora_token_refresher is None:
        _agora_token_refresher = create_agora_token_refresher()

    if _agora_token_refresher:
        return _agora_token_refresher.get_token()

    return None


def patch_session(session: requests.Session) -> bool:
    """Patch a requests Session to inject Agora OAuth headers.

    Monkey-patches ``session.send`` so that every outgoing request
    carries the latest ``accessToken`` (and optional service
    credentials).  The original ``send`` is preserved and called
    transparently.

    Returns:
        True if the session was patched, False if Agora OAuth is
        not configured.
    """
    global _agora_token_refresher

    if _agora_token_refresher is None:
        _agora_token_refresher = create_agora_token_refresher()

    if _agora_token_refresher is None:
        return False

    refresher = _agora_token_refresher
    svc_user = os.getenv("AGORA_OAUTH_USERNAME", "")
    svc_pass = os.getenv("AGORA_OAUTH_PASSWORD", "")
    original_send = session.send

    def _send_with_agora(
        request: requests.PreparedRequest,
        **kwargs: object,
    ) -> requests.Response:
        token = refresher.get_token()
        if token:
            request.headers["accessToken"] = token
        if svc_user:
            request.headers["agora-service-user"] = svc_user
        if svc_pass:
            request.headers["agora-service-password"] = svc_pass
        return original_send(request, **kwargs)  # type: ignore[arg-type]

    session.send = _send_with_agora  # type: ignore[assignment]
    logger.info("Patched session with Agora OAuth header injection")
    return True
