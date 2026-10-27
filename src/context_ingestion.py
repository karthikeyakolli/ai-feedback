"""Product Context Ingestion module with rigorous SSRF protection and safety safeguards.

Supports manual product description, pasted documentation, and safe bounded
public webpage extraction. Preserves source provenance and keeps product context
strictly isolated from feedback clustering features.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import requests
import trafilatura


@dataclass
class ProductContext:
    """Encapsulates product domain grounding and documentation."""

    name: str
    description: str
    target_users: str = ""
    goals_or_concerns: str = ""
    source_type: str = "manual"  # 'manual', 'pasted_docs', 'webpage_url'
    source_url: str | None = None
    extracted_title: str | None = None
    raw_extracted_text: str | None = None
    is_approved: bool = False
    extraction_status: str = "manual"  # 'success', 'partial', 'failed', 'manual'
    extraction_message: str = ""
    retrieved_at: str | None = None
    version: int = 1

    def get_summary_text(self) -> str:
        """Returns consolidated descriptive text for context display."""
        parts = []
        if self.name:
            parts.append(f"Product: {self.name}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.target_users:
            parts.append(f"Target Users: {self.target_users}")
        if self.goals_or_concerns:
            parts.append(f"Goals/Concerns: {self.goals_or_concerns}")
        return "\n".join(parts)


@dataclass
class ExtractionResult:
    """Outcome of a public URL text extraction."""

    url: str
    title: str | None
    extracted_text: str
    status: str  # 'success', 'partial', 'failed'
    message: str
    status_code: int | None = None


# Restricted private, loopback, link-local, and reserved IP networks
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    # IPv6 blocks
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("ff00::/8"),
]


def is_ip_allowed(ip_str: str) -> tuple[bool, str]:
    """Validates whether an IP address belongs to allowed public routable address space."""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return False, f"Invalid IP address format: {ip_str}"

    if ip_obj.is_loopback:
        return False, "Access to loopback addresses is forbidden."
    if ip_obj.is_private:
        return False, "Access to private network addresses is forbidden."
    if ip_obj.is_reserved:
        return False, "Access to reserved network addresses is forbidden."
    if ip_obj.is_link_local:
        return False, "Access to link-local addresses is forbidden."
    if ip_obj.is_multicast:
        return False, "Access to multicast addresses is forbidden."
    if not ip_obj.is_global:
        return False, "Access to non-global IP addresses is forbidden."

    for net in _BLOCKED_NETWORKS:
        if ip_obj in net:
            return False, f"IP address {ip_str} belongs to blocked subnet {net}."

    return True, "IP is valid and public."


def validate_url_safety(url: str) -> tuple[bool, str, str | None]:
    """Validates URL scheme, resolves hostname, and checks against SSRF destinations.

    Returns: (is_safe, error_or_success_message, resolved_ip)
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string.", None

    url_clean = url.strip()
    try:
        parsed = urlparse(url_clean)
    except Exception as exc:
        return False, f"Malformed URL: {exc}", None

    # Enforce scheme
    if parsed.scheme.lower() not in {"http", "https"}:
        return False, f"Unsupported URL scheme '{parsed.scheme}'. Only http and https are permitted.", None

    hostname = parsed.hostname
    if not hostname:
        return False, "URL does not contain a valid hostname.", None

    # Check for direct localhost aliases
    if hostname.lower() in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "local"}:
        return False, "Access to localhost or loopback hostnames is forbidden.", None

    # Resolve hostname via DNS
    try:
        addr_info = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
        if not addr_info:
            return False, f"Could not resolve hostname '{hostname}'.", None
    except socket.gaierror as exc:
        return False, f"DNS resolution failed for '{hostname}': {exc}", None

    # Verify every resolved IP
    primary_ip = None
    for entry in addr_info:
        sockaddr = entry[4]
        ip_candidate = sockaddr[0]
        allowed, reason = is_ip_allowed(ip_candidate)
        if not allowed:
            return False, f"Destination resolved to restricted IP ({ip_candidate}): {reason}", None
        if not primary_ip:
            primary_ip = ip_candidate

    return True, "URL destination verified safe.", primary_ip


def extract_webpage_content(
    url: str,
    timeout_seconds: int = 10,
    max_response_bytes: int = 2 * 1024 * 1024,  # 2MB
    max_redirects: int = 3,
) -> ExtractionResult:
    """Safely fetches a public webpage and extracts readable text with SSRF guardrails.

    Validates initial URL and re-validates DNS for every redirect hop.
    Enforces response size limits, timeouts, and content-type checks.
    """
    current_url = url.strip()

    for hop in range(max_redirects + 1):
        is_safe, msg, _ = validate_url_safety(current_url)
        if not is_safe:
            return ExtractionResult(
                url=url,
                title=None,
                extracted_text="",
                status="failed",
                message=f"Security check failed on hop {hop}: {msg}",
            )

        try:
            # Request headers identifying standard benign fetcher
            headers = {
                "User-Agent": "AIProductIntelligenceAnalyzer/1.0 (Public Research Bot; Python)",
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
            }
            resp = requests.get(
                current_url,
                headers=headers,
                timeout=timeout_seconds,
                allow_redirects=False,
                stream=True,
            )

            # Check for redirect
            if resp.is_redirect or resp.status_code in {301, 302, 303, 307, 308}:
                redirect_url = resp.headers.get("Location")
                if not redirect_url:
                    return ExtractionResult(
                        url=url,
                        title=None,
                        extracted_text="",
                        status="failed",
                        message=f"Received redirect status {resp.status_code} but missing Location header.",
                        status_code=resp.status_code,
                    )
                # Resolve relative redirects
                from urllib.parse import urljoin
                current_url = urljoin(current_url, redirect_url)
                continue

            if resp.status_code != 200:
                return ExtractionResult(
                    url=url,
                    title=None,
                    extracted_text="",
                    status="failed",
                    message=f"HTTP request returned non-200 status code: {resp.status_code}",
                    status_code=resp.status_code,
                )

            # Validate Content-Type
            content_type = resp.headers.get("Content-Type", "").lower()
            if not any(t in content_type for t in ["text/html", "text/plain", "application/xhtml+xml"]):
                return ExtractionResult(
                    url=url,
                    title=None,
                    extracted_text="",
                    status="failed",
                    message=f"Unsupported Content-Type '{content_type}'. Only HTML or plain text are permitted.",
                    status_code=resp.status_code,
                )

            # Bounded content read
            content_chunks = []
            total_bytes = 0
            for chunk in resp.iter_content(chunk_size=8192):
                total_bytes += len(chunk)
                if total_bytes > max_response_bytes:
                    return ExtractionResult(
                        url=url,
                        title=None,
                        extracted_text="",
                        status="failed",
                        message=f"Webpage response exceeded maximum limit of {max_response_bytes} bytes.",
                        status_code=resp.status_code,
                    )
                content_chunks.append(chunk)

            raw_bytes = b"".join(content_chunks)
            encoding = resp.encoding or "utf-8"
            html_text = raw_bytes.decode(encoding, errors="replace")

            # Extract title using regex before stripping
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
            extracted_title = title_match.group(1).strip() if title_match else None

            # Extract clean main text using trafilatura
            extracted_text = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )

            if not extracted_text:
                # Fallback: simple tag removal if trafilatura finds no main article
                clean_simple = re.sub(r"<[^>]+>", " ", html_text)
                clean_simple = re.sub(r"\s+", " ", clean_simple).strip()
                if len(clean_simple) > 100:
                    extracted_text = clean_simple[:2000]
                    return ExtractionResult(
                        url=url,
                        title=extracted_title,
                        extracted_text=extracted_text,
                        status="partial",
                        message="Trafilatura found no primary article body; extracted general page text.",
                        status_code=resp.status_code,
                    )
                return ExtractionResult(
                    url=url,
                    title=extracted_title,
                    extracted_text="",
                    status="failed",
                    message="Webpage contained insufficient readable text content.",
                    status_code=resp.status_code,
                )

            return ExtractionResult(
                url=url,
                title=extracted_title,
                extracted_text=extracted_text.strip(),
                status="success",
                message="Successfully extracted readable content.",
                status_code=resp.status_code,
            )

        except requests.exceptions.Timeout:
            return ExtractionResult(
                url=url,
                title=None,
                extracted_text="",
                status="failed",
                message=f"HTTP request timed out after {timeout_seconds} seconds.",
            )
        except requests.exceptions.RequestException as exc:
            return ExtractionResult(
                url=url,
                title=None,
                extracted_text="",
                status="failed",
                message=f"Network error during extraction: {exc}",
            )

    return ExtractionResult(
        url=url,
        title=None,
        extracted_text="",
        status="failed",
        message=f"Exceeded maximum redirect limit of {max_redirects} hops.",
    )
