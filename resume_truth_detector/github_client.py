from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

import requests

from .skills import SKILL_LIBRARY


GITHUB_REPO_PATTERN = re.compile(r"github\.com/([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)
SUPPORTIVE_DOMAINS = {
    "huggingface.co",
    "kaggle.com",
    "streamlit.app",
    "colab.research.google.com",
    "vercel.app",
}


@dataclass
class GitHubSnapshot:
    username: str | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    repos: list[dict[str, Any]] = field(default_factory=list)
    project_links: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


class GitHubAnalyzer:
    def __init__(self) -> None:
        self.session = requests.Session()
        token = os.getenv("GITHUB_TOKEN")
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "resume-truth-detector",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def collect(self, username: str | None, project_links: list[str]) -> GitHubSnapshot:
        snapshot = GitHubSnapshot(
            username=username.strip() if username else None,
            project_links=project_links,
        )
        seen_repos: set[str] = set()

        if snapshot.username:
            profile = self._get_json(f"https://api.github.com/users/{snapshot.username}")
            if isinstance(profile, dict) and "error" in profile:
                snapshot.issues.append(profile["error"])
            elif isinstance(profile, dict):
                snapshot.profile = profile

            repos = self._get_json(
                f"https://api.github.com/users/{snapshot.username}/repos?per_page=100&sort=updated"
            )
            if isinstance(repos, dict) and "error" in repos:
                snapshot.issues.append(repos["error"])
            elif isinstance(repos, list):
                for repo in repos:
                    full_name = repo.get("full_name")
                    if full_name and full_name not in seen_repos:
                        snapshot.repos.append(repo)
                        seen_repos.add(full_name)

        for link in project_links:
            match = GITHUB_REPO_PATTERN.search(link)
            if not match:
                continue
            owner, repo_name = match.group(1), match.group(2).replace(".git", "")
            full_name = f"{owner}/{repo_name}"
            if full_name in seen_repos:
                continue
            repo = self._get_json(f"https://api.github.com/repos/{full_name}")
            if isinstance(repo, dict) and "error" not in repo:
                snapshot.repos.append(repo)
                seen_repos.add(full_name)

        return snapshot

    def summarize_skill(self, snapshot: GitHubSnapshot, skill_name: str) -> dict:
        profile = SKILL_LIBRARY[skill_name]
        keywords = {
            *[skill_name.lower()],
            *[item.lower() for item in profile["aliases"]],
            *[item.lower() for item in profile["repo_keywords"]],
            *[item.lower() for item in profile["languages"]],
        }

        relevant_repos: list[dict[str, Any]] = []
        for repo in snapshot.repos:
            haystack = " ".join(
                [
                    str(repo.get("name", "")),
                    str(repo.get("description", "")),
                    " ".join(repo.get("topics", []) or []),
                    str(repo.get("language", "")),
                ]
            ).lower()
            if any(self._contains_keyword(haystack, keyword) for keyword in keywords):
                relevant_repos.append(repo)

        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=180)
        recent_relevant = sum(
            1
            for repo in relevant_repos
            if self._parse_date(repo.get("updated_at")) >= recent_cutoff
        )
        relevant_stars = sum(int(repo.get("stargazers_count", 0) or 0) for repo in relevant_repos)
        profile_score = min(
            100,
            int(snapshot.profile.get("followers", 0) or 0) * 2
            + int(snapshot.profile.get("public_repos", len(snapshot.repos)) or len(snapshot.repos)) * 3
            + (12 if snapshot.profile.get("bio") else 0)
            + (10 if snapshot.profile.get("blog") else 0),
        )

        external_link_hits = 0
        for link in snapshot.project_links:
            domain = (urlparse(link).netloc or "").lower().replace("www.", "")
            if domain in SUPPORTIVE_DOMAINS:
                external_link_hits += 1
            lowered = link.lower()
            if any(self._contains_keyword(lowered, keyword) for keyword in keywords):
                external_link_hits += 1

        return {
            "relevant_repo_count": len(relevant_repos),
            "recent_relevant_repo_count": recent_relevant,
            "relevant_star_count": relevant_stars,
            "profile_score": min(profile_score, 100),
            "external_link_hits": external_link_hits,
            "matched_projects": [repo.get("full_name", repo.get("name", "project")) for repo in relevant_repos[:4]],
            "issues": snapshot.issues,
            "total_repo_count": len(snapshot.repos),
        }

    def _get_json(self, url: str) -> dict[str, Any] | list[dict[str, Any]]:
        try:
            response = self.session.get(url, timeout=8)
            if response.status_code >= 400:
                payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                message = payload.get("message") or f"GitHub returned status {response.status_code}"
                return {"error": message}
            return response.json()
        except requests.RequestException as exc:
            return {"error": f"GitHub lookup unavailable: {exc}"}

    @staticmethod
    def _parse_date(value: str | None) -> datetime:
        if not value:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _contains_keyword(text: str, keyword: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", text.lower()))
