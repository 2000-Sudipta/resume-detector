from __future__ import annotations

import re
from statistics import mean

from .github_client import GitHubAnalyzer
from .modeling import predict_realism_probability
from .skills import detect_claim_strength, find_skills_in_text, keyword_hits, normalize_skill_name


ACTION_TERMS = [
    "built",
    "deployed",
    "implemented",
    "trained",
    "fine-tuned",
    "optimized",
    "shipped",
    "designed",
    "automated",
]
PROJECT_TERMS = ["project", "projects", "portfolio", "case study", "hackathon", "internship", "research"]
CERTIFICATION_TERMS = ["certified", "certificate", "coursera", "udemy", "aws", "azure", "gcp", "nptel"]
YEARS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", re.IGNORECASE)
CLAIM_SPLIT_PATTERN = re.compile(r"[\n,;]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n")


class ResumeTruthDetector:
    def __init__(self, model_bundle: dict) -> None:
        self.model_bundle = model_bundle
        self.github = GitHubAnalyzer()

    def analyze(
        self,
        resume_text: str,
        claimed_skills_text: str,
        github_username: str | None,
        project_links_text: str,
    ) -> dict:
        project_links = [item.strip() for item in CLAIM_SPLIT_PATTERN.split(project_links_text) if item.strip()]
        claims = self._extract_claims(resume_text, claimed_skills_text)
        github_snapshot = self.github.collect(github_username, project_links)

        assessments: list[dict] = []
        realism_scores: list[float] = []

        for claim in claims:
            github_metrics = self.github.summarize_skill(github_snapshot, claim["skill_area"])
            feature_row = self._build_feature_row(
                resume_text=resume_text,
                claim_level=claim["claim_level"],
                skill_area=claim["skill_area"],
                github_metrics=github_metrics,
            )
            realism_probability = predict_realism_probability(self.model_bundle, feature_row)
            realism_scores.append(realism_probability)
            assessments.append(
                self._format_claim_result(
                    claim=claim,
                    feature_row=feature_row,
                    github_metrics=github_metrics,
                    realism_probability=realism_probability,
                )
            )

        overall_realism = mean(realism_scores) if realism_scores else 0.5
        overall_label, overall_message = self._overall_verdict(overall_realism)
        dominant_risks = self._top_risks(assessments)

        return {
            "overall": {
                "label": overall_label,
                "message": overall_message,
                "realism_score": round(overall_realism * 100, 1),
                "exaggeration_risk": round((1 - overall_realism) * 100, 1),
            },
            "claims": assessments,
            "github": {
                "username": github_snapshot.username,
                "issues": github_snapshot.issues,
                "repo_count": len(github_snapshot.repos),
                "link_count": len(project_links),
            },
            "resume_summary": self._resume_summary(resume_text),
            "dominant_risks": dominant_risks,
            "flow": [
                "Resume text is parsed to detect claimed technical skills and intensity words like expert or proficient.",
                "For each skill, the analyzer measures evidence in the resume, projects, and GitHub footprint.",
                "A scikit-learn classifier compares the evidence pattern against synthetic labeled examples and returns realism probability.",
                "The UI converts the score into a human-readable verdict, confidence level, and improvement hints.",
            ],
        }

    def _extract_claims(self, resume_text: str, claimed_skills_text: str) -> list[dict]:
        claims: dict[str, dict] = {}

        for raw_claim in CLAIM_SPLIT_PATTERN.split(claimed_skills_text):
            clean_claim = raw_claim.strip()
            skill_area = normalize_skill_name(clean_claim)
            if not skill_area:
                continue
            self._store_claim(claims, skill_area, clean_claim, max(detect_claim_strength(clean_claim), 2))

        if not claims:
            for sentence in SENTENCE_SPLIT_PATTERN.split(resume_text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                for skill_area in find_skills_in_text(sentence):
                    self._store_claim(claims, skill_area, sentence, detect_claim_strength(sentence))

        if not claims:
            for skill_area in find_skills_in_text(resume_text):
                self._store_claim(claims, skill_area, f"Implicit mention of {skill_area}", 1)

        if not claims:
            self._store_claim(claims, "Machine Learning", "Default analysis target: Machine Learning", 2)

        return list(claims.values())

    def _store_claim(self, claims: dict[str, dict], skill_area: str, source_text: str, claim_level: int) -> None:
        existing = claims.get(skill_area)
        if not existing or claim_level > existing["claim_level"]:
            claims[skill_area] = {
                "skill_area": skill_area,
                "source_text": source_text.strip(),
                "claim_level": max(claim_level, 1),
            }

    def _build_feature_row(
        self,
        resume_text: str,
        claim_level: int,
        skill_area: str,
        github_metrics: dict,
    ) -> dict:
        lowered_resume = resume_text.lower()
        skill_sentence_hits = sum(
            1
            for sentence in SENTENCE_SPLIT_PATTERN.split(resume_text)
            if skill_area in find_skills_in_text(sentence)
        )
        years_signal = max([float(value) for value in YEARS_PATTERN.findall(resume_text)] or [0.0])

        return {
            "skill_area": skill_area,
            "claim_level": claim_level,
            "resume_keyword_hits": (
                keyword_hits(resume_text, skill_area, "evidence_keywords")
                + keyword_hits(resume_text, skill_area, "languages")
                + keyword_hits(resume_text, skill_area, "aliases")
            ),
            "action_hits": sum(lowered_resume.count(term) for term in ACTION_TERMS),
            "project_mentions": sum(lowered_resume.count(term) for term in PROJECT_TERMS),
            "years_signal": min(years_signal, 10.0),
            "certification_hits": sum(lowered_resume.count(term) for term in CERTIFICATION_TERMS),
            "skill_sentence_hits": skill_sentence_hits,
            "github_relevant_repos": github_metrics["relevant_repo_count"],
            "github_recent_repos": github_metrics["recent_relevant_repo_count"],
            "github_relevant_stars": github_metrics["relevant_star_count"],
            "github_profile_score": github_metrics["profile_score"],
            "external_link_hits": github_metrics["external_link_hits"],
        }

    def _format_claim_result(
        self,
        claim: dict,
        feature_row: dict,
        github_metrics: dict,
        realism_probability: float,
    ) -> dict:
        verdict, confidence = self._claim_verdict(realism_probability)
        evidence_points: list[str] = []
        risk_points: list[str] = []

        if feature_row["resume_keyword_hits"] >= 4:
            evidence_points.append("Resume includes multiple domain-specific keywords.")
        if feature_row["project_mentions"] >= 2:
            evidence_points.append("Project-oriented language appears repeatedly in the resume.")
        if feature_row["github_relevant_repos"] >= 2:
            evidence_points.append("GitHub shows multiple repos aligned with this skill.")
        if feature_row["github_recent_repos"] >= 1:
            evidence_points.append("Recent activity supports that the skill is still practiced.")
        if feature_row["years_signal"] >= 2:
            evidence_points.append("Experience duration claims add supporting context.")

        public_signal_available = github_metrics["total_repo_count"] > 0 or feature_row["external_link_hits"] > 0

        if claim["claim_level"] >= 3 and feature_row["github_relevant_repos"] == 0:
            if public_signal_available:
                risk_points.append("High-confidence claim but no matching GitHub evidence was found.")
            else:
                risk_points.append("High-confidence claim but no public GitHub or project evidence was provided.")
        if feature_row["resume_keyword_hits"] <= 1:
            risk_points.append("The resume contains very little technical detail for this skill.")
        if feature_row["project_mentions"] == 0:
            risk_points.append("No project or delivery language appears near the claim.")
        if public_signal_available and not github_metrics["matched_projects"]:
            risk_points.append("No public projects were matched to the claimed skill.")

        if not evidence_points:
            evidence_points.append("Only limited supporting signals were found.")

        return {
            "skill": claim["skill_area"],
            "claim_source": claim["source_text"],
            "claim_level": claim["claim_level"],
            "verdict": verdict,
            "confidence": confidence,
            "realism_score": round(realism_probability * 100, 1),
            "exaggeration_risk": round((1 - realism_probability) * 100, 1),
            "evidence_points": evidence_points[:3],
            "risk_points": risk_points[:3],
            "matched_projects": github_metrics["matched_projects"],
            "signals": {
                "resume_keywords": feature_row["resume_keyword_hits"],
                "project_mentions": feature_row["project_mentions"],
                "years_signal": feature_row["years_signal"],
                "github_repos": feature_row["github_relevant_repos"],
                "recent_activity": feature_row["github_recent_repos"],
                "external_links": feature_row["external_link_hits"],
            },
        }

    @staticmethod
    def _claim_verdict(realism_probability: float) -> tuple[str, float]:
        if realism_probability >= 0.68:
            return "Looks realistic", round(realism_probability * 100, 1)
        if realism_probability >= 0.48:
            return "Needs stronger evidence", round((0.5 + abs(realism_probability - 0.5)) * 100, 1)
        return "Likely exaggerated", round((1 - realism_probability) * 100, 1)

    @staticmethod
    def _overall_verdict(score: float) -> tuple[str, str]:
        if score >= 0.68:
            return "Credible profile", "Most claims are supported by resume depth, project evidence, and activity signals."
        if score >= 0.48:
            return "Mixed evidence", "Some claims look reasonable, but a few need stronger proof through projects or GitHub activity."
        return "High exaggeration risk", "The claims currently look stronger than the public evidence that backs them up."

    @staticmethod
    def _top_risks(assessments: list[dict]) -> list[str]:
        risks: list[str] = []
        for assessment in assessments:
            risks.extend(assessment["risk_points"])
        deduped: list[str] = []
        for risk in risks:
            if risk not in deduped:
                deduped.append(risk)
        return deduped[:4]

    @staticmethod
    def _resume_summary(resume_text: str) -> dict:
        lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
        words = [word for word in re.split(r"\s+", resume_text.strip()) if word]
        skills = find_skills_in_text(resume_text)
        return {
            "line_count": len(lines),
            "word_count": len(words),
            "detected_skills": skills[:6],
        }
