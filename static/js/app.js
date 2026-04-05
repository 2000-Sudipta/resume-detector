const root = document.documentElement;
const form = document.getElementById("ats-form");
const composeView = document.getElementById("workspace-view");
const resultsView = document.getElementById("results-view");
const themeToggle = document.getElementById("theme-toggle");
const themeLabel = themeToggle?.querySelector(".theme-toggle-label");
const loadDemoButton = document.getElementById("load-demo");
const heroRunButton = document.getElementById("hero-run");
const backToWorkspaceButton = document.getElementById("back-to-workspace");
const analyzeButton = document.getElementById("analyze-button");
const statusPill = document.getElementById("status-pill");
const resultsEmpty = document.getElementById("results-empty");
const results = document.getElementById("results");
const summaryGrid = document.getElementById("summary-grid");
const rankingBody = document.getElementById("ranking-body");
const compareBlock = document.getElementById("compare-block");
const compareLeftName = document.getElementById("compare-left-name");
const compareRightName = document.getElementById("compare-right-name");
const compareBody = document.getElementById("compare-body");
const candidateGrid = document.getElementById("candidate-grid");
const minExperience = document.getElementById("min_experience");
const experienceValue = document.getElementById("experience-value");

const demoPayload = {
    resume_texts: `Rahul Sharma
Data Scientist
Email: rahul@example.com
Location: Bengaluru

Summary
3+ years building ML and NLP solutions for startup products.

Skills
Python, SQL, Machine Learning, NLP, scikit-learn, TensorFlow, Flask

Experience
2022 - Present
Built churn prediction, recommendation, and text classification systems.
Improved model accuracy by 18% and reduced manual review time by 35%.
Led deployment of a Flask model scoring API.

Projects
Built 3 ML projects in NLP and forecasting.

Education
B.Tech in Computer Science

--- Candidate ---

Anita Verma
Data Analyst
Email: anita@example.com
Location: Remote

Summary
2 years of analytics experience with dashboards and reporting.

Skills
SQL, Tableau, Excel, Python

Experience
2023 - Present
Built dashboard reports for sales and marketing teams.
Improved reporting turnaround by 20%.

Education
Bachelors in Statistics`,
    job_description: `Role: Data Scientist
Required skills: Python, SQL, Machine Learning, NLP, scikit-learn
Experience required: 2+ years
Education: Bachelors or above
Need someone who can work in a startup environment, own model delivery, and collaborate cross-functionally.`,
    must_have_skills: "Machine Learning, NLP",
    location_preference: "Bengaluru",
    salary_min: "8",
    salary_max: "18",
    culture_keywords: "startup, ownership, cross-functional",
    priority_mode: "balanced",
};

const demoSkillChips = ["Python", "SQL", "Machine Learning", "NLP", "scikit-learn"];


function applyTheme(theme) {
    root.dataset.theme = theme;
    localStorage.setItem("talent-prism-theme", theme);
    if (themeLabel) {
        themeLabel.textContent = theme === "dark" ? "Dark Mode" : "Light Mode";
    }
}


function initializeTheme() {
    const savedTheme = localStorage.getItem("talent-prism-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    applyTheme(savedTheme || (prefersDark ? "dark" : "light"));
}


function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}


function verdictClass(verdict) {
    if (verdict === "Shortlist") return "verdict-shortlist";
    if (verdict === "Review Manually") return "verdict-review";
    return "verdict-reject";
}


function createScoreBar(label, value) {
    return `
        <div class="score-bar">
            <div class="score-bar-head">
                <span>${escapeHtml(label)}</span>
                <strong>${Number(value).toFixed(1)}%</strong>
            </div>
            <div class="track"><span data-width="${Number(value).toFixed(1)}"></span></div>
        </div>
    `;
}


function renderSummary(summary, job) {
    const items = [
        ["Candidates", summary.candidate_count],
        ["Top Candidate", summary.top_candidate],
        ["Shortlist", summary.shortlist_count],
        ["Review", summary.review_count],
        ["Reject", summary.reject_count],
        ["Priority", job.priority_mode.replace(/\b\w/g, (char) => char.toUpperCase())],
    ];

    summaryGrid.innerHTML = items
        .map(
            ([label, value], index) => `
                <article class="summary-card" style="animation-delay:${index * 60}ms">
                    <span class="pill-muted">${escapeHtml(label)}</span>
                    <strong>${escapeHtml(value)}</strong>
                </article>
            `
        )
        .join("");
}


function renderRanking(ranking) {
    rankingBody.innerHTML = ranking
        .map(
            (row) => `
                <tr>
                    <td>${row.rank}</td>
                    <td>${escapeHtml(row.name)}</td>
                    <td>${Number(row.score).toFixed(1)}%</td>
                    <td>${Number(row.skills_match).toFixed(1)}%</td>
                    <td>${Number(row.experience_match).toFixed(1)}%</td>
                    <td><span class="small-pill ${verdictClass(row.verdict)}">${escapeHtml(row.verdict)}</span></td>
                </tr>
            `
        )
        .join("");
}


function renderCompare(compare) {
    if (!compare) {
        compareBlock.classList.add("hidden");
        return;
    }

    compareLeftName.textContent = compare.left.name;
    compareRightName.textContent = compare.right.name;
    compareBody.innerHTML = compare.rows
        .map(
            (row) => `
                <tr>
                    <td>${escapeHtml(row.feature)}</td>
                    <td>${escapeHtml(row.left)}</td>
                    <td>${escapeHtml(row.right)}</td>
                </tr>
            `
        )
        .join("");
    compareBlock.classList.remove("hidden");
}


function renderCandidates(candidates) {
    candidateGrid.innerHTML = candidates
        .map(
            (candidate, index) => `
                <article class="candidate-card" style="animation-delay:${index * 90}ms">
                    <div class="candidate-head">
                        <div>
                            <h3>${escapeHtml(candidate.name)}</h3>
                            <p class="candidate-subtext">${escapeHtml(candidate.verdict_reason)}</p>
                        </div>
                        <span class="verdict-tag ${verdictClass(candidate.verdict)}">${escapeHtml(candidate.verdict)}</span>
                    </div>

                    <div class="candidate-meta">
                        <span>ATS Score: <strong>${Number(candidate.overall_score).toFixed(1)}%</strong></span>
                        <span>Experience: <strong>${Number(candidate.experience_years).toFixed(1)} yrs</strong></span>
                        <span>Education: <strong>${escapeHtml(candidate.education_level)}</strong></span>
                        <span>Location: <strong>${escapeHtml(candidate.contact.location)}</strong></span>
                    </div>

                    <div class="score-cluster">
                        ${createScoreBar("Skills Match", candidate.breakdown.skills_match)}
                        ${createScoreBar("Experience Match", candidate.breakdown.experience_match)}
                        ${createScoreBar("Education Match", candidate.breakdown.education_match)}
                        ${createScoreBar("Culture Match", candidate.breakdown.culture_match)}
                        ${createScoreBar("Authenticity", candidate.breakdown.authenticity_score)}
                    </div>

                    <div class="pill-row">
                        ${(candidate.matched_skills || []).map((skill) => `<span class="small-pill">${escapeHtml(skill)}</span>`).join("") || '<span class="small-pill">No matched must-haves</span>'}
                    </div>

                    <div class="list-block">
                        <h4>Reasoning</h4>
                        <ul>${candidate.reasoning.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
                    </div>

                    <div class="list-block">
                        <h4>Key Highlights</h4>
                        <ul>${candidate.highlights.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>No standout highlights detected.</li>"}</ul>
                    </div>

                    <div class="list-block">
                        <h4>Red Flags</h4>
                        <ul>${candidate.red_flags.map((item) => `<li>${escapeHtml(item)}</li>`).join("") || "<li>No major red flags detected.</li>"}</ul>
                    </div>
                </article>
            `
        )
        .join("");
}


function animateBars() {
    requestAnimationFrame(() => {
        document.querySelectorAll(".track span[data-width]").forEach((bar) => {
            bar.style.width = `${bar.dataset.width}%`;
        });
    });
}


function setActiveView(viewName, shouldPushState = false) {
    const showResults = viewName === "results";
    composeView?.classList.toggle("is-active", !showResults);
    resultsView?.classList.toggle("is-active", showResults);
    window.scrollTo({ top: 0 });

    if (shouldPushState) {
        const hash = showResults ? "#results" : "#workspace";
        window.history.pushState({ view: viewName }, "", hash);
    }
}


function showResultsView(shouldPushState = false) {
    setActiveView("results", shouldPushState);
}


function showWorkspaceView(shouldPushState = false) {
    setActiveView("workspace", shouldPushState);
}


function setLoading(isLoading) {
    analyzeButton.disabled = isLoading;
    analyzeButton.classList.toggle("is-loading", isLoading);
    statusPill.textContent = isLoading ? "Scoring candidates..." : statusPill.textContent;
}


async function submitForm(event) {
    event.preventDefault();
    setLoading(true);
    statusPill.textContent = "Scoring candidates...";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: new FormData(form),
        });
        const payload = await response.json();

        if (!response.ok) {
            throw new Error(payload.error || "ATS analysis could not be completed.");
        }

        renderSummary(payload.summary, payload.job);
        renderRanking(payload.ranking || []);
        renderCompare(payload.compare);
        renderCandidates(payload.candidates || []);

        resultsEmpty.classList.add("hidden");
        results.classList.remove("hidden");
        results.classList.remove("is-ready");
        void results.offsetWidth;
        results.classList.add("is-ready");
        statusPill.textContent = `${payload.summary.shortlist_count} shortlist / ${payload.summary.review_count} review`;
        showResultsView(true);
        animateBars();
    } catch (error) {
        results.classList.add("hidden");
        resultsEmpty.classList.remove("hidden");
        resultsEmpty.innerHTML = `
            <h3>ATS analysis could not finish</h3>
            <p>${escapeHtml(error.message)}</p>
        `;
        statusPill.textContent = "Action required";
        showResultsView(true);
    } finally {
        setLoading(false);
    }
}


function loadDemo() {
    Object.entries(demoPayload).forEach(([name, value]) => {
        const field = form.elements.namedItem(name);
        if (field) {
            field.value = value;
        }
    });

    document.querySelectorAll('input[name="must_have_skill_chip"]').forEach((checkbox) => {
        checkbox.checked = demoSkillChips.includes(checkbox.value);
    });

    updateExperienceLabel();
    statusPill.textContent = "Demo loaded";
}


function updateExperienceLabel() {
    const value = Number(minExperience?.value || 0);
    if (experienceValue) {
        experienceValue.textContent = `${value} year${value === 1 ? "" : "s"}`;
    }
}


function initializeRevealAnimations() {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("is-visible");
                }
            });
        },
        { threshold: 0.15 }
    );

    document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));
}


themeToggle?.addEventListener("click", () => {
    applyTheme(root.dataset.theme === "dark" ? "light" : "dark");
});

form?.addEventListener("submit", submitForm);
loadDemoButton?.addEventListener("click", loadDemo);
heroRunButton?.addEventListener("click", () => {
    form?.scrollIntoView({ behavior: "smooth", block: "start" });
});
backToWorkspaceButton?.addEventListener("click", () => {
    showWorkspaceView(true);
});
minExperience?.addEventListener("input", updateExperienceLabel);
window.addEventListener("popstate", () => {
    if (window.location.hash === "#results") {
        showResultsView(false);
        return;
    }
    showWorkspaceView(false);
});

initializeTheme();
initializeRevealAnimations();
updateExperienceLabel();
if (window.location.hash === "#results") {
    showResultsView(false);
} else {
    showWorkspaceView(false);
}
