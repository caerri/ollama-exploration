"""Canvas LMS API client — read-only course data and Obsidian vault sync.

Exports:
  get_canvas_context(days)  — formatted text block of upcoming assignments
  sync_courses(vault_path)  — full course mirror to Obsidian vault
  list_courses()            — active courses (for debugging/REPL)

All Canvas API access is isolated here. The sync writes deterministic
markdown files to the vault — no LLM is involved in the output.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from markdownify import markdownify as md

from config import get_env

# ---------------------------------------------------------------------------
# Cache for context injection (same pattern as calendar/obsidian clients)
# ---------------------------------------------------------------------------
_cache: tuple[float, str] = (0.0, "")
_CACHE_TTL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _base_url() -> str:
    return get_env("CANVAS_API_URL", "").rstrip("/")


def _token() -> str:
    return get_env("CANVAS_API_TOKEN", "")


def _headers() -> dict:
    return {"Authorization": f"Bearer {_token()}"}


def _api_get(endpoint: str, params: dict | None = None) -> list[dict]:
    """GET a Canvas API endpoint, following pagination automatically.

    Returns the full list of JSON objects across all pages.
    Returns an empty list on any error.
    """
    base = _base_url()
    if not base or not _token():
        return []

    url = f"{base}/api/v1/{endpoint.lstrip('/')}"
    all_results: list[dict] = []
    p = dict(params or {})
    p.setdefault("per_page", "50")

    try:
        while url:
            resp = requests.get(url, headers=_headers(), params=p, timeout=15)
            if resp.status_code != 200:
                return all_results
            data = resp.json()
            if isinstance(data, list):
                all_results.extend(data)
            elif isinstance(data, dict):
                all_results.append(data)
                return all_results

            # Follow pagination via Link header
            url = None
            p = None  # params only needed for first request
            link_header = resp.headers.get("Link", "")
            for part in link_header.split(","):
                if 'rel="next"' in part:
                    match = re.search(r"<([^>]+)>", part)
                    if match:
                        url = match.group(1)
    except (requests.RequestException, ValueError):
        pass

    return all_results


def _html_to_md(html: str | None) -> str:
    """Convert HTML content to clean markdown."""
    if not html:
        return ""
    result = md(html, heading_style="ATX", strip=["img"])
    # Collapse excessive blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _safe_filename(name: str) -> str:
    """Convert a string into a safe filename for the vault."""
    # Remove characters that are problematic in filenames
    safe = re.sub(r'[<>:"/\\|?*]', "", name)
    safe = safe.strip(". ")
    return safe[:100] if safe else "untitled"


def _parse_due(due_str: str | None) -> datetime | None:
    """Parse a Canvas ISO 8601 due date string."""
    if not due_str:
        return None
    try:
        return datetime.fromisoformat(due_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def _format_due(dt: datetime | None) -> str:
    """Format a due date for display."""
    if not dt:
        return "no due date"
    local = dt.astimezone()
    return local.strftime("%b %-d")


def _short_course_name(full_name: str) -> str:
    """Shorten a Canvas course name for display.

    'Full-Stack Web Development Using JavaScript, Node.js, and AI Tools' → kept as-is
    but truncated at 60 chars for the context block.
    """
    if len(full_name) > 60:
        return full_name[:57] + "..."
    return full_name


# ---------------------------------------------------------------------------
# Public: context injection for relay app
# ---------------------------------------------------------------------------

def get_canvas_context(days: int | None = None) -> str:
    """Fetch upcoming assignments and return a formatted text block.

    Returns empty string if API unavailable or no active courses.
    Results are cached for 5 minutes.
    """
    if days is None:
        days = 14

    global _cache
    now = time.time()
    if _cache[1] and (now - _cache[0]) < _CACHE_TTL:
        return _cache[1]

    courses = _api_get("courses", {"enrollment_state": "active"})
    if not courses:
        return ""

    cutoff = datetime.now(timezone.utc) + timedelta(days=days)
    all_assignments: list[dict] = []

    for course in courses:
        cid = course.get("id")
        cname = course.get("name", "Unknown Course")
        assignments = _api_get(
            f"courses/{cid}/assignments",
            {"order_by": "due_at"},
        )
        for a in assignments:
            due = _parse_due(a.get("due_at"))
            submitted = a.get("has_submitted_submissions", False)
            if submitted:
                continue  # skip already-submitted
            # Include if: overdue, or due within window, or no due date
            if due and due > cutoff:
                continue
            all_assignments.append({
                "course": cname,
                "name": a.get("name", "Unnamed"),
                "due": due,
                "due_str": _format_due(due),
                "points": a.get("points_possible", 0),
            })

    if not all_assignments:
        result = (
            "--- Your Assignments ---\n"
            "No upcoming assignments.\n"
            "--- End Assignments ---"
        )
        _cache = (now, result)
        return result

    # Sort: overdue first, then by due date, no-date last
    def sort_key(a):
        if a["due"] is None:
            return (1, datetime.max.replace(tzinfo=timezone.utc))
        return (0, a["due"])

    all_assignments.sort(key=sort_key)

    # Group by course
    lines = ["--- Your Assignments ---"]
    current_course = None
    for a in all_assignments:
        if a["course"] != current_course:
            current_course = a["course"]
            lines.append(_short_course_name(current_course))
        pts = f"{a['points']:.0f} pts" if a["points"] else "ungraded"
        overdue = ""
        if a["due"] and a["due"] < datetime.now(timezone.utc):
            overdue = " [OVERDUE]"
        lines.append(
            f"  {a['name']} | due: {a['due_str']} | {pts}{overdue}"
        )

    lines.append("--- End Assignments ---")
    result = "\n".join(lines)
    _cache = (now, result)
    return result


# ---------------------------------------------------------------------------
# Public: full course sync to Obsidian vault
# ---------------------------------------------------------------------------

def sync_courses(vault_path: str | None = None) -> dict:
    """Mirror active Canvas courses to the Obsidian vault.

    Creates a structured folder per course under Areas/School/.
    Returns a summary dict: {"courses": [...], "files_written": int}.
    """
    if vault_path is None:
        vault_path = get_env(
            "OBSIDIAN_VAULT_PATH",
            "~/Library/Mobile Documents/iCloud~md~obsidian/Documents/vault-zero",
        )

    vault = Path(vault_path).expanduser()
    school_dir = vault / "Areas" / "School"
    school_dir.mkdir(parents=True, exist_ok=True)

    courses = _api_get("courses", {
        "enrollment_state": "active",
        "include[]": "syllabus_body",
    })
    if not courses:
        return {"courses": [], "files_written": 0}

    summary = {"courses": [], "files_written": 0}
    all_assignments_for_dashboard: list[dict] = []

    for course in courses:
        cid = course.get("id")
        cname = course.get("name", "Unknown Course")
        safe_name = _safe_filename(cname)
        course_dir = school_dir / safe_name
        course_dir.mkdir(parents=True, exist_ok=True)

        course_summary = {"name": cname, "id": cid, "sections": []}

        # --- Syllabus ---
        syllabus_html = course.get("syllabus_body")
        if syllabus_html:
            syllabus_md = _html_to_md(syllabus_html)
            if syllabus_md:
                path = course_dir / "Syllabus.md"
                path.write_text(
                    f"---\ncourse: {cname}\ntype: syllabus\n"
                    f"synced: {datetime.now().isoformat(timespec='seconds')}\n---\n\n"
                    f"# Syllabus\n\n{syllabus_md}\n",
                    encoding="utf-8",
                )
                summary["files_written"] += 1
                course_summary["sections"].append("Syllabus")

        # --- Assignments ---
        assignments = _api_get(f"courses/{cid}/assignments", {"order_by": "due_at"})
        if assignments:
            asg_dir = course_dir / "Assignments"
            asg_dir.mkdir(exist_ok=True)
            for a in assignments:
                aname = a.get("name", "Unnamed")
                due = _parse_due(a.get("due_at"))
                pts = a.get("points_possible", 0)
                submitted = a.get("has_submitted_submissions", False)
                desc = _html_to_md(a.get("description"))

                content = (
                    f"---\ncourse: {cname}\ntype: assignment\n"
                    f"due: {a.get('due_at', 'none')}\n"
                    f"points: {pts}\n"
                    f"submitted: {submitted}\n"
                    f"synced: {datetime.now().isoformat(timespec='seconds')}\n---\n\n"
                    f"# {aname}\n\n"
                    f"**Due:** {_format_due(due)}  \n"
                    f"**Points:** {pts}  \n"
                    f"**Submitted:** {'yes' if submitted else 'no'}\n"
                )
                if desc:
                    content += f"\n## Description\n\n{desc}\n"

                path = asg_dir / f"{_safe_filename(aname)}.md"
                path.write_text(content, encoding="utf-8")
                summary["files_written"] += 1

                # Collect for dashboard
                all_assignments_for_dashboard.append({
                    "course": cname,
                    "name": aname,
                    "due": due,
                    "due_str": _format_due(due),
                    "points": pts,
                    "submitted": submitted,
                })

            course_summary["sections"].append(f"Assignments ({len(assignments)})")

        # --- Announcements ---
        announcements = _api_get(
            "announcements",
            {"context_codes[]": f"course_{cid}"},
        )
        if announcements:
            ann_dir = course_dir / "Announcements"
            ann_dir.mkdir(exist_ok=True)
            for ann in announcements:
                title = ann.get("title", "Untitled")
                posted = ann.get("posted_at", "")
                author = ann.get("author", {}).get("display_name", "Unknown")
                body = _html_to_md(ann.get("message"))

                # Date prefix for sorting
                date_prefix = ""
                if posted:
                    try:
                        dt = datetime.fromisoformat(posted.replace("Z", "+00:00"))
                        date_prefix = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pass

                fname = f"{date_prefix} — {_safe_filename(title)}.md" if date_prefix else f"{_safe_filename(title)}.md"
                content = (
                    f"---\ncourse: {cname}\ntype: announcement\n"
                    f"posted: {posted}\nauthor: {author}\n"
                    f"synced: {datetime.now().isoformat(timespec='seconds')}\n---\n\n"
                    f"# {title}\n\n"
                    f"*Posted by {author}*\n\n{body}\n"
                )
                path = ann_dir / fname
                path.write_text(content, encoding="utf-8")
                summary["files_written"] += 1

            course_summary["sections"].append(f"Announcements ({len(announcements)})")

        # --- Modules ---
        modules = _api_get(f"courses/{cid}/modules")
        if modules:
            mod_dir = course_dir / "Modules"
            mod_dir.mkdir(exist_ok=True)
            for module in modules:
                mname = module.get("name", "Unnamed Module")
                mid = module.get("id")
                items = _api_get(f"courses/{cid}/modules/{mid}/items")

                lines = [
                    f"---\ncourse: {cname}\ntype: module\n"
                    f"synced: {datetime.now().isoformat(timespec='seconds')}\n---\n\n"
                    f"# {mname}\n",
                ]
                for item in items:
                    itype = item.get("type", "")
                    ititle = item.get("title", "Untitled")
                    indent = "  " * item.get("indent", 0)
                    if itype == "SubHeader":
                        lines.append(f"\n### {ititle}\n")
                    else:
                        lines.append(f"{indent}- **{itype}:** {ititle}")

                path = mod_dir / f"{_safe_filename(mname)}.md"
                path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                summary["files_written"] += 1

            course_summary["sections"].append(f"Modules ({len(modules)})")

        # --- Pages ---
        pages = _api_get(f"courses/{cid}/pages")
        if pages:
            pages_dir = course_dir / "Pages"
            pages_dir.mkdir(exist_ok=True)
            for page in pages:
                ptitle = page.get("title", "Untitled")
                purl = page.get("url", "")
                # Fetch full page content
                if purl:
                    full_pages = _api_get(f"courses/{cid}/pages/{purl}")
                    if full_pages:
                        body = _html_to_md(full_pages[0].get("body"))
                    else:
                        body = ""
                else:
                    body = ""

                content = (
                    f"---\ncourse: {cname}\ntype: page\n"
                    f"synced: {datetime.now().isoformat(timespec='seconds')}\n---\n\n"
                    f"# {ptitle}\n\n{body}\n"
                )
                path = pages_dir / f"{_safe_filename(ptitle)}.md"
                path.write_text(content, encoding="utf-8")
                summary["files_written"] += 1

            course_summary["sections"].append(f"Pages ({len(pages)})")

        # --- Files (metadata only — list what's available) ---
        files = _api_get(f"courses/{cid}/files")
        if files:
            files_dir = course_dir / "Files"
            files_dir.mkdir(exist_ok=True)

            # Write an index instead of downloading everything
            lines = [
                f"---\ncourse: {cname}\ntype: file-index\n"
                f"synced: {datetime.now().isoformat(timespec='seconds')}\n---\n\n"
                f"# Course Files\n\n"
                f"| File | Size | Type |\n"
                f"|---|---|---|\n"
            ]
            for f in files:
                fname = f.get("display_name", f.get("filename", "unknown"))
                size_bytes = f.get("size", 0)
                if size_bytes > 1_048_576:
                    size = f"{size_bytes / 1_048_576:.1f} MB"
                elif size_bytes > 1024:
                    size = f"{size_bytes / 1024:.1f} KB"
                else:
                    size = f"{size_bytes} B"
                ftype = f.get("content-type", f.get("mime_class", ""))
                lines.append(f"| {fname} | {size} | {ftype} |")

            path = files_dir / "INDEX.md"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            summary["files_written"] += 1
            course_summary["sections"].append(f"Files ({len(files)})")

        summary["courses"].append(course_summary)

    # --- Dashboard: cross-course assignment overview ---
    _write_dashboard(school_dir / "Dashboard.md", all_assignments_for_dashboard)
    summary["files_written"] += 1

    return summary


def _write_dashboard(path: Path, assignments: list[dict]) -> None:
    """Write a cross-course assignment dashboard to the vault."""
    now_utc = datetime.now(timezone.utc)
    week_out = now_utc + timedelta(days=7)

    overdue = [a for a in assignments if not a["submitted"] and a["due"] and a["due"] < now_utc]
    upcoming = [a for a in assignments if not a["submitted"] and a["due"] and now_utc <= a["due"]]
    no_date = [a for a in assignments if not a["submitted"] and not a["due"]]

    # Sort each group
    overdue.sort(key=lambda a: a["due"])
    upcoming.sort(key=lambda a: a["due"])

    lines = [
        f"---\ntype: dashboard\n"
        f"synced: {datetime.now().isoformat(timespec='seconds')}\n---\n\n"
        f"# Assignment Dashboard\n",
    ]

    if overdue:
        lines.append("## Overdue\n")
        for a in overdue:
            lines.append(
                f"- [ ] **{a['name']}** — {a['course']} — "
                f"was due {a['due_str']} ({a['points']:.0f} pts)"
            )
        lines.append("")

    lines.append("## Upcoming\n")
    if upcoming:
        for a in upcoming:
            bold = "**" if a["due"] and a["due"] <= week_out else ""
            lines.append(
                f"- [ ] {bold}{a['name']}{bold} — {a['course']} — "
                f"due {a['due_str']} ({a['points']:.0f} pts)"
            )
    else:
        lines.append("Nothing upcoming.")
    lines.append("")

    if no_date:
        lines.append("## No Due Date\n")
        for a in no_date:
            lines.append(
                f"- [ ] {a['name']} — {a['course']} ({a['points']:.0f} pts)"
            )
        lines.append("")

    # Submitted summary
    submitted = [a for a in assignments if a["submitted"]]
    if submitted:
        lines.append(f"## Completed ({len(submitted)})\n")
        for a in submitted:
            lines.append(f"- [x] {a['name']} — {a['course']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public: course listing (debugging/REPL)
# ---------------------------------------------------------------------------

def list_courses() -> list[dict]:
    """Return active courses with id and name."""
    courses = _api_get("courses", {"enrollment_state": "active"})
    return [
        {"id": c.get("id"), "name": c.get("name", "Unknown")}
        for c in courses
    ]
