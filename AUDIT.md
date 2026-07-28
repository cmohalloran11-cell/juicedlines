# JUICED 2.0 — Phase 1 Completion Audit

_Audit date: 2026-07-28 · Scope: `static/dashboard.html` (the live SPA) + the Python projection/API backend it reads._

## 0. Architecture reality (read this first)

The audit brief is written for a **React/TypeScript** app (re-renders, component splitting, TS typing,
unused npm packages, virtual scrolling). Juiced is **not** that. It is:

- **Frontend:** one file, `static/dashboard.html` — 730 lines of vanilla JS, no build step, no framework, no packages.
- **Backend:** Python/FastAPI + a vendored projection engine (`projector_bridge.py`, `analytics.py`, `valuation.py`).
- **Deploy:** static — the page is served by a host (Vercel/Pages) and reads prebuilt JSON off the `data` branch.

This was a deliberate, approved strategy (strangler pattern, keep the engine, don't rewrite in Next.js). So the
React/TS-specific steps are audited **by intent** (perf, dead code, structure, maintainability) against the code
that actually exists — not against a component tree that doesn't. Every finding below cites `file:line`.

---

## 1. Step-by-step checklist verdicts

### Step 1 — Projections table columns
Current columns ([dashboard.html:502](static/dashboard.html#L502), `richRow` [dashboard.html:506](static/dashboard.html#L506)):
★ · Player · Team · Prop · Book · Line · Proj · Edge · Juice · Confidence · Over% · AI · +add.

| Required | Status | Note |
|---|---|---|
| Player | ✅ | |
| Team | ✅ | |
| **Opponent** | ❌ | Not in payload. Data exists (`analytics.py:293`), not plumbed to `_drop` (`dashboard.py:25`). |
| Prop | ✅ | |
| Sportsbook | ✅ | Book logo column |
| Line | ✅ | |
| Juiced Projection | ✅ | |
| Edge | ✅ | (Edge% = EV%) |
| Juice Score | ✅ | |
| Confidence | ✅ | |
| Over % | ✅ | |
| **Last Updated** | ❌ | No per-row timestamp; only a static "live" label in the status bar. |
| AI Explain | ✅ | sparkle column opens drawer on AI tab |
| Expand Row | ⚠️ | Row click opens the drawer, but there's no expand affordance/chevron. |

### Step 1 — Filters
Present: Sportsbooks, Prop Categories, Juice Score, Confidence, Min Edge%, player Search, Sport toggle, High-Juice + Watchlist quick-filters ([dashboard.html:454-488](static/dashboard.html#L454)).

| Required | Status |
|---|---|
| Sports / Sportsbooks / Prop Categories / Search / Confidence / Juice Score / Edge / Watchlist | ✅ |
| **Date** | ❌ today-only, no slate navigation |
| **Projection Difference** | ❌ (only EV% edge; no proj−line filter) |
| **Home/Away** | ❌ data not in payload |
| **Weather** | ❌ no weather filter on props |
| **Game Time** | ❌ (`startTime` exists, unused as filter) |
| **Projection Movement** | ❌ (CLV ledger has it, not exposed) |
| **Line Movement** | ❌ (same) |
| **Favorites** | ❌ (only Watchlist; no separate favorites) |
| **Saved Filter Presets** | ❌ |

### Step 1 — Sorting / Search / Pagination / Responsive
- **Sorting:** ❌ "every column sorts" not met — sort is a 4-option dropdown ([dashboard.html:486](static/dashboard.html#L486)); table headers are not clickable.
- **Search:** ✅ instant, but player-name only (not team/prop).
- **Pagination:** ✅ pager, 25/page. Virtual scrolling not needed at this size.
- **Responsive:** ❌ **mobile is broken.** `.app{grid-template-columns:232px 1fr}` ([dashboard.html:17](static/dashboard.html#L17)) has no mobile breakpoint — the 232px sidebar never collapses; there is no hamburger. Only one `@media(max-width:1200px)` rule exists ([dashboard.html:153](static/dashboard.html#L153)). Desktop ✅, tablet ⚠️, mobile ❌.

### Step 2 — Projection drawer (`openProp` [dashboard.html:660](static/dashboard.html#L660))
| Required | Status |
|---|---|
| Player Header / Projection / Line / Edge / Juice / Confidence | ✅ |
| Distribution Chart | ✅ (`drawDist`) |
| Confidence Breakdown | ⚠️ **present but FAKE** (see C1) |
| **Projection Trend** | ❌ not in drawer |
| **Recent Games** | ❌ crammed into the "Research" button → dumps raw text into the AI box ([dashboard.html:692](static/dashboard.html#L692)); no games table/chart |
| **Market History** | ❌ line-history chart exists on the dashboard, not in the drawer |
| AI Explain | ✅ |
| Correlated Props | ⚠️ **present but FAKE** (see C2) |
| Quick Actions | ✅ (Watchlist / Portfolio / Research) |

Note: `.dtabs` CSS ([dashboard.html:152](static/dashboard.html#L152)) implies a tabbed drawer that was never built — dead CSS.

### Step 3 — Engine integration
Real, from the engine/ledger (good): projection, edge, edgePct, probability, juiceScore, confidence, floor, ceiling, line, over%. **Fabricated values still shipping — see C1–C4.**

### Steps 4–8 summary
- **Perf:** duplicate `/api/projections?limit=1000` fetches (`vProjections` + `watchlist.list`); dashboard re-renders fully every 60s. No skeleton loaders (text "Loading…" only).
- **UI polish:** ❌ no sticky headers, ❌ no resizable columns, ❌ no pinned columns. Spacing/radius/typography are consistent (CSS vars) ✅.
- **UX states:** loading ✅(crude), empty ✅(good), error ✅ but ❌ **no retry button**; tooltips mostly ❌; keyboard nav ⚠️ (⌘K shown, not wired).
- **Freshness:** ❌ essentially unbuilt — see C3.
- **Tech debt:** `app.html` (148 KB legacy board) is orphaned/dead; **Bet Journal duplicates Portfolio** (`journal:()=>vPortfolio('Bet Journal')` [dashboard.html:642](static/dashboard.html#L642)); dead `.dtabs`/`demoTable`.

---

## 2. Prioritized TODO

### 🔴 CRITICAL — fabricated data presented as model output ("no fake numbers should remain")

**C1 — Confidence Breakdown donut is a hardcoded constant.**
- **Problem:** `FACTOR_WEIGHTS` (Recent Form 30 / Matchup 25 / Usage 20 / Injuries 10 / Weather 5 / Other 10) is a fixed array (`model_health.py:24`) rendered identically for every prop, in the dashboard donut (`renderDonut` [dashboard.html:409](static/dashboard.html#L409)) and the drawer (`FACTORS` [dashboard.html:645](static/dashboard.html#L645)). The center shows real confidence; the slices are invented.
- **Why it matters:** It's the single most-visible "analysis" in the drawer and it's fake — the exact thing this audit exists to kill. Undermines the "measured, not vibes" positioning.
- **Solution:** Expose the **real** components of `confidence_score` (`valuation.py:74-79`): Sample Size (0.5·`n/30`), Decisiveness (0.3·`2|p−.5|`), Method (0.2·engine-vs-empirical). Return them per-prop from `/api/simulation` (and in `_drop`), and render those three real contributions. Relabel the section "What drives this confidence."
- **Effort:** M (½ day — backend expose + frontend swap).

**C2 — Correlated Props shows fixed 59/49/55 for every player.**
- **Problem:** `corrHTML` ([dashboard.html:657](static/dashboard.html#L657)) and the Correlations page ([dashboard.html:587](static/dashboard.html#L587)) hardcode H·R 59%, H·RBI 49%, R·RBI 55% — the model's *default priors* — while the page text claims a per-player matrix is "computed live... visible in each combo prop's drawer." It is not.
- **Why it matters:** Presents a global constant as this player's measured dependence; the copy makes a claim the UI contradicts.
- **Solution:** Either (a) surface the real per-player matrix (`projector_bridge` computes it for combos) via `/api/simulation` for H/R/RBI props, or (b) if unavailable for a given player, label it honestly ("model default priors; per-player matrix used in combo simulation") and only show it on combo/relevant props. Do (b) now, (a) when combos are surfaced.
- **Effort:** S for honest labeling; M for real per-player values.

**C3 — Data freshness is fake ("Uptime 99.99%", static "live").**
- **Problem:** Status bar prints `Uptime 99.99%` (invented) and `Projections live` (a static string, not a timestamp) — `renderStatus` [dashboard.html:520](static/dashboard.html#L520). `projections.updated_at` is fetched (`PROJUP`) but never shown. Step 7 wants real "updated X ago" for Projections/Market/Simulation/News/Weather + Model Version.
- **Why it matters:** Users "should always know how fresh the data is"; a fabricated uptime is worse than nothing.
- **Solution:** Delete the uptime line. Render real relative timestamps from the payloads' `updated_at` (projections, dashboard, weather) + `code_sha`/model version from `/api/version`. Add a small "Updated Xm ago" chip to the Projections header and the drawer.
- **Effort:** S.

**C4 — `loadResearch` leaks raw JSON to users.**
- **Problem:** Fallback prints `JSON.stringify(a).slice(0,300)` into the drawer ([dashboard.html:692](static/dashboard.html#L692)) when no summary exists.
- **Why it matters:** Users see raw object text — looks broken.
- **Solution:** Replace with a real Recent-Games table from `/api/analytics` (`recent` array already exists) + a clean empty state. Folds into the Step-2 "Recent Games" gap.
- **Effort:** M (pairs with the drawer Recent-Games/Trend build).

### 🟠 HIGH — spec gaps that block "production quality"

**H1 — Sidebar redesign (Step 10).** Flat 18-item list, no grouping, **collapse button is decorative** (`.collapse` has no handler, [dashboard.html:159](static/dashboard.html#L159)), no persisted state, no section headers, no version footer. Build the grouped/collapsible sidebar (Dashboard / Research / Analysis / My Workspace / Community / Bottom), persist collapsed state (localStorage), tooltips when collapsed, active highlighting (exists), NEW/PRO/BETA badges (partial), "Juiced v2.0" footer. **Effort:** M.

**H2 — Remove Bet Journal, fold into Portfolio (Step 10).** `journal` route is a duplicate of `vPortfolio` ([dashboard.html:642](static/dashboard.html#L642)); both appear in nav. Remove the standalone page. **Effort:** S (nav) — but see H3 for the real Portfolio.

**H3 — Portfolio is a toy tracker, not the flagship workspace (Step 10).** `vPortfolio` ([dashboard.html:611](static/dashboard.html#L611)) is a 4-tile + basic table. Spec wants summary cards (ROI/P&L/streak/best sport…), sub-nav (Overview/My Entries/Saved Props/Notes/Performance/Exposure/History), charts, CSV export, folders, notes+tags. This is the biggest single build; needs backend for saved-props/notes/exposure. **Effort:** L (multi-session).

**H4 — Mobile layout broken.** Sidebar never collapses on small screens (Step 1 responsive). Add mobile breakpoint + off-canvas sidebar/hamburger. **Effort:** M.

**H5 — Missing "Opponent" column + matchup context.** Data exists (`analytics.py:293` `_today_opponents`, is_home, opp_pitcher). Plumb `opponent`/`is_home` into `dashboard._drop` and add the column + a Home/Away filter. Unlocks H1-filter items too. **Effort:** M.

### 🟡 MEDIUM — usability & polish

- **M1 — Column-header sorting** (all columns), replacing/augmenting the dropdown. **S.**
- **M2 — Sticky table headers** (`thead` sticky within the scroll container). **S.**
- **M3 — Skeleton loaders** replacing "Loading…" text on every view. **S–M.**
- **M4 — Error states get a Retry button** (`render` catch [dashboard.html:698](static/dashboard.html#L698)). **S.**
- **M5 — Saved Filter Presets** (localStorage; name + restore a `PF` snapshot). **M.**
- **M6 — Game Time + Projection/Line Movement filters** (from `startTime` + CLV ledger). **M.**
- **M7 — Drawer Projection Trend + Market History** charts (analytics `recent` + `/api/lines/history`). **M.**
- **M8 — Tooltips** on metric headers (what Juice/Edge/Confidence mean). **S.**
- **M9 — De-dupe the `/api/projections?limit=1000` fetches** (cache PROJ, share with watchlist). **S.**

### 🟢 LOW — hygiene

- **L1 — Remove dead `app.html`** (148 KB orphan) or move to `/legacy`. **S.**
- **L2 — Remove dead CSS/helpers** (`.dtabs`, unused `demoTable` paths). **S.**
- **L3 — Wire ⌘K** to focus global search. **S.**
- **L4 — Team/prop in the instant search**, not just player. **S.**
- **L5 — Resizable / pinned columns** (nice-to-have; heavier in vanilla). **M.**

### 🔵 FUTURE — needs data providers or infra (not oversights)

- **F1 — Real injury feed** (probable/questionable/out) — needs a provider; today only lineup-scratch detection.
- **F2 — Per-prop weather join** (stadium→forecast→each prop) for the Weather filter — engine has park factors, not per-prop tagging.
- **F3 — News freshness** (Step 7) — no news source wired.
- **F4 — NBA/NFL/NHL models** — offseason, no lines (tracked separately).
- **F5 — Community** — needs a real signed-in user base; intentionally deferred.

---

## 3. Recommended sequence

1. **Integrity pass (C1–C4)** — kill every fake number; this is the literal ask and it's low-risk.
2. **Sidebar redesign + remove Bet Journal (H1, H2)** — high visible impact, self-contained.
3. **Freshness + Opponent/matchup + column sorting + sticky headers + skeletons + retry (C3, H5, M1–M4)** — the "production-quality" layer.
4. **Portfolio workspace (H3)** — the flagship, as a focused multi-step build.
5. Remaining MEDIUM filters/drawer charts, then LOW hygiene.

Guardrail for all of it: 56 tests must stay green, and nothing ships a number the engine didn't produce.
