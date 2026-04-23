# Lessons Learned

Deployment, infrastructure, and tooling incidents — and the fixes that
informed the current setup. Append (don't rewrite) when something burns.

---

## Streamlit Cloud — silent "preparing" hang from missing transitive deps

**Symptom:** App build hangs in **preparing** state with no error in the logs.

**Cause:** Streamlit Cloud installs **only** what is listed in
`requirements.txt`. Transitive dependencies that the app actually imports
(directly or via `app.py`'s use of pandas display tooling) are not pulled in
automatically. In the keratoconus-predictor sibling project, missing
`pandas`, `ipython`, and `sparklines` caused this exact hang.

**Fix / current state:** `requirements.txt` lists every direct *and*
indirectly-used package. If you add a new feature that imports anything
new, pin it explicitly here too — don't rely on transitive install.

---

## Hiding the GitHub source link — two settings, two scopes

**Context:** Streamlit Cloud injects a "View source" / GitHub badge into
the deployed app's hamburger menu by default. We wanted to remove it.

**What works at which layer:**

| Layer | Setting | Effect | Limitation |
|---|---|---|---|
| Repo (`.streamlit/config.toml`) | `[client] toolbarMode = "viewer"` | Hides developer toolbar items (Deploy button, "Rerun", etc.) | Does **not** hide the Cloud-injected GitHub badge |
| Cloud dashboard | Settings → General → "Show app source code" → off | Removes the GitHub badge from the menu | Repo is still public; URL can be guessed |
| GitHub | `gh repo edit --visibility private` | Source actually unreadable to non-collaborators | Requires re-authorizing Streamlit Cloud with `repo` scope; private-repo limit on free tier |

**Lesson:** "Hide the source link" and "protect the source" are different
problems. If the repo is public, hiding the badge is cosmetic — anyone with
the GitHub username can browse to the repo. Choose the layer based on
whether you care about discovery or about access.

---

## Local dev environment — `shap_env` is the right base, just missing two

**Context:** First-time local run on this machine. The system Python and
the parent project's `stagingAndProgression` env didn't both have streamlit
*and* shap conveniently. The keratoconus/SHAP-friendly env on this box is
`shap_env` (Python 3.11 under miniconda).

**Ready in `shap_env`:** numpy, pandas, scikit-learn, joblib, shap,
matplotlib, ipython.

**Missing (install once):** `streamlit`, `sparklines`.

```bash
/home/lgmendes/miniconda3/envs/shap_env/bin/pip install "streamlit>=1.28.0" "sparklines>=0.4.2"
/home/lgmendes/miniconda3/envs/shap_env/bin/streamlit run app.py
```

**Lesson:** Don't make a fresh venv for this — `shap_env` already carries
the heavy deps (shap + scikit-learn pinned to a working pair). Adding
streamlit on top costs ~80 MB and a few seconds.

---

## Streamlit Cloud subdomains are global and effectively permanent

**Context:** Picking the App URL (`<name>.streamlit.app`).

**Constraints:**
- Subdomain namespace is shared across all of Streamlit Community Cloud,
  not scoped to your account.
- Once taken, the subdomain is hard to release cleanly — even after you
  delete the app, the name may be cached or held.
- Renaming later means a new URL; any links / QR codes / posters printed
  for ARVO are stale.

**Lesson:** Pick a name that survives the next paper too. For a
research-group app expecting follow-up scenarios (12-month variant,
progression model), date-stamp or scope the subdomain
(`dr-…-arvo2026`) so the next model can claim its own URL without
conflict, instead of grabbing a generic name like `dr-predictor` that
locks you in.
