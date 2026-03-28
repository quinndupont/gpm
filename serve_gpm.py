#!/usr/bin/env python3
"""
Local GPM test server: chat (educator / poet), metrics, educator↔poet revision loop.
Run from repo root: python serve_gpm.py [--port 11435] [--config config/inference_config.yaml]

Requires: pip install llama-cpp-python pyyaml
Import order: SwappingPipeline pulls in pipeline first (Jinja chat-template patch).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
import time
import types
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "inference_config.yaml"

# Pipeline / Jinja patch before llama_cpp is constructed anywhere
from scripts.inference.pipeline import _infer_short_from_gguf_path  # noqa: E402
from scripts.inference.swapping_pipeline import SwappingPipeline  # noqa: E402
from scripts.training.model_registry import DEFAULT_STOP_TOKENS, stop_tokens_for  # noqa: E402

from models.prompts.loader import get_persona  # noqa: E402
from scripts.benchmarks.rev_flux.line_change import (
    lines_changed_per_round,
    revision_line_word_edit_details,
    revision_round_changes,
    revision_round_word_changes,
    words_changed_per_round,
)


def load_yaml_config():
    import yaml

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def list_gguf_models() -> list[str]:
    models_dir = ROOT / "models"
    if not models_dir.is_dir():
        return []
    out = []
    for p in sorted(models_dir.glob("*.gguf")):
        try:
            rel = p.relative_to(ROOT)
            out.append(str(rel).replace("\\", "/"))
        except ValueError:
            continue
    return out


def resolve_model_path(s: str) -> str:
    """Return absolute path under ROOT for a repo-relative or models/ path."""
    s = (s or "").strip()
    if not s:
        raise ValueError("model path required")
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((ROOT / s.lstrip("./")).resolve())


def stop_tokens_for_path(model_path: str, role: str) -> list[str]:
    cfg = load_yaml_config()
    if role == "educator":
        edu = cfg.get("educator", {})
        stop = edu.get("generation_brief", {}).get("stop") or edu.get("critique", {}).get("stop")
    else:
        poet = cfg.get("poet", {})
        stop = poet.get("generation", {}).get("stop") or poet.get("revision", {}).get("stop")
    if stop:
        return stop
    short = _infer_short_from_gguf_path(model_path)
    return stop_tokens_for(short_name=short) if short else DEFAULT_STOP_TOKENS


def default_generation_for_role(role: str) -> dict:
    cfg = load_yaml_config()
    if role == "educator":
        g = cfg.get("educator", {}).get("generation_brief") or cfg.get("educator", {}).get(
            "critique", {},
        )
        return {
            "temperature": g.get("temperature", 0.4),
            "max_tokens": g.get("max_tokens", 2048),
            "top_p": g.get("top_p", 0.9),
            "repeat_penalty": g.get("repeat_penalty", 1.1),
        }
    g = cfg.get("poet", {}).get("generation") or {}
    return {
        "temperature": g.get("temperature", 0.8),
        "max_tokens": g.get("max_tokens", 4096),
        "top_p": g.get("top_p", 0.95),
        "repeat_penalty": g.get("repeat_penalty", 1.15),
    }


def merge_llama_load_defaults(role: str) -> dict:
    cfg = load_yaml_config()
    sec = cfg.get("educator" if role == "educator" else "poet", {})
    return {
        "n_ctx": sec.get("n_ctx", 32768),
        "n_gpu_layers": sec.get("n_gpu_layers", -1),
        "n_threads": sec.get("n_threads", 8),
        "use_mmap": sec.get("use_mmap", True),
        "use_mlock": sec.get("use_mlock", False),
    }


# Single-threaded server: one cached Llama for chat
_chat_llama = None
_chat_cache_key: tuple | None = None


def _get_chat_llama(model_path: str, load: dict):
    global _chat_llama, _chat_cache_key
    key = (model_path, json.dumps(load, sort_keys=True))
    if _chat_cache_key == key and _chat_llama is not None:
        return _chat_llama
    if _chat_llama is not None:
        del _chat_llama
        _chat_llama = None
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("pip install llama-cpp-python")
    load_kw = {**load, "model_path": model_path, "verbose": False}
    _chat_llama = Llama(**load_kw)
    _chat_cache_key = key
    return _chat_llama


def iter_revision_loop_events(
    pipe: SwappingPipeline,
    user_request: str,
    max_revisions: int,
    stop_on_approval: bool,
):
    """Educator revision mode only: stream phase events (mirrors PoetryPipeline.generate)."""
    if not stop_on_approval:
        def _no_approve(self, critique, draft="", brief=""):
            return False

        pipe._educator_approves = types.MethodType(_no_approve, pipe)

    t0 = time.perf_counter()
    yield {"event": "start", "t_sec": 0.0}

    t1 = time.perf_counter()
    brief = pipe._educator_generate(pipe._build_brief_prompt(user_request), task="brief")
    yield {"event": "brief", "text": brief, "phase_sec": round(time.perf_counter() - t1, 3)}

    t2 = time.perf_counter()
    draft = pipe._poet_generate(brief)
    yield {"event": "draft", "text": draft, "phase_sec": round(time.perf_counter() - t2, 3)}

    revision_history: list = []
    approved = False
    approved_at_round: int | None = None

    for i in range(max_revisions):
        t_crit = time.perf_counter()
        crit_prompt = pipe._build_critique_prompt(draft, brief, revision_history)
        diag = getattr(pipe, "_last_implementation_diagnostic", "") or ""
        if diag.strip():
            yield {
                "event": "implementation_diagnostic",
                "round": i + 1,
                "text": diag.strip(),
            }
        critique = pipe._educator_generate(crit_prompt, task="critique")
        revision_history.append({"draft": draft, "critique": critique, "iteration": i})
        yield {
            "event": "critique",
            "round": i + 1,
            "text": critique,
            "phase_sec": round(time.perf_counter() - t_crit, 3),
        }

        if stop_on_approval and pipe._educator_approves(critique, draft=draft, brief=brief):
            approved = True
            approved_at_round = i + 1
            yield {"event": "approved", "round": i + 1}
            break

        prev_history = revision_history[:-1]
        past_summary = pipe._summarize_critique_history(prev_history) if prev_history else ""

        t_rb = time.perf_counter()
        revision_brief = pipe._educator_generate(
            pipe._build_revision_brief_prompt(
                draft, critique, brief, prev_history, "", False, past_summary,
            ),
            task="revision_brief",
        )
        yield {
            "event": "revision_brief",
            "round": i + 1,
            "text": revision_brief,
            "phase_sec": round(time.perf_counter() - t_rb, 3),
        }

        t_rev = time.perf_counter()
        revision_prompt = pipe._build_poet_revision_prompt(
            draft, critique, revision_brief, prev_history,
            "", False, past_summary, brief=brief,
        )
        draft = pipe._poet_generate(revision_prompt, is_revision=True)
        yield {
            "event": "revised_draft",
            "round": i + 1,
            "text": draft,
            "phase_sec": round(time.perf_counter() - t_rev, 3),
        }

    # RevFlux-style revision analytics (line + word).
    # Serve_gpm's revision_history stores the drafts that were critiqued; when the loop
    # ends, we may need to append the final poem so diffs cover the full trajectory.
    revision_history_full = list(revision_history)
    if revision_history_full and (revision_history_full[-1].get("draft") != draft):
        revision_history_full.append(
            {"draft": draft, "critique": None, "iteration": "final"},
        )

    revisions_required = len([h for h in revision_history_full if h.get("critique")])
    graduation = 1 if approved else 0

    line_rounds = revision_round_changes(revision_history_full)
    word_rounds = revision_round_word_changes(revision_history_full)
    lines_changed = lines_changed_per_round(line_rounds, threshold=0.5)
    words_changed = words_changed_per_round(word_rounds, threshold=0.5)

    per_revision = []
    for r in range(1, len(revision_history_full)):
        line_list = line_rounds[r] if r < len(line_rounds) else []
        word_list = word_rounds[r] if r < len(word_rounds) else []
        avg_line = (sum(line_list) / len(line_list)) if line_list else 0.0
        avg_word = (sum(word_list) / len(word_list)) if word_list else 0.0
        per_revision.append(
            {
                "round": r,
                "lines_changed": lines_changed[r] if r < len(lines_changed) else 0,
                "words_changed": words_changed[r] if r < len(words_changed) else 0,
                "avg_line_change_pct": round(float(avg_line), 3),
                "avg_word_change_pct": round(float(avg_word), 3),
            }
        )

    per_line_details = revision_line_word_edit_details(
        revision_history_full,
        line_threshold=0.5,
        word_threshold=0.5,
        token_limit=6,
        max_lines_per_round=12,
    )

    # Aggregate top inserted/deleted tokens across changed lines (per transition round).
    def _top_items(c: Counter, limit: int = 10) -> list[list[int | str]]:
        items = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[: max(0, limit)]
        return [[tok, cnt] for tok, cnt in items]

    top_word_edits_per_round = []
    for r in range(len(revision_history_full)):
        if r == 0:
            top_word_edits_per_round.append({"inserted": [], "deleted": []})
            continue
        c_ins: Counter = Counter()
        c_del: Counter = Counter()
        for d in per_line_details[r] if r < len(per_line_details) else []:
            for tok, cnt in d.get("inserted_tokens", []):
                c_ins[tok] += int(cnt)
            for tok, cnt in d.get("deleted_tokens", []):
                c_del[tok] += int(cnt)
        top_word_edits_per_round.append(
            {"inserted": _top_items(c_ins), "deleted": _top_items(c_del)}
        )

    rev_flux = {
        "revisions_required": revisions_required,
        "graduation": graduation,
        "approved_at_round": approved_at_round,
        "max_revisions": max_revisions,
        "per_revision": per_revision,
        "per_line_details": per_line_details,
        "top_word_edits_per_round": top_word_edits_per_round,
    }

    yield {
        "event": "done",
        "final_poem": draft,
        "approved": approved,
        "rev_flux": rev_flux,
        "total_sec": round(time.perf_counter() - t0, 3),
    }


def ndjson_chunk(line: str) -> bytes:
    encoded = (line + "\n").encode("utf-8")
    return f"{len(encoded):x}\r\n".encode() + encoded + b"\r\n"


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>GPM local tester</title>
  <style>
    :root { font-family: system-ui, sans-serif; background: #111; color: #e8e8e8; }
    body { max-width: 900px; margin: 0 auto; padding: 1rem; }
    h1 { font-size: 1.25rem; font-weight: 600; }
    .tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }
    .tabs button {
      background: #2a2a2a; border: 1px solid #444; color: #ddd;
      padding: 0.4rem 0.8rem; cursor: pointer; border-radius: 4px;
    }
    .tabs button.active { background: #3d5a80; border-color: #5a7ab0; }
    section.panel { display: none; }
    section.panel.active { display: block; }
    label { display: block; margin-top: 0.6rem; font-size: 0.85rem; color: #aaa; }
    textarea, input, select {
      width: 100%; box-sizing: border-box; background: #1a1a1a; color: #eee;
      border: 1px solid #444; border-radius: 4px; padding: 0.5rem;
    }
    textarea { min-height: 120px; font-family: ui-monospace, monospace; font-size: 0.85rem; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
    @media (max-width: 640px) { .row { grid-template-columns: 1fr; } }
    .metrics {
      background: #1e2a1e; border: 1px solid #354; padding: 0.5rem 0.75rem;
      border-radius: 4px; font-size: 0.8rem; margin-top: 0.75rem;
    }
    .revflux {
      white-space: pre-wrap;
      font-family: ui-monospace, monospace;
      font-size: 0.78rem;
      line-height: 1.25;
    }
    .log {
      background: #0d0d0d; border: 1px solid #333; padding: 0.75rem;
      max-height: 320px; overflow: auto; white-space: pre-wrap; font-size: 0.8rem;
      margin-top: 0.5rem;
    }
    button.run {
      margin-top: 0.75rem; background: #2d6a4f; border: 1px solid #40916c;
      color: #fff; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer;
    }
    button.run:disabled { opacity: 0.5; cursor: not-allowed; }
    .hint { font-size: 0.75rem; color: #777; margin-top: 0.25rem; }
  </style>
</head>
<body>
  <h1>GPM local tester</h1>
  <div class="tabs">
    <button type="button" class="tab active" data-tab="edu">Educator chat</button>
    <button type="button" class="tab" data-tab="poet">Poet chat</button>
    <button type="button" class="tab" data-tab="loop">Educator ↔ Poet loop</button>
  </div>

  <section id="panel-edu" class="panel active">
    <label>Model (GGUF)</label>
    <select id="edu-model"></select>
    <div class="row">
      <div>
        <label>n_ctx</label>
        <input type="number" id="edu-n-ctx" data-k="edu-n_ctx"/>
        <label>n_gpu_layers (-1 = all)</label>
        <input type="number" id="edu-n-gpu" data-k="edu-n_gpu"/>
      </div>
      <div>
        <label>n_threads</label>
        <input type="number" id="edu-n-threads" data-k="edu-n_threads"/>
        <label>temperature / max_tokens</label>
        <div style="display:flex;gap:0.5rem">
          <input type="number" step="0.05" id="edu-temp" data-k="edu-temp"/>
          <input type="number" id="edu-max-tok" data-k="edu-max-tok"/>
        </div>
      </div>
    </div>
    <label>Messages (JSON array optional — or type user message below)</label>
    <textarea id="edu-messages" placeholder='[{"role":"user","content":"Critique this line: ..."}]'></textarea>
    <label>User message (if messages empty)</label>
    <textarea id="edu-user" placeholder="Your prompt..."></textarea>
    <button class="run" type="button" id="edu-send">Send</button>
    <div class="metrics" id="edu-metrics"></div>
    <div class="log" id="edu-out"></div>
  </section>

  <section id="panel-poet" class="panel">
    <label>Model (GGUF)</label>
    <select id="poet-model"></select>
    <div class="row">
      <div>
        <label>n_ctx</label>
        <input type="number" id="poet-n-ctx" data-k="poet-n_ctx"/>
        <label>n_gpu_layers</label>
        <input type="number" id="poet-n-gpu" data-k="poet-n_gpu"/>
      </div>
      <div>
        <label>n_threads</label>
        <input type="number" id="poet-n-threads" data-k="poet-n_threads"/>
        <label>temperature / max_tokens</label>
        <div style="display:flex;gap:0.5rem">
          <input type="number" step="0.05" id="poet-temp" data-k="poet-temp"/>
          <input type="number" id="poet-max-tok" data-k="poet-max-tok"/>
        </div>
      </div>
    </div>
    <label>Messages or user message</label>
    <textarea id="poet-messages" placeholder='[{"role":"user","content":"Write a sonnet about..."}]'></textarea>
    <textarea id="poet-user" placeholder="Or single user prompt..."></textarea>
    <button class="run" type="button" id="poet-send">Send</button>
    <div class="metrics" id="poet-metrics"></div>
    <div class="log" id="poet-out"></div>
  </section>

  <section id="panel-loop" class="panel">
    <div class="row">
      <div>
        <label>Educator GGUF</label>
        <select id="loop-edu-model"></select>
      </div>
      <div>
        <label>Poet GGUF</label>
        <select id="loop-poet-model"></select>
      </div>
    </div>
    <label>Poem request</label>
    <textarea id="loop-request" placeholder="Write a villanelle about..."></textarea>
    <div class="row">
      <div>
        <label>max revisions (rounds of critique → revise)</label>
        <input type="number" id="loop-max-rev" min="1" max="10" value="3" data-k="loop-max-rev"/>
      </div>
      <div>
        <label><input type="checkbox" id="loop-stop-approval" checked data-k="loop-stop-approval"/> Stop when educator approves (e.g. poem has found its shape)</label>
      </div>
    </div>
    <button class="run" type="button" id="loop-run">Run loop</button>
    <div class="metrics" id="loop-metrics"></div>
    <div class="metrics revflux" id="loop-revflux"></div>
    <div class="log" id="loop-out"></div>
    <p class="hint">Loads one model at a time (swapping). Uses educator revision brief → poet revises.</p>
  </section>

<script>
const LS = "gpm_serve_ui_v1";
function loadLS() {
  try { return JSON.parse(localStorage.getItem(LS) || "{}"); } catch { return {}; }
}
function saveLS(patch) {
  const o = loadLS();
  Object.assign(o, patch);
  localStorage.setItem(LS, JSON.stringify(o));
}
function bindPersist(inp) {
  const k = inp.dataset.k;
  if (!k) return;
  if (inp.type === "checkbox") {
    inp.addEventListener("change", () => saveLS({ [k]: inp.checked }));
  } else {
    inp.addEventListener("change", () => saveLS({ [k]: inp.value }));
  }
}

function numOrUndef(el) {
  const v = (el && el.value != null) ? String(el.value).trim() : "";
  if (v === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

async function fetchModels() {
  const r = await fetch("/api/models");
  const j = await r.json();
  return j.models || [];
}

function fillSelect(sel, models, selected) {
  sel.innerHTML = "";
  models.forEach(m => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  });
  if (selected && models.includes(selected)) sel.value = selected;
  else if (models.length) sel.value = models[0];
}

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("panel-" + btn.dataset.tab).classList.add("active");
  });
});

(async function init() {
  const [models, d] = await Promise.all([
    fetchModels(),
    fetch("/api/defaults").then(r => r.json()),
  ]);
  const o = loadLS();
  const edu = d.educator || {}, poet = d.poet || {};
  document.getElementById("edu-n-ctx").value = o["edu-n_ctx"] ?? edu.n_ctx ?? "";
  document.getElementById("edu-n-gpu").value = o["edu-n_gpu"] ?? edu.n_gpu_layers ?? "";
  document.getElementById("edu-n-threads").value = o["edu-n_threads"] ?? edu.n_threads ?? "";
  document.getElementById("edu-temp").value = o["edu-temp"] ?? edu.temperature ?? "";
  document.getElementById("edu-max-tok").value = o["edu-max-tok"] ?? edu.max_tokens ?? "";
  document.getElementById("poet-n-ctx").value = o["poet-n_ctx"] ?? poet.n_ctx ?? "";
  document.getElementById("poet-n-gpu").value = o["poet-n_gpu"] ?? poet.n_gpu_layers ?? "";
  document.getElementById("poet-n-threads").value = o["poet-n_threads"] ?? poet.n_threads ?? "";
  document.getElementById("poet-temp").value = o["poet-temp"] ?? poet.temperature ?? "";
  document.getElementById("poet-max-tok").value = o["poet-max-tok"] ?? poet.max_tokens ?? "";
  if (o["loop-max-rev"] != null) document.getElementById("loop-max-rev").value = o["loop-max-rev"];
  if (o["loop-stop-approval"] != null) document.getElementById("loop-stop-approval").checked = o["loop-stop-approval"];

  ["edu-model", "poet-model", "loop-edu-model", "loop-poet-model"].forEach(id => {
    const sel = document.getElementById(id);
    const key = id + "-sel";
    fillSelect(sel, models, o[key]);
    sel.addEventListener("change", () => saveLS({ [key]: sel.value }));
  });
  document.querySelectorAll("input[data-k], textarea[data-k]").forEach(bindPersist);
})();

function parseMessages(raw, fallbackUser) {
  const t = raw.trim();
  if (t.startsWith("[")) return JSON.parse(t);
  return [{ role: "user", content: fallbackUser.trim() || t }];
}

function asciiBar(pct, width) {
  const v = Number.isFinite(pct) ? pct : 0;
  const n = Math.max(0, Math.min(width, Math.round((v / 100) * width)));
  return "#".repeat(n) + "-".repeat(width - n);
}

function formatTokPairs(pairs) {
  // pairs: [[token, count], ...]
  if (!Array.isArray(pairs)) return "";
  return pairs
    .slice(0, 8)
    .map(x => {
      const tok = (x && x[0] != null) ? String(x[0]) : "";
      const cnt = (x && x[1] != null) ? String(x[1]) : "0";
      return tok + "(" + cnt + ")";
    })
    .join(" ");
}

function renderRevFluxText(rv) {
  if (!rv) return "";
  const gradTxt = rv.graduation === 1 ? "APPROVED" : "NOT_APPROVED";
  const lines = [];
  lines.push("RevFlux revision analytics");
  lines.push("Graduation: " + gradTxt + "  |  Revisions required: " + rv.revisions_required);
  if (rv.approved_at_round != null) lines.push("Approved at round: " + rv.approved_at_round);
  if (rv.max_revisions != null) lines.push("Max revisions: " + rv.max_revisions);
  lines.push("");

  if (Array.isArray(rv.per_revision) && rv.per_revision.length) {
    lines.push("Per transition round (draft[r-1] -> draft[r])");
    lines.push("Round  LinesChg WordsChg  AvgLine% AvgWord%  LineBar           WordBar");
    rv.per_revision.forEach(pr => {
      const r = pr.round;
      const lc = pr.lines_changed ?? 0;
      const wc = pr.words_changed ?? 0;
      const al = (pr.avg_line_change_pct ?? 0).toFixed(1);
      const aw = (pr.avg_word_change_pct ?? 0).toFixed(1);
      const lineBar = asciiBar(pr.avg_line_change_pct, 18);
      const wordBar = asciiBar(pr.avg_word_change_pct, 18);
      lines.push(
        String(r).padStart(5) + "  " +
        String(lc).padStart(9) + " " +
        String(wc).padStart(8) + "  " +
        String(al).padStart(8) + "  " +
        String(aw).padStart(8) + "  " +
        lineBar + "  " + wordBar
      );
    });
    lines.push("");
  } else {
    lines.push("No poet revision transitions to diff (approval may have happened immediately).");
    lines.push("");
  }

  if (Array.isArray(rv.per_line_details)) {
    lines.push("Changed lines + top word edits (per transition round)");
    const maxRounds = rv.per_line_details.length;
    for (let r = 1; r < maxRounds; r++) {
      const details = rv.per_line_details[r] || [];
      if (!details.length) continue;
      lines.push("");
      lines.push("Round " + r + " changed lines: " + details.length);
      details.slice(0, 6).forEach(d => {
        const li = d.line_idx;
        const lp = (d.line_pct ?? 0).toFixed(1);
        const wp = (d.word_pct ?? 0).toFixed(1);
        const from = d.from_line != null ? String(d.from_line) : "";
        const to = d.to_line != null ? String(d.to_line) : "";
        lines.push("  L" + li + ": " + lp + "% line | " + wp + "% words");
        lines.push("    -" + from);
        lines.push("    +" + to);
        if (d.deleted_tokens || d.inserted_tokens) {
          const del = formatTokPairs(d.deleted_tokens);
          const ins = formatTokPairs(d.inserted_tokens);
          if (del) lines.push("    del: " + del);
          if (ins) lines.push("    ins: " + ins);
        }
      });
    }
  }

  return lines.join("\\n");
}

async function streamChat(role, opts) {
  const out = document.getElementById(role === "educator" ? "edu-out" : "poet-out");
  const met = document.getElementById(role === "educator" ? "edu-metrics" : "poet-metrics");
  out.textContent = "";
  met.textContent = "…";
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  if (!res.ok) { out.textContent = "HTTP " + res.status; return; }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\\n");
    buf = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const j = JSON.parse(line);
        if (j.message && j.message.content) out.textContent += j.message.content;
        if (j.metrics) met.textContent = JSON.stringify(j.metrics, null, 2);
        if (j.error) out.textContent += "\\n[error] " + j.error;
      } catch (_) { out.textContent += line + "\\n"; }
    }
  }
}

document.getElementById("edu-send").addEventListener("click", async () => {
  const model = document.getElementById("edu-model").value;
  const messages = parseMessages(
    document.getElementById("edu-messages").value,
    document.getElementById("edu-user").value
  );
  const loadEdu = {
    n_ctx: numOrUndef(document.getElementById("edu-n-ctx")),
    n_gpu_layers: numOrUndef(document.getElementById("edu-n-gpu")),
    n_threads: numOrUndef(document.getElementById("edu-n-threads")),
  };
  await streamChat("educator", {
    role: "educator",
    model,
    messages,
    load: loadEdu,
    generation: {
      temperature: numOrUndef(document.getElementById("edu-temp")),
      max_tokens: numOrUndef(document.getElementById("edu-max-tok")),
    },
  });
});

document.getElementById("poet-send").addEventListener("click", async () => {
  const model = document.getElementById("poet-model").value;
  const messages = parseMessages(
    document.getElementById("poet-messages").value,
    document.getElementById("poet-user").value
  );
  await streamChat("poet", {
    role: "poet",
    model,
    messages,
    load: {
      n_ctx: numOrUndef(document.getElementById("poet-n-ctx")),
      n_gpu_layers: numOrUndef(document.getElementById("poet-n-gpu")),
      n_threads: numOrUndef(document.getElementById("poet-n-threads")),
    },
    generation: {
      temperature: numOrUndef(document.getElementById("poet-temp")),
      max_tokens: numOrUndef(document.getElementById("poet-max-tok")),
    },
  });
});

document.getElementById("loop-run").addEventListener("click", async () => {
  const out = document.getElementById("loop-out");
  const met = document.getElementById("loop-metrics");
  const rfd = document.getElementById("loop-revflux");
  out.textContent = "";
  met.textContent = "…";
  rfd.textContent = "";
  const body = {
    user_request: document.getElementById("loop-request").value,
    educator_model: document.getElementById("loop-edu-model").value,
    poet_model: document.getElementById("loop-poet-model").value,
    max_revisions: +document.getElementById("loop-max-rev").value || 3,
    stop_on_approval: document.getElementById("loop-stop-approval").checked,
  };
  const res = await fetch("/api/revision-loop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) { out.textContent = "HTTP " + res.status; return; }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\\n");
    buf = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const j = JSON.parse(line);
        out.textContent += "\\n---\\n[" + (j.event || "?") + "]\\n" +
          (j.text != null ? j.text : JSON.stringify(j)) + "\\n";
        if (j.event === "done") {
          met.textContent = JSON.stringify(j, null, 2);
          if (j.rev_flux) rfd.textContent = renderRevFluxText(j.rev_flux);
        }
      } catch (e) { out.textContent += line + "\\n"; }
    }
  }
});
</script>
</body>
</html>
"""


class GPMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(args[0] if args else format)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            data = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/models":
            models = list_gguf_models()
            body = json.dumps({"models": models}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/defaults":
            cfg = load_yaml_config()
            edu, poet = cfg.get("educator") or {}, cfg.get("poet") or {}
            payload = {
                "educator": {
                    "n_ctx": edu.get("n_ctx", 32768),
                    "n_gpu_layers": edu.get("n_gpu_layers", -1),
                    "n_threads": edu.get("n_threads", 8),
                    **default_generation_for_role("educator"),
                },
                "poet": {
                    "n_ctx": poet.get("n_ctx", 32768),
                    "n_gpu_layers": poet.get("n_gpu_layers", -1),
                    "n_threads": poet.get("n_threads", 8),
                    **default_generation_for_role("poet"),
                },
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json_error(400, "Invalid JSON")
            return

        if path == "/api/chat":
            self._handle_chat(data)
            return
        if path == "/api/revision-loop":
            self._handle_revision_loop(data)
            return
        self.send_response(404)
        self.end_headers()

    def _json_error(self, code, msg):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": msg}).encode("utf-8"))

    def _handle_chat(self, data):
        messages = data.get("messages", [])
        role = (data.get("role") or "educator").lower()
        if role not in ("educator", "poet"):
            self._json_error(400, "role must be educator or poet")
            return
        if not messages:
            self._json_error(400, "messages required")
            return

        model_rel = data.get("model") or ""
        try:
            model_path = resolve_model_path(model_rel)
        except ValueError as e:
            self._json_error(400, str(e))
            return

        load_in = data.get("load") or {}
        load = merge_llama_load_defaults(role)
        for k in ("n_ctx", "n_gpu_layers", "n_threads", "use_mmap", "use_mlock"):
            if k in load_in and load_in[k] is not None:
                load[k] = load_in[k]

        gen_in = data.get("generation") or {}
        gen = default_generation_for_role(role)
        for k in ("temperature", "max_tokens", "top_p", "repeat_penalty"):
            if k in gen_in and gen_in[k] is not None:
                gen[k] = gen_in[k]

        if role == "educator":
            try:
                system = get_persona("educator_neutral")
            except FileNotFoundError:
                system = get_persona("educator_condensed")
        else:
            system = get_persona("poet")
        stops = stop_tokens_for_path(model_path, role)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            llm = _get_chat_llama(model_path, load)
            stream = llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, *messages],
                stream=True,
                stop=stops,
                **gen,
            )
            t0 = time.perf_counter()
            first_token_t = None
            n_chars = 0
            for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                    n_chars += len(content)
                    line = json.dumps({"message": {"content": content}})
                    self.wfile.write(ndjson_chunk(line))
                    self.wfile.flush()
            t1 = time.perf_counter()
            gen_time = (t1 - first_token_t) if first_token_t else (t1 - t0)
            est_tokens = max(1, n_chars // 4)
            tok_s = est_tokens / gen_time if gen_time > 0 else 0.0
            metrics = {
                "role": role,
                "model": model_rel,
                "first_token_ms": round((first_token_t - t0) * 1000, 1) if first_token_t else None,
                "total_sec": round(t1 - t0, 3),
                "gen_sec": round(gen_time, 3) if first_token_t else round(t1 - t0, 3),
                "est_tokens": est_tokens,
                "tok_per_sec": round(tok_s, 1),
            }
            self.wfile.write(ndjson_chunk(json.dumps({"metrics": metrics, "done": True})))
        except Exception as e:
            self.wfile.write(ndjson_chunk(json.dumps({"error": str(e), "done": True})))
        self.wfile.write(b"0\r\n\r\n")

    def _handle_revision_loop(self, data):
        user_request = (data.get("user_request") or "").strip()
        if not user_request:
            self._json_error(400, "user_request required")
            return
        try:
            edu_path = resolve_model_path(data.get("educator_model") or "")
            poet_path = resolve_model_path(data.get("poet_model") or "")
        except ValueError as e:
            self._json_error(400, str(e))
            return

        max_revisions = int(data.get("max_revisions") or 3)
        max_revisions = max(1, min(20, max_revisions))
        stop_on_approval = data.get("stop_on_approval", True)

        edu_load = data.get("educator_load") or {}
        poet_load = data.get("poet_load") or {}

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            pipe = SwappingPipeline(
                config_path=CONFIG_PATH,
                educator_model_override=f"gguf:{edu_path}",
                poet_model_override=f"gguf:{poet_path}",
                educator_load_overrides=edu_load if edu_load else None,
                poet_load_overrides=poet_load if poet_load else None,
            )
            pipe.revision_mode = "educator"

            for ev in iter_revision_loop_events(
                pipe, user_request, max_revisions, stop_on_approval,
            ):
                self.wfile.write(ndjson_chunk(json.dumps(ev)))
                self.wfile.flush()
        except Exception as e:
            self.wfile.write(ndjson_chunk(json.dumps({"event": "error", "message": str(e)})))
        self.wfile.write(b"0\r\n\r\n")


def main():
    global CONFIG_PATH
    parser = argparse.ArgumentParser(description="GPM local test server")
    parser.add_argument("port", nargs="?", type=int, default=11435)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    CONFIG_PATH = args.config.resolve()
    server = HTTPServer(("127.0.0.1", args.port), GPMHandler)
    print(f"GPM server http://127.0.0.1:{args.port}/ (POST /api/chat, /api/revision-loop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
