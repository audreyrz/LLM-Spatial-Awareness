"""
spatial_graph.py

Turn annotated movement phrases from a novel excerpt into a place graph.

Pipeline
    1. Load full text + annotation spans (start_ind / end_ind into the text).
    2. For each span (in text order), give an LLM the span plus surrounding
       context, the places seen so far, and the last few events, and get back a
       structured MoveEvent.  Results are cached to JSONL so reruns are free.
    3. Build a networkx MultiDiGraph:
         nodes  = places
         edges  = traversal (explicit, within one event),
                  sequential (inferred, between consecutive events of one actor),
                  contained_in (place hierarchy)
    4. Export GraphML + JSON + per-actor trajectories (+ optional PNG).

Usage (free / open-source, runs locally; no API key)
    # Option A: Ollama (easiest)
    #   install from https://ollama.com, then:  ollama pull qwen2.5:7b-instruct
    python spatial_graph.py \
        --text allTheLightWeCannotSee_excerpt.txt \
        --annotations consensus_final_allTheLightWeCannotSee.json \
        --backend ollama --model qwen2.5:7b-instruct --out out/ --plot

    # Option B: any OpenAI-compatible local server (vLLM, llama.cpp server, LM Studio)
    #   --backend openai --base-url http://localhost:8000/v1 --model <name>

    # Option C: Hugging Face transformers in-process (needs a GPU for anything big)
    #   --backend hf --model Qwen/Qwen2.5-7B-Instruct

    pip install pydantic networkx matplotlib        # (+ transformers torch for --backend hf)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

Direction = Literal["up", "down", "in", "out", "left", "right", "forward", "back"]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class MoveEvent(BaseModel):
    span_start: int = 0
    span_end: int = 0
    span_text: str = ""
    actor: Optional[str] = None            # who moves ("Werner")
    is_real_movement: bool = True          # False for intentions / negations / hypotheticals
    motion: str = ""                       # lemma-ish: lap, pass, walk_up, lead, retreat
    source: Optional[str] = None           # place moved from (None if not inferable)
    via: Optional[str] = None              # place/feature passed through or by
    destination: Optional[str] = None      # place moved to
    direction: Optional[Direction] = None
    parent_place: Optional[str] = None     # containing place of destination ("apartment #5")
    place_hedged: bool = False             # "what must be a kitchen"
    repeat_count: Optional[int] = None     # "three times" -> 3
    confidence: float = Field(default=0.5, ge=0, le=1)
    rationale: str = ""                    # short justification; useful for debugging


# --------------------------------------------------------------------------- #
# Loading + alignment
# --------------------------------------------------------------------------- #
def load_inputs(text_path: str, ann_path: str) -> tuple[str, list[dict]]:
    text = Path(text_path).read_text(encoding="utf-8")
    anns = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    anns = sorted(anns, key=lambda a: a["start_ind"])
    return text, anns


def align_span(text: str, ann: dict, search_radius: int = 300) -> tuple[int, int, bool]:
    """Check text[start:end] == ann['text']; if not, look for the phrase nearby.
    Returns (start, end, exact_match)."""
    s, e, t = ann["start_ind"], ann["end_ind"], ann["text"]
    if text[s:e] == t:
        return s, e, True
    lo = max(0, s - search_radius)
    hi = min(len(text), e + search_radius)
    idx = text.find(t, lo, hi)
    if idx != -1:
        return idx, idx + len(t), False
    return s, e, False  # give up; keep the given offsets


def context_window(text: str, s: int, e: int, before: int = 600, after: int = 250) -> str:
    """Context with the target span marked by << >>."""
    lo, hi = max(0, s - before), min(len(text), e + after)
    return text[lo:s] + "<<" + text[s:e] + ">>" + text[e:hi]


# --------------------------------------------------------------------------- #
# LLM extraction
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You convert annotated movement phrases from fiction into structured spatial events.

You are given a passage in which ONE movement phrase is marked with <<double angle brackets>>.
Fill in the schema for that phrase only, using the surrounding text to resolve what the
phrase leaves implicit (e.g. "he walks up" -> which building/floor, who is "he").

Rules:
- Place names: short, lowercase noun phrases, e.g. "block", "apartment building", "apartment #5",
  "drawing room", "kitchen". No articles, no possessives unless needed to disambiguate.
- REUSE a name from KNOWN_PLACES when the text refers to the same place. Only invent a new name
  for a genuinely new place. Never merge places that are different (kitchen != drawing room).
- Only fill source / via / destination when the text supports it. Use null otherwise. Do NOT guess.
- via = something passed by/through en route (storefront, swinging door), not an endpoint.
- Loops (laps the block): source = destination = the looped place, repeat_count = number of laps.
- is_real_movement = false for intentions, offers, negations, hypotheticals ("I was about to go").
- parent_place: the larger place containing the destination, if evident (drawing room -> apartment #5).
- confidence in [0,1] reflects how well the text supports source/destination.
- actor: the character who moves, by name if resolvable from context (not a pronoun).

Return ONLY a JSON object with keys:
actor, is_real_movement, motion, source, via, destination, direction, parent_place,
place_hedged, repeat_count, confidence, rationale.
direction is one of: up, down, in, out, left, right, forward, back, or null.
No prose, no markdown fences."""


def build_user_prompt(context: str, known_places: list[str], recent: list[MoveEvent]) -> str:
    recent_str = "\n".join(
        f'- {ev.actor}: "{ev.span_text}" -> src={ev.source}, via={ev.via}, dst={ev.destination}'
        for ev in recent
    ) or "(none)"
    return (
        f"KNOWN_PLACES: {json.dumps(known_places)}\n\n"
        f"PREVIOUS_EVENTS:\n{recent_str}\n\n"
        f"PASSAGE:\n{context}"
    )


def parse_json_loose(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    return json.loads(m.group(0) if m else raw)


def _post_json(url: str, payload: dict, headers: Optional[dict] = None, timeout: int = 300) -> dict:
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def make_ollama_extractor(model: str, host: str = "http://localhost:11434") -> Callable[[str, str], str]:
    """Ollama native API. format='json' constrains output to valid JSON (helps small models a lot)."""

    def call(system: str, user: str) -> str:
        out = _post_json(f"{host.rstrip('/')}/api/chat", {
            "model": model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 4096},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })
        return out["message"]["content"]

    return call


def make_openai_compat_extractor(model: str, base_url: str, api_key: str = "none") -> Callable[[str, str], str]:
    """Any OpenAI-compatible server: vLLM, llama.cpp `llama-server`, LM Studio, text-generation-webui..."""

    def call(system: str, user: str) -> str:
        out = _post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            {"model": model, "temperature": 0, "max_tokens": 600,
             "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        return out["choices"][0]["message"]["content"]

    return call


def make_hf_extractor(model: str, max_new_tokens: int = 500) -> Callable[[str, str], str]:
    """Run a Hugging Face instruct model locally via transformers."""
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model, device_map="auto", torch_dtype="auto")

    def call(system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = pipe(msgs, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
        return out[0]["generated_text"]

    return call


def make_extractor(args) -> Callable[[str, str], str]:
    if args.backend == "ollama":
        return make_ollama_extractor(args.model, args.host)
    if args.backend == "openai":
        return make_openai_compat_extractor(args.model, args.base_url)
    if args.backend == "hf":
        return make_hf_extractor(args.model)
    raise ValueError(args.backend)


def extract_events(
    text: str,
    anns: list[dict],
    llm: Callable[[str, str], str],
    cache_path: Optional[Path] = None,
    n_recent: int = 3,
    retries: int = 2,
) -> list[MoveEvent]:
    """Run the LLM over each span in order, carrying place registry + recent events forward."""
    cache: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[f'{rec["span_start"]}:{rec["span_end"]}'] = rec

    events: list[MoveEvent] = []
    known: list[str] = []
    cache_f = cache_path.open("a", encoding="utf-8") if cache_path else None

    try:
        for i, ann in enumerate(anns):
            s, e, exact = align_span(text, ann)
            if not exact:
                print(f"[warn] span {i} offsets don't match text exactly: {ann['text']!r}", file=sys.stderr)
            key = f"{s}:{e}"

            if key in cache:
                ev = MoveEvent(**cache[key])
            else:
                prompt = build_user_prompt(context_window(text, s, e), known, events[-n_recent:])
                ev, err = None, None
                for attempt in range(retries + 1):  # small local models sometimes emit bad JSON
                    try:
                        data = parse_json_loose(llm(SYSTEM_PROMPT, prompt))
                        data = {k: v for k, v in data.items() if k in MoveEvent.model_fields}
                        ev = MoveEvent(**data)
                        break
                    except (json.JSONDecodeError, ValidationError, TypeError, KeyError) as e_:
                        err = e_
                if ev is None:
                    print(f"[warn] span {i} failed to parse ({err}); recording as low-confidence stub", file=sys.stderr)
                    ev = MoveEvent(is_real_movement=False, confidence=0.0, rationale=f"parse failure: {err}")
                ev.span_start, ev.span_end, ev.span_text = s, e, text[s:e] or ann["text"]
                if cache_f:
                    cache_f.write(ev.model_dump_json() + "\n")
                    cache_f.flush()

            events.append(ev)
            for p in (ev.source, ev.via, ev.destination, ev.parent_place):
                if p and p not in known:
                    known.append(p)
    finally:
        if cache_f:
            cache_f.close()
    return events


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def norm(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = re.sub(r"^(the|a|an)\s+", "", name.strip().lower())
    return re.sub(r"\s+", " ", name) or None


def event_place_sequence(ev: MoveEvent) -> list[tuple[str, str]]:
    """Ordered (place, role) pairs this event visits: source -> via -> destination."""
    seq = []
    for role, p in (("source", ev.source), ("via", ev.via), ("destination", ev.destination)):
        p = norm(p)
        if p:
            seq.append((p, role))
    return seq


def build_graph(
    events: list[MoveEvent],
    min_conf: float = 0.0,
    sequential_edges: bool = True,
    max_sequential_gap: int = 1500,
) -> tuple[nx.MultiDiGraph, dict[str, list[dict]]]:
    G = nx.MultiDiGraph()
    trajectories: dict[str, list[dict]] = defaultdict(list)
    last_place: dict[str, tuple[str, MoveEvent]] = {}  # actor -> (last place, its event)

    def add_place(p: str, ev: MoveEvent, hedged: bool = False):
        if p not in G:
            G.add_node(p, first_seen=ev.span_start, mentions=[], hedged=False)
        G.nodes[p]["mentions"].append([ev.span_start, ev.span_end])
        G.nodes[p]["hedged"] = G.nodes[p]["hedged"] or hedged

    for idx, ev in enumerate(events):
        if not ev.is_real_movement or ev.confidence < min_conf:
            continue
        actor = ev.actor or "unknown"
        seq = event_place_sequence(ev)
        for p, role in seq:
            # "place_hedged" ("what must be a kitchen") describes the destination only
            add_place(p, ev, hedged=ev.place_hedged and role == "destination")

        # containment
        parent, dest = norm(ev.parent_place), norm(ev.destination)
        if parent and dest and parent != dest:
            add_place(parent, ev)
            G.add_edge(dest, parent, kind="contained_in", span=[ev.span_start, ev.span_end])

        # trajectory record
        for p, role in seq:
            trajectories[actor].append(
                {"event": idx, "place": p, "role": role, "span": [ev.span_start, ev.span_end], "motion": ev.motion}
            )

        # sequential edge from actor's previous place into this event's first place
        if sequential_edges and seq and actor in last_place:
            prev_p, prev_ev = last_place[actor]
            first_p = seq[0][0]
            gap = ev.span_start - prev_ev.span_end
            nested = G.has_edge(prev_p, first_p) and any(
                d.get("kind") == "contained_in" for d in G.get_edge_data(prev_p, first_p).values()
            ) or G.has_edge(first_p, prev_p) and any(
                d.get("kind") == "contained_in" for d in G.get_edge_data(first_p, prev_p).values()
            )
            if prev_p != first_p and gap <= max_sequential_gap and not nested:
                G.add_edge(
                    prev_p, first_p, kind="sequential", actor=actor, gap_chars=gap,
                    motion=ev.motion, direction=ev.direction, conf=ev.confidence,
                    span=[ev.span_start, ev.span_end],
                )

        # explicit edges within event
        attrs = dict(
            kind="traversal", actor=actor, motion=ev.motion, direction=ev.direction,
            conf=ev.confidence, span=[ev.span_start, ev.span_end], span_text=ev.span_text,
        )
        if seq and ev.source and ev.destination and norm(ev.source) == norm(ev.destination):
            # loop (e.g. "laps the block three times")
            p = norm(ev.source)
            G.add_edge(p, p, repeat_count=ev.repeat_count, **attrs)
        else:
            for (a, _), (b, _) in zip(seq, seq[1:]):
                if a != b:
                    G.add_edge(a, b, **attrs)

        if seq:
            last_place[actor] = (seq[-1][0], ev)

    return G, dict(trajectories)


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export(G: nx.MultiDiGraph, trajectories: dict, events: list[MoveEvent], out: Path, plot: bool):
    out.mkdir(parents=True, exist_ok=True)
    (out / "events.json").write_text(json.dumps([e.model_dump() for e in events], indent=2, ensure_ascii=False))
    (out / "trajectories.json").write_text(json.dumps(trajectories, indent=2, ensure_ascii=False))

    # GraphML can't store lists/None, so flatten a copy
    H = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **{k: json.dumps(v) if isinstance(v, list) else v for k, v in d.items()})
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, **{k: (json.dumps(x) if isinstance(x, list) else x) for k, x in d.items() if x is not None})
    nx.write_graphml(H, out / "graph.graphml")

    node_link = nx.node_link_data(G, edges="edges")
    (out / "graph.json").write_text(json.dumps(node_link, indent=2, ensure_ascii=False))

    if plot:
        plot_graph(G, out / "graph.png")


def plot_graph(G: nx.MultiDiGraph, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(nx.Graph(G), seed=0, k=1.2)
    style = {"traversal": ("black", "solid"), "sequential": ("gray", "dashed"), "contained_in": ("tab:blue", "dotted")}
    nx.draw_networkx_nodes(G, pos, node_color="#f3e9d2", edgecolors="black", node_size=1800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    for kind, (color, ls) in style.items():
        edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("kind") == kind]
        if edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, edge_color=color, style=ls, arrows=True,
                connectionstyle="arc3,rad=0.12", node_size=1800, ax=ax,
            )
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[Line2D([0], [0], color=c, ls=ls, label=k) for k, (c, ls) in style.items()],
        loc="lower left",
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--backend", choices=["ollama", "openai", "hf"], default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b-instruct",
                    help="ollama tag / server model name / HF repo id, depending on --backend")
    ap.add_argument("--host", default="http://localhost:11434", help="ollama host")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible server (--backend openai)")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--no-sequential", action="store_true", help="skip inferred edges between consecutive events")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text, anns = load_inputs(args.text, args.annotations)

    events = extract_events(text, anns, make_extractor(args), cache_path=out / "events_cache.jsonl")
    G, traj = build_graph(events, min_conf=args.min_conf, sequential_edges=not args.no_sequential)
    export(G, traj, events, out, args.plot)

    print(f"{len(events)} spans -> {G.number_of_nodes()} places, {G.number_of_edges()} edges")
    for u, v, d in G.edges(data=True):
        print(f"  {u} -[{d.get('kind')}:{d.get('motion') or ''}]-> {v}")


if __name__ == "__main__":
    main()