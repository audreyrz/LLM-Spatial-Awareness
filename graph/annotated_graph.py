"""
distance_graph.py

Build a spatial graph directly from annotations that already carry
origin / destination / distance -- no LLM extraction needed for this task,
since the schema gives you the structured fields directly:

    text, start_ind, end_ind, type ("Literal" | "Hypothetical" | ...),
    tense, origin, destination, transportation_mode, scale,
    distance, distance_unit, characters, scale_uncertainty

This is a separate, self-contained pipeline from spatial_graph.py (the
LLM-based extractor for plain movement spans) -- use this one whenever your
annotations already resolve origin/destination.

Usage
    python distance_graph.py --annotations task2/annotations.json --out out2/ --plot

    # merge known name variants ("bone-gray dock" == "dock") by hand:
    python distance_graph.py --annotations ... --aliases aliases.json --out out2/ --plot

    # also include Hypothetical / non-Literal events as a separate edge style:
    python distance_graph.py --annotations ... --include-non-literal --out out2/ --plot

    pip install networkx matplotlib
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_annotations(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    anns = data["annotations"] if isinstance(data, dict) and "annotations" in data else data
    return sorted(anns, key=lambda a: a.get("start_ind", 0))


def load_aliases(path: Optional[str]) -> dict[str, str]:
    """JSON file: {"raw name variant": "canonical name", ...}. Matching is case-insensitive."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k.strip().lower(): v.strip() for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Name normalization
# --------------------------------------------------------------------------- #
def norm_key(name: str) -> str:
    """Lowercase key used only for equality/lookup -- never shown to the user."""
    name = re.sub(r"^(the|a|an)\s+", "", name.strip().lower())
    return re.sub(r"\s+", " ", name)


class PlaceRegistry:
    """Tracks canonical display names. Two mentions merge only if they share a
    normalized key or are explicitly aliased -- never guessed automatically,
    since "edge of dock" and "dock" are related but NOT the same place."""

    def __init__(self, aliases: dict[str, str]):
        self.aliases = aliases          # normalized-variant -> canonical string
        self.canonical_by_key: dict[str, str] = {}   # normalized key -> display name

    def resolve(self, raw: Optional[str]) -> Optional[str]:
        if raw is None or not raw.strip():
            return None
        raw = raw.strip()
        key = norm_key(raw)
        if key in self.aliases:
            raw = self.aliases[key]
            key = norm_key(raw)
        if key not in self.canonical_by_key:
            self.canonical_by_key[key] = raw       # first-seen casing wins
        return self.canonical_by_key[key]

    def suggest_near_duplicates(self, threshold: float = 0.72) -> list[tuple[str, str, float]]:
        """Flag place-name pairs that LOOK related but weren't merged, so a human
        can add them to an aliases.json file. Never auto-merges these."""
        names = list(self.canonical_by_key.values())
        out = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ka, kb = norm_key(a), norm_key(b)
                if ka == kb:
                    continue
                ratio = difflib.SequenceMatcher(None, ka, kb).ratio()
                contains = ka in kb or kb in ka
                if contains or ratio >= threshold:
                    out.append((a, b, round(ratio, 2)))
        return sorted(out, key=lambda t: -t[2])


# --------------------------------------------------------------------------- #
# Distance normalization
# --------------------------------------------------------------------------- #
_TO_METERS = {
    "m": 1, "meter": 1, "meters": 1, "metre": 1, "metres": 1,
    "km": 1000, "kilometer": 1000, "kilometers": 1000,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "mi": 1609.34, "mile": 1609.34, "miles": 1609.34,
}


def distance_in_meters(distance: Optional[float], unit: Optional[str]) -> Optional[float]:
    if distance is None or not unit or unit.lower() == "none":
        return None
    factor = _TO_METERS.get(unit.strip().lower())
    return round(distance * factor, 3) if factor else None


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph(
    anns: list[dict],
    aliases: dict[str, str],
    include_non_literal: bool = True,
    fill_sequential_gaps: bool = True,
) -> tuple[nx.MultiDiGraph, dict, list]:
    reg = PlaceRegistry(aliases)
    G = nx.MultiDiGraph()
    trajectories: dict[str, list[dict]] = defaultdict(list)
    unresolved: list[dict] = []           # events missing origin and/or destination -- kept for inspection
    last_place: dict[str, str] = {}       # character -> last known place, for gap-filling

    def add_node(place: str, ann: dict):
        if place not in G:
            G.add_node(place, mentions=[])
        G.nodes[place]["mentions"].append([ann["start_ind"], ann["end_ind"]])

    for idx, ann in enumerate(anns):
        origin = reg.resolve(ann.get("origin"))
        dest = reg.resolve(ann.get("destination"))
        ev_type = (ann.get("type") or "literal").strip().lower()
        is_literal = ev_type == "literal"
        characters = ann.get("characters") or ["unknown"]
        dist_m = distance_in_meters(ann.get("distance"), ann.get("distance_unit"))

        if not is_literal and not include_non_literal:
            continue
        if origin is None and dest is None:
            unresolved.append({**ann, "reason": "no origin or destination resolved"})
            continue

        for p in (origin, dest):
            if p:
                add_node(p, ann)

        edge_attrs = dict(
            kind="traversal" if is_literal else ev_type,     # "traversal" | "hypothetical" | ...
            is_literal=is_literal,
            characters=characters,
            transportation_mode=ann.get("transportation_mode"),
            tense=ann.get("tense"),
            scale=ann.get("scale"),
            scale_uncertainty=ann.get("scale_uncertainty"),
            distance=ann.get("distance"),
            distance_unit=ann.get("distance_unit"),
            distance_m=dist_m,
            span=[ann["start_ind"], ann["end_ind"]],
            text=ann.get("text"),
        )

        # gap-filling: connect a character's last known place to this event's
        # origin if they don't match (handles events with an empty destination
        # upstream, e.g. "escorted us off the dock" leaves location unresolved
        # until the next event names a new origin).
        if fill_sequential_gaps and origin:
            for c in characters:
                prev = last_place.get(c)
                if prev and prev != origin:
                    G.add_edge(prev, origin, kind="sequential", characters=[c],
                               span=[ann["start_ind"], ann["end_ind"]])

        if origin and dest and origin != dest:
            G.add_edge(origin, dest, **edge_attrs)
        elif origin and dest and origin == dest:
            G.add_edge(origin, dest, **{**edge_attrs, "kind": "no_op"})
        else:
            unresolved.append({**ann, "reason": "missing origin" if not origin else "missing destination"})

        # per-character trajectory + last-known-place update
        for c in characters:
            for role, p in (("origin", origin), ("destination", dest)):
                if p:
                    trajectories[c].append({
                        "event": idx, "place": p, "role": role,
                        "span": [ann["start_ind"], ann["end_ind"]],
                        "type": ev_type, "text": ann.get("text"),
                    })
            last_place[c] = dest or origin or last_place.get(c)

    return G, dict(trajectories), unresolved


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export(G: nx.MultiDiGraph, trajectories: dict, unresolved: list, near_dupes: list, out: Path, plot: bool):
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectories.json").write_text(json.dumps(trajectories, indent=2, ensure_ascii=False))
    (out / "unresolved_events.json").write_text(json.dumps(unresolved, indent=2, ensure_ascii=False))
    (out / "place_alias_suggestions.json").write_text(json.dumps(
        [{"a": a, "b": b, "similarity": r} for a, b, r in near_dupes], indent=2, ensure_ascii=False
    ))

    H = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **{k: json.dumps(v) for k, v in d.items()})
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
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(nx.Graph(G), seed=0, k=1.2)
    style = {
        "traversal": ("black", "solid"),
        "sequential": ("gray", "dashed"),
        "hypothetical": ("tab:red", "dotted"),
        "no_op": ("lightgray", "dotted"),
    }
    nx.draw_networkx_nodes(G, pos, node_color="#f3e9d2", edgecolors="black", node_size=1800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    seen_kinds = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        kind = d.get("kind", "traversal")
        color, ls = style.get(kind, ("tab:orange", "dashdot"))
        seen_kinds.add(kind)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=color, style=ls, arrows=True,
            connectionstyle="arc3,rad=0.15", node_size=1800, ax=ax,
        )
        if d.get("distance_m"):
            mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
            ax.annotate(f"{d['distance_m']:.0f}m", (mx, my), fontsize=7, color="tab:blue")
    ax.legend(
        handles=[Line2D([0], [0], color=style.get(k, ("tab:orange", "dashdot"))[0],
                         ls=style.get(k, ("tab:orange", "dashdot"))[1], label=k) for k in seen_kinds],
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
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--aliases", default=None, help="JSON file: {\"raw name\": \"canonical name\"}")
    ap.add_argument("--out", default="out2")
    ap.add_argument("--include-non-literal", action="store_true", default=True)
    ap.add_argument("--literal-only", dest="include_non_literal", action="store_false")
    ap.add_argument("--no-gap-fill", dest="fill_sequential_gaps", action="store_false", default=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    anns = load_annotations(args.annotations)
    aliases = load_aliases(args.aliases)

    G, traj, unresolved = build_graph(
        anns, aliases,
        include_non_literal=args.include_non_literal,
        fill_sequential_gaps=args.fill_sequential_gaps,
    )
    reg_placeholder = PlaceRegistry(aliases)
    for a in anns:  # rebuild registry state for near-dup detection (cheap, and keeps build_graph pure)
        reg_placeholder.resolve(a.get("origin"))
        reg_placeholder.resolve(a.get("destination"))
    near_dupes = reg_placeholder.suggest_near_duplicates()

    export(G, traj, unresolved, near_dupes, out, args.plot)

    print(f"{len(anns)} annotations -> {G.number_of_nodes()} places, {G.number_of_edges()} edges, "
          f"{len(unresolved)} unresolved, {len(near_dupes)} possible-duplicate name pairs")
    for u, v, d in G.edges(data=True):
        dist = f" ({d['distance_m']}m)" if d.get("distance_m") else ""
        print(f"  {u} -[{d.get('kind')}:{d.get('transportation_mode') or ''}]-> {v}{dist}")
    if near_dupes:
        print("\nPossible duplicate place names (see place_alias_suggestions.json):")
        for a, b, r in near_dupes[:10]:
            print(f"  '{a}' ~ '{b}' (similarity {r})")


if __name__ == "__main__":
    main()