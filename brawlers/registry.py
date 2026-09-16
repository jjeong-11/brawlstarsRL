"""
brawlers/registry.py
====================

One weights slot per brawler, and the rules for what actually loads.

Why this exists
---------------
A single policy has to average over every brawler in the game, and they do not
want the same thing. Edgar wants to close distance; Piper wants the opposite;
Barley wants to be behind a wall lobbing over it. Those are contradictory
optimal policies over the *same* observation vector, so one network trained on
all of them learns the mean of the contradictions -- which is nobody's strategy.

So each brawler gets its own directory:

    models/
      shelly/
        policy.zip            the RecurrentPPO weights
        vecnormalize.pkl      the reward normaliser (part of the model -- see
                              rl/train.py, resuming without it restarts the
                              running statistics from scratch)
        checkpoints/          periodic saves during that brawler's training
        meta.json             what produced these weights
      piper/
        ...

Nothing is trained yet, so `resolve()` falls back to the shared
`brawlstars_move.zip` at the repo root. The fallback is not a placeholder to be
removed later: it stays useful forever as the initialisation a new brawler's
run forks from, so brawler #40 does not start from random weights.

The important property is that the *path layout* is settled now. Once
per-brawler runs start producing files, they land where the UI and the loader
already look, and no calling code changes.

Resolution order for a brawler:
    1. models/<id>/policy.zip          -- trained for this brawler
    2. brawlstars_move.zip             -- the shared base policy
    3. nothing                         -- caller falls back to random headings

`ModelSlot.source` reports which of the three happened, so the UI can say
"base policy (not brawler-specific)" instead of quietly implying the weights
are Piper's when they are not. That distinction is the entire point of the
module: a bad *policy* and a *generic* policy look identical on screen.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent

DEFAULT_ROSTER = _HERE / "roster.json"
DEFAULT_MODELS_ROOT = _ROOT / "models"
# The policy rl/train.py currently writes. Everything forks from this until a
# brawler has weights of its own.
DEFAULT_BASE_POLICY = _ROOT / "brawlstars_move"          # .zip added by SB3

# Portraits. Brawlify's CDN is MIT-licensed, Cloudflare-cached, explicitly meant
# to be linked programmatically, and has no traffic limit or IP blocking — so
# the icons do not need to live in the repo. `tools/sync_brawlers.py` caches
# them into webui/static/icons/ for offline use; the UI prefers a local file and
# falls back to the CDN, so it works either way.
#   https://github.com/Brawlify/CDN  ·  Supercell Fan Content Policy applies.
DEFAULT_ICON_DIR = _ROOT / "webui" / "static" / "icons"
ICON_URL_TEMPLATE = "https://cdn.brawlify.com/brawlers/borderless/{scid}.png"


@dataclass(frozen=True)
class Brawler:
    """One roster entry."""

    id: str                     # slug used on disk and in URLs: "el_primo"
    name: str                   # display name: "EL PRIMO"
    rarity: str = "Unknown"
    scid: Optional[int] = None  # Supercell's brawler id, 16000010
    # Clones, summons and pets are read by the perception stack as additional
    # players. Every anchor, enemy-count and kill-attribution number measured on
    # a multi-body brawler is therefore suspect -- the same reason the old
    # Sirius recordings were deleted rather than kept as fixtures. The UI warns
    # before starting a session on one.
    multi_body: bool = False

    @property
    def display(self) -> str:
        return self.name


@dataclass(frozen=True)
class ModelSlot:
    """Where a brawler's weights live, and which file will actually load."""

    brawler_id: str
    dir: pathlib.Path             # models/<id>/
    policy_path: pathlib.Path     # models/<id>/policy.zip   (may not exist yet)
    vecnorm_path: pathlib.Path    # models/<id>/vecnormalize.pkl
    checkpoint_dir: pathlib.Path  # models/<id>/checkpoints/
    meta_path: pathlib.Path       # models/<id>/meta.json
    resolved: Optional[pathlib.Path]   # what load() will open, or None
    source: str                   # "brawler" | "base" | "none"

    @property
    def is_brawler_specific(self) -> bool:
        return self.source == "brawler"

    @property
    def label(self) -> str:
        """One line for the UI. Never implies base weights are brawler-tuned."""
        if self.source == "brawler":
            return f"{self.brawler_id} weights ({self.policy_path.name})"
        if self.source == "base":
            return f"shared base policy ({self.resolved.name}) — not yet tuned for {self.brawler_id}"
        return "no weights — random headings (planner still runs)"

    def to_dict(self) -> dict:
        return {
            "brawler_id": self.brawler_id,
            "dir": str(self.dir),
            "policy_path": str(self.policy_path),
            "resolved": str(self.resolved) if self.resolved else None,
            "source": self.source,
            "is_brawler_specific": self.is_brawler_specific,
            "label": self.label,
            "has_vecnormalize": self.vecnorm_path.exists(),
        }


class BrawlerRegistry:
    """The roster plus its per-brawler weights layout."""

    def __init__(self,
                 roster_path: Optional[pathlib.Path] = None,
                 models_root: Optional[pathlib.Path] = None,
                 base_policy: Optional[pathlib.Path] = None,
                 icon_dir: Optional[pathlib.Path] = None) -> None:
        self.roster_path = pathlib.Path(roster_path or DEFAULT_ROSTER)
        self.models_root = pathlib.Path(models_root or DEFAULT_MODELS_ROOT)
        self.base_policy = pathlib.Path(base_policy or DEFAULT_BASE_POLICY)
        self.icon_dir = pathlib.Path(icon_dir or DEFAULT_ICON_DIR)
        self.icon_url_template = ICON_URL_TEMPLATE
        self._brawlers: Dict[str, Brawler] = {}
        self._order: List[str] = []
        self._rarity_order: List[str] = []
        self.reload()

    # --- roster ----------------------------------------------------------- #
    def reload(self) -> None:
        """Re-read roster.json. Cheap; call it after editing the file."""
        doc = json.loads(self.roster_path.read_text(encoding="utf-8"))
        entries: Iterable[dict] = doc.get("brawlers", [])
        self._rarity_order = list(doc.get("rarity_order", []))
        self.icon_url_template = doc.get("icon_url_template") or ICON_URL_TEMPLATE
        self._brawlers, self._order = {}, []
        for e in entries:
            b = Brawler(
                id=str(e["id"]),
                name=str(e.get("name", e["id"])),
                rarity=str(e.get("rarity", "Unknown")),
                scid=e.get("scid"),
                multi_body=bool(e.get("multi_body", False)),
            )
            self._brawlers[b.id] = b
            self._order.append(b.id)

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, brawler_id: object) -> bool:
        return brawler_id in self._brawlers

    def all(self) -> List[Brawler]:
        return [self._brawlers[i] for i in self._order]

    def get(self, brawler_id: str) -> Brawler:
        try:
            return self._brawlers[brawler_id]
        except KeyError:
            raise KeyError(
                f"unknown brawler {brawler_id!r}. "
                f"Add it to {self.roster_path.name} — nothing else needs to change."
            ) from None

    def grouped(self) -> "list[tuple[str, list[Brawler]]]":
        """(rarity, brawlers) in roster order, for a grouped <select>."""
        buckets: Dict[str, List[Brawler]] = {}
        for b in self.all():
            buckets.setdefault(b.rarity, []).append(b)
        known = [r for r in self._rarity_order if r in buckets]
        rest = sorted(r for r in buckets if r not in self._rarity_order)
        return [(r, buckets[r]) for r in known + rest]

    # --- weights ---------------------------------------------------------- #
    def slot(self, brawler_id: str) -> ModelSlot:
        """Where this brawler's weights live and which file resolves.

        Read-only: creates nothing. Call :meth:`prepare` when you are about to
        train and actually want the directory.
        """
        self.get(brawler_id)                      # validate
        d = self.models_root / brawler_id
        policy = d / "policy.zip"
        base = self._base_policy_file()

        if policy.exists():
            resolved, source = policy, "brawler"
        elif base is not None:
            resolved, source = base, "base"
        else:
            resolved, source = None, "none"

        return ModelSlot(
            brawler_id=brawler_id,
            dir=d,
            policy_path=policy,
            vecnorm_path=d / "vecnormalize.pkl",
            checkpoint_dir=d / "checkpoints",
            meta_path=d / "meta.json",
            resolved=resolved,
            source=source,
        )

    # --- portraits -------------------------------------------------------- #
    def local_icon(self, brawler_id: str) -> Optional[pathlib.Path]:
        """The cached portrait, if `tools/sync_brawlers.py --icons` fetched it."""
        b = self.get(brawler_id)
        if b.scid is None:
            return None
        p = self.icon_dir / f"{b.scid}.png"
        return p if p.exists() else None

    def icon_url(self, brawler_id: str) -> Optional[str]:
        """The CDN URL for this brawler's portrait, or None without an scid."""
        b = self.get(brawler_id)
        if b.scid is None:
            return None
        return self.icon_url_template.format(scid=b.scid, id=b.id)

    def _base_policy_file(self) -> Optional[pathlib.Path]:
        """The shared fallback, tolerating a path given with or without .zip."""
        p = self.base_policy
        for cand in (p if p.suffix == ".zip" else p.with_suffix(".zip"), p):
            if cand.exists() and cand.is_file():
                return cand
        return None

    def prepare(self, brawler_id: str) -> ModelSlot:
        """Create models/<id>/ (and checkpoints/) ready for a training run."""
        slot = self.slot(brawler_id)
        slot.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if not slot.meta_path.exists():
            b = self.get(brawler_id)
            slot.meta_path.write_text(json.dumps({
                "brawler": b.id,
                "name": b.name,
                "rarity": b.rarity,
                "scid": b.scid,
                "trained": False,
                "forked_from": str(self._base_policy_file() or ""),
                "note": "created by BrawlerRegistry.prepare(); no training yet",
            }, indent=2) + "\n", encoding="utf-8")
        return slot

    def trained(self) -> List[str]:
        """Brawlers that have real weights of their own (not the base)."""
        return [b.id for b in self.all() if (self.models_root / b.id / "policy.zip").exists()]

    def load_policy(self, brawler_id: str, verbose: bool = True):
        """Load this brawler's policy, or the base, or return None.

        None is not a dead end: `rl/actions.decode_action` still runs the whole
        planner on a random heading, which exercises relocation, A*, clearance
        and gas escape against real frames. It just tells you nothing about the
        policy, so the caller must say which case it is in.
        """
        slot = self.slot(brawler_id)
        if slot.resolved is None:
            if verbose:
                print(f"[brawlers] no weights for {brawler_id} — random headings")
            return None, slot
        try:
            try:
                from sb3_contrib import RecurrentPPO as _Algo   # matches train.py
            except Exception:
                from stable_baselines3 import PPO as _Algo
            # SB3 appends .zip itself and rejects the path if it is already there.
            model = _Algo.load(str(slot.resolved.with_suffix("")))
            if verbose:
                print(f"[brawlers] {brawler_id}: loaded {slot.label}")
            return model, slot
        except Exception as e:
            # A shape mismatch here means the saved policy predates a change to
            # the action space or observation layout -- same failure rl/train.py
            # explains on resume. Do not crash the session over it.
            if verbose:
                print(f"[brawlers] could not load {slot.resolved.name} for "
                      f"{brawler_id} ({e}) — random headings")
            import dataclasses
            return None, dataclasses.replace(slot, resolved=None, source="none")


# A module-level default so callers do not each build their own.
_DEFAULT: Optional[BrawlerRegistry] = None


def default_registry() -> BrawlerRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = BrawlerRegistry()
    return _DEFAULT


if __name__ == "__main__":       # python -m brawlers.registry
    reg = default_registry()
    print(f"{len(reg)} brawlers from {reg.roster_path}")
    trained = reg.trained()
    print(f"trained: {len(trained)} {trained or '(none yet)'}")
    cached = sum(1 for b in reg.all() if reg.local_icon(b.id))
    print(f"icons:   {cached}/{len(reg)} cached locally "
          f"(the rest load from the CDN; `python tools/sync_brawlers.py --icons` "
          f"caches them)")
    for rarity, group in reg.grouped():
        print(f"\n{rarity} ({len(group)})")
        for b in group:
            s = reg.slot(b.id)
            flag = " [multi-body]" if b.multi_body else ""
            print(f"  {b.id:<18} {s.source:<8} {b.name}{flag}")
