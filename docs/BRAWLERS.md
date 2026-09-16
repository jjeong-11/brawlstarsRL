# Brawler reference

**Generated — do not edit.** `python tools/dump_brawler_reference.py`

Source of truth: [`brawlers/roster.json`](../brawlers/roster.json) (names, rarity, multi-body) and [`brawlers/abilities.json`](../brawlers/abilities.json) (class, range, thrower flag). The code reads those; this file is the readable view of them, so it cannot drift from what the bot does.

## How aim mode is decided

Aim mode is **derived, never listed**. A hand-written list of 106 aim modes is 106 opinions that rot independently; a rule over two published facts is one opinion you can argue with and re-apply automatically after a rebalance. See [`brawlers/abilities.py`](../brawlers/abilities.py).

| Condition | Mode | Why |
| --- | --- | --- |
| class = Artillery | **manual** | arcs over walls — auto-aim will not fire without line of sight |
| class = Marksman | **manual** | slow single projectile — has to be led, and auto-aim does not lead |
| range ≥ 9.5 tiles | **manual** | travel time makes auto-aim's lack of lead costly, whatever the class |
| otherwise | **auto** | lead error is smaller than the target, and auto-aim already picks the nearest enemy in range |

That gives **24 manual**, **82 auto**.

## Confidence

Most published ranges are *bands*, not numbers, so `range_tiles` is often a band midpoint. The attack gate uses `range_max` — letting a marginal shot through costs `attack_cost` (0.05); blocking a real one costs a kill.

| Level | Meaning | Rows |
| --- | --- | --- |
| `gamefile` | read from the game's own data files. Exact. | 0 |
| `measured` | an exact tile value stated by a source, cross-checked. | 10 |
| `bucket` | only a range BAND is published, so range_tiles is the band midpoint. Good enough to gate a shot, not to tune one. | 95 |
| `unknown` | no published range yet (a brand-new brawler). The range gate treats these as permissive rather than guessing. | 1 |

Run `python tools/sync_brawler_abilities.py --write` to replace every estimate with the exact value from the game's own `characters.csv` + `skills.csv`.

## All brawlers

`aim` is derived. `range` shows the tile value and how it is known: `exact` from the game files, `exact*` stated by a cross-checked source, `band` a published range band's midpoint.

| Brawler | Rarity | Class | Range | | Aim | Notes |
| --- | --- | --- | ---: | --- | --- | --- |
| SHELLY | Starting Brawler | Damage Dealer | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| COLT | Rare | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto |  |
| BULL | Rare | Tank | 5.7 <sub>5–6.4</sub> | band | auto |  |
| BROCK | Rare | Marksman | 8.7 <sub>8–9.4</sub> | band | **manual** |  |
| RICO | Super Rare | Damage Dealer | 9.7 | exact* | **manual** |  |
| SPIKE | Legendary | Damage Dealer | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| BARLEY | Rare | Artillery | 7.2 <sub>6.5–7.9</sub> | band | **manual** | arcs over walls |
| JESSIE | Super Rare | Controller | 8.7 <sub>8–9.4</sub> | band | auto | multi-body — perception unreliable |
| NITA | Rare | Damage Dealer | 5.7 <sub>5–6.4</sub> | band | auto | multi-body — perception unreliable |
| DYNAMIKE | Super Rare | Artillery | 7.2 <sub>6.5–7.9</sub> | band | **manual** | arcs over walls |
| EL PRIMO | Rare | Tank | 2.7 <sub>2–3.4</sub> | band | auto |  |
| MORTIS | Mythic | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| CROW | Legendary | Assassin | 8.7 <sub>8–9.4</sub> | band | auto |  |
| POCO | Rare | Support | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| BO | Epic | Controller | 8.7 <sub>8–9.4</sub> | band | auto |  |
| PIPER | Epic | Marksman | 10 | exact* | **manual** |  |
| PAM | Epic | Support | 8.7 <sub>8–9.4</sub> | band | auto | multi-body — perception unreliable |
| TARA | Mythic | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto | multi-body — perception unreliable |
| DARRYL | Super Rare | Tank | 5.7 <sub>5–6.4</sub> | band | auto |  |
| PENNY | Super Rare | Artillery | 8.7 <sub>8–9.4</sub> | band | **manual** | arcs over walls, multi-body — perception unreliable |
| FRANK | Epic | Tank | 5.7 <sub>5–6.4</sub> | band | auto |  |
| GENE | Mythic | Controller | 5.7 <sub>5–6.4</sub> | band | auto |  |
| TICK | Super Rare | Artillery | 8.7 <sub>8–9.4</sub> | band | **manual** | arcs over walls, multi-body — perception unreliable |
| LEON | Legendary | Assassin | 9.7 | exact* | **manual** | multi-body — perception unreliable |
| ROSA | Rare | Tank | 4.2 <sub>3.5–4.9</sub> | band | auto |  |
| CARL | Super Rare | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto |  |
| BIBI | Epic | Tank | 4.2 <sub>3.5–4.9</sub> | band | auto |  |
| 8-BIT | Super Rare | Damage Dealer | 10 | exact* | **manual** | multi-body — perception unreliable |
| SANDY | Legendary | Controller | 5.7 <sub>5–6.4</sub> | band | auto |  |
| BEA | Epic | Marksman | 10 | exact* | **manual** |  |
| EMZ | Epic | Controller | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| MR. P | Mythic | Controller | 7.2 <sub>6.5–7.9</sub> | band | auto | multi-body — perception unreliable |
| MAX | Mythic | Support | 8.7 <sub>8–9.4</sub> | band | auto |  |
| JACKY | Super Rare | Tank | 2.7 <sub>2–3.4</sub> | band | auto |  |
| GALE | Epic | Controller | 8.7 <sub>8–9.4</sub> | band | auto |  |
| NANI | Epic | Marksman | 8.7 <sub>8–9.4</sub> | band | **manual** | multi-body — perception unreliable |
| SPROUT | Mythic | Artillery | 5.7 <sub>5–6.4</sub> | band | **manual** | arcs over walls |
| SURGE | Legendary | Damage Dealer | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| COLETTE | Epic | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto |  |
| AMBER | Legendary | Controller | 8.7 <sub>8–9.4</sub> | band | auto |  |
| LOU | Mythic | Controller | 8.7 <sub>8–9.4</sub> | band | auto |  |
| BYRON | Mythic | Support | 10 | exact* | **manual** |  |
| EDGAR | Epic | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| RUFFS | Mythic | Support | 8.7 <sub>8–9.4</sub> | band | auto |  |
| STU | Epic | Assassin | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| BELLE | Epic | Marksman | 10 | exact* | **manual** |  |
| SQUEAK | Mythic | Controller | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| GROM | Epic | Artillery | 7.2 <sub>6.5–7.9</sub> | band | **manual** | arcs over walls |
| BUZZ | Mythic | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| GRIFF | Epic | Controller | 8.7 <sub>8–9.4</sub> | band | auto |  |
| ASH | Epic | Tank | 4.2 <sub>3.5–4.9</sub> | band | auto | multi-body — perception unreliable |
| MEG | Legendary | Tank | 8.7 <sub>8–9.4</sub> | band | auto |  |
| LOLA | Epic | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto | multi-body — perception unreliable |
| FANG | Mythic | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| EVE | Mythic | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto | multi-body — perception unreliable |
| JANET | Mythic | Marksman | 4.2 <sub>3.5–4.9</sub> | band | **manual** |  |
| BONNIE | Epic | Marksman | 8.7 <sub>8–9.4</sub> | band | **manual** |  |
| OTIS | Mythic | Controller | 8.7 <sub>8–9.4</sub> | band | auto |  |
| SAM | Epic | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| GUS | Super Rare | Support | 8.7 <sub>8–9.4</sub> | band | auto | multi-body — perception unreliable |
| BUSTER | Mythic | Tank | 5.7 <sub>5–6.4</sub> | band | auto |  |
| CHESTER | Legendary | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto |  |
| GRAY | Mythic | Support | 8.7 <sub>8–9.4</sub> | band | auto |  |
| MANDY | Epic | Marksman | 8.7 <sub>8–9.4</sub> | band | **manual** |  |
| R-T | Mythic | Damage Dealer | 10 | exact* | **manual** | multi-body — perception unreliable |
| WILLOW | Mythic | Controller | 7.2 <sub>6.5–7.9</sub> | band | auto | multi-body — perception unreliable |
| MAISIE | Epic | Marksman | 8.7 <sub>8–9.4</sub> | band | **manual** |  |
| HANK | Epic | Tank | 2.7 <sub>2–3.4</sub> | band | auto |  |
| CORDELIUS | Legendary | Assassin | 5.7 <sub>5–6.4</sub> | band | auto |  |
| DOUG | Mythic | Support | 2.7 <sub>2–3.4</sub> | band | auto |  |
| PEARL | Epic | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto |  |
| CHUCK | Mythic | Damage Dealer | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| CHARLIE | Mythic | Controller | 8.7 <sub>8–9.4</sub> | band | auto | multi-body — perception unreliable |
| MICO | Mythic | Assassin | 4.2 <sub>3.5–4.9</sub> | band | auto |  |
| KIT | Legendary | Support | 4.2 <sub>3.5–4.9</sub> | band | auto | multi-body — perception unreliable |
| LARRY & LAWRIE | Epic | Artillery | 7.2 <sub>6.5–7.9</sub> | band | **manual** | arcs over walls, multi-body — perception unreliable |
| MELODIE | Mythic | Assassin | 8.7 <sub>8–9.4</sub> | band | auto |  |
| ANGELO | Epic | Marksman | 10 | exact* | **manual** |  |
| DRACO | Legendary | Tank | 4.2 <sub>3.5–4.9</sub> | band | auto |  |
| LILY | Mythic | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| BERRY | Epic | Support | 5.7 <sub>5–6.4</sub> | band | auto |  |
| CLANCY | Mythic | Damage Dealer | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| MOE | Mythic | Damage Dealer | 5.7 <sub>5–6.4</sub> | band | auto |  |
| KENJI | Legendary | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| SHADE | Epic | Assassin | 4.2 <sub>3.5–4.9</sub> | band | auto |  |
| JUJU | Mythic | Artillery | 5.7 <sub>5–6.4</sub> | band | **manual** | arcs over walls, multi-body — perception unreliable |
| MEEPLE | Epic | Controller | 7.2 <sub>6.5–7.9</sub> | band | auto | multi-body — perception unreliable |
| OLLIE | Mythic | Tank | 5.7 <sub>5–6.4</sub> | band | auto | multi-body — perception unreliable |
| LUMI | Mythic | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto |  |
| FINX | Mythic | Controller | 8.7 <sub>8–9.4</sub> | band | auto |  |
| JAE-YONG | Mythic | Support | 8.7 <sub>8–9.4</sub> | band | auto |  |
| KAZE | Ultra Legendary | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| ALLI | Mythic | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| TRUNK | Epic | Tank | 2.7 <sub>2–3.4</sub> | band | auto |  |
| MINA | Mythic | Damage Dealer | 8.7 <sub>8–9.4</sub> | band | auto |  |
| ZIGGY | Mythic | Controller | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| PIERCE | Legendary | Marksman | 10 | exact* | **manual** |  |
| GIGI | Mythic | Assassin | 2.7 <sub>2–3.4</sub> | band | auto |  |
| GLOWY | Mythic | Support | 7.2 <sub>6.5–7.9</sub> | band | auto |  |
| SIRIUS | Ultra Legendary | Controller | 7.2 <sub>6.5–7.9</sub> | band | auto | multi-body — perception unreliable |
| NAJIA | Mythic | Damage Dealer | 5.7 <sub>5–6.4</sub> | band | auto |  |
| DAMIAN | Mythic | Tank | 2.7 <sub>2–3.4</sub> | band | auto |  |
| STARR NOVA | Mythic | Assassin | 5.7 <sub>5–6.4</sub> | band | auto | multi-body — perception unreliable |
| BOLT | Epic | Tank | ? | — | auto |  |
| NORI | Legendary | Assassin | 4.2 <sub>3.5–4.9</sub> | band | auto |  |
| WENDY | Mythic | Support | 8.7 <sub>8–9.4</sub> | band | auto | multi-body — perception unreliable |

## Manual aimers, by range

| Brawler | Range | Class | Reason |
| --- | ---: | --- | --- |
| PIPER | 10 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| 8-BIT | 10 | Damage Dealer | 10 tiles — at this range the projectile's travel time makes auto-aim's lack of lead costly |
| BEA | 10 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| BYRON | 10 | Support | 10 tiles — at this range the projectile's travel time makes auto-aim's lack of lead costly |
| BELLE | 10 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| R-T | 10 | Damage Dealer | 10 tiles — at this range the projectile's travel time makes auto-aim's lack of lead costly |
| ANGELO | 10 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| PIERCE | 10 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| RICO | 9.7 | Damage Dealer | 9.7 tiles — at this range the projectile's travel time makes auto-aim's lack of lead costly |
| LEON | 9.7 | Assassin | 9.7 tiles — at this range the projectile's travel time makes auto-aim's lack of lead costly |
| BROCK | 8.7 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| PENNY | 8.7 | Artillery | arcs over walls — auto-aim will not fire without line of sight |
| TICK | 8.7 | Artillery | arcs over walls — auto-aim will not fire without line of sight |
| NANI | 8.7 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| BONNIE | 8.7 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| MANDY | 8.7 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| MAISIE | 8.7 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |
| BARLEY | 7.2 | Artillery | arcs over walls — auto-aim will not fire without line of sight |
| DYNAMIKE | 7.2 | Artillery | arcs over walls — auto-aim will not fire without line of sight |
| GROM | 7.2 | Artillery | arcs over walls — auto-aim will not fire without line of sight |
| LARRY & LAWRIE | 7.2 | Artillery | arcs over walls — auto-aim will not fire without line of sight |
| SPROUT | 5.7 | Artillery | arcs over walls — auto-aim will not fire without line of sight |
| JUJU | 5.7 | Artillery | arcs over walls — auto-aim will not fire without line of sight |
| JANET | 4.2 | Marksman | slow single projectile — has to be led, and auto-aim does not lead |

## Sources

- brawlplanet.com/tier-list/attack_range
- brawlplanet.com/brawlers
- liquipedia.net/brawlstars/Artillery + brawlstars.fandom.com + pockettactics.com

Fetched 2026-08-13. The Artillery set was cross-checked against three independent sources before being written; they agree on exactly eight brawlers.

Not affiliated with, endorsed or sponsored by Supercell.
