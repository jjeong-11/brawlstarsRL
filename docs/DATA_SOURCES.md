# Data sources: footage, battle records, CV datasets, mechanics

Where to get training data for each phase of `docs/PLAN_pro_imitation.md`, what
each source can and cannot give, and the rules that apply. Compiled 2026-11-17.

**How much to trust this page.** The research ran in a sandbox that could
reach only GitHub. YouTube, Twitch, Supercell, the Brawl Stars API, Brawlify,
Kaggle, Hugging Face, Roboflow and the wikis were all blocked, so most entries
come from **search-result snippets**, not from opening the page. Tags:

| Tag | Meaning |
|---|---|
| **[G]** | Read directly on GitHub. Highest confidence. |
| **[S]** | Seen in a search result. The URL is real; the details are only as good as the snippet. |
| **[U]** | Not verified; background knowledge or inference. A lead, not a fact. |

Not checked for any video: duration, overlays/face-cam, commentary, licence.
The listing tool below fills in duration and licence from the YouTube API.

---

## 1. What each phase needs, and the tool for it

| Need | Best source | Tool in this repo | Runs here? |
|---|---|---|---|
| First-person Solo Showdown play (Phase 2 imitation, sim calibration) | Creators' videos **with permission**, or your own recordings | `tools/list_footage.py` (metadata manifest) -> `tools/fit_from_footage.py` (local files) | No: needs a YouTube API key |
| Real Solo Showdown outcomes (base rates, which brawlers/maps to target, opponent calibration) | Official Brawl Stars API battle logs | `tools/crawl_battlelogs.py` | No: the key is bound to your IP |
| Map layouts for the localiser | Brawlify map renders + `media/showdownmaps/` | `tools/fetch_game_data.py coverage` | No: host blocked here |
| Detector training (Phase 1d) | Auto-labels from our perception + PylaAI model zoo + Roboflow sets | not yet | - |
| Mechanics numbers (speed, gas, ranges) | Game-data tables, wiki, measurement on footage | `tools/sync_brawler_abilities.py`, `fit_from_footage.py calibrate` | partly |

All three new tools run on your own machine. Their output goes under
`datasets/`, which is git-ignored: it holds other players' tags and Supercell
art, so keep it private and publish only aggregates.

```bash
# Battle records (key from developer.brawlstars.com, created for this machine's IP)
export BRAWLSTARS_API_KEY=...
python tools/crawl_battlelogs.py crawl --seed-rankings global --max-requests 2000
python tools/crawl_battlelogs.py stats --min-trophies 800          # mean place per brawler
python tools/crawl_battlelogs.py stats --by map --min-trophies 800

# Footage manifest (metadata only; YouTube Data API v3 key)
export YOUTUBE_API_KEY=...
python tools/list_footage.py --modes soloShowdown --min-minutes 3

# Map coverage of the localiser
python tools/fetch_game_data.py coverage          # this season's Showdown maps
python tools/fetch_game_data.py maps --renders    # renders to datasets/brawlify/maps/
```

`crawl_battlelogs.py` reconstructs full placements: each battle log only
records its owner's place, but the other nine players' logs contain the same
battle, so the crawler keys battles by (time, map, players) and merges.
`stats` then reports how often each brawler wins and makes the top four at a
given trophy level. That is the bar to compare the bot's mean place against.

---

## 2. First-person footage

"First-person" means the player's own screen with the full HUD (joysticks,
ammo bar, super button). That is what the perception stack reads. Casted
esports broadcasts use an observer camera and are **not** usable for it.

No channel was found offering Creative Commons licensing. Treat every video
as all rights reserved: see section 6.

### 2a. Solo Showdown, high skill (priority order)

| Source | What | Tag |
|---|---|---|
| **SoloSavage** https://www.youtube.com/@SoloSavageBS | Solo Showdown specialist, ~80k trophies | [S] |
| Showdown-only 100k pusher (uploader unidentified) https://www.youtube.com/watch?v=BNR7ZtxTgvU ; live streams https://www.youtube.com/watch?v=UDftvI7MmtE , https://www.youtube.com/watch?v=9h2PkCO3uuY | Pure Solo Showdown at 83-100k trophies; live archives = long, unedited | [S] |
| 2026 "Prestige" Solo Showdown videos (current UI, several "no teaming"): EbeGMmpM4UE, uBs93E1jCvA, 6_aBKSKTBHs, _fkliYrOn-E, mA60j4jXrLc, XRQfKFKuWYw | Best match to the current game version | [S] |
| **CryingMan** https://www.youtube.com/channel/UCGShu88Lh2ZAtXX0qbV9fXA | "Rank 35 <brawler> in Showdown" series, 2019-2024; ~1.2k videos. Older UI | [S] |
| **Trebor** https://www.youtube.com/@TreborBS (Streams tab), Twitch `TreborMi` | Top-2 world ladder; Showdown ("supervivencia") at 66k+, Duels; daily streams | [S] |
| iKaoss11, SnakeThug (@SnakeThug7), therealfreeze guides, Hyra | Strong play, more editing/commentary | [S] |
| Low-edit playlist https://www.youtube.com/playlist?list=PLik8NpWfN5Arp6njtolq-Ty3e-FeXLrJx | "low edit videos where I play solo showdown" | [S] |
| Rank-35 full-gameplay singles: P2qUvVXqbiI (Leon), GJEnvxBt_ww (Max), oZLZx2HMRic (Brock), WGUGRbfsuB8 (Piper, "1/85" series), Zqe70-_AIdI (Kit) | Expand each into its channel | [S] |

### 2b. Other modes

| Mode | Sources | Tag |
|---|---|---|
| Ranked 3v3 (Gem Grab, Brawl Ball, Bounty, Heist, Hot Zone, Knockout) | Drage (@DrageBS), Jc Buller (@jcbuller, long sessions), SpenLC, bedlam (@bedlamBS), SymantecBS, lukii, bobbyBS (player-POV **scrims**), Cori, Sitetampo, Tensai; Japanese streamers INS Jene, CR Milkreo, RUI_BS | [S] |
| Ranked, no commentary | "Season NN Update - Ranked Mode - Road to Legendary" series (e.g. aWDFPORlYUM) | [S] |
| Duels | Trebor, "200,000 trophies in Duels" (pACqmhXuXuI) | [S] |
| Duo Showdown | "RANK 35 Leon in Duo Showdown (Full Gameplay)" (oK7g6P7DtoM) | [S] |
| Rotating modes (Wipeout, Loaded Showdown, ...) | 950+ part no-commentary "Gameplay Walkthrough" series; casual skill, clean HUD: perception calibration only | [S] |
| Twitch | ~50 concurrent Brawl Stars streamers; VODs expire (retention 7-60 days [U]). TreborMi, bedlambs, cccrystal_bs, gugu11_bs, zaiko_bs | [S] |

`tools/footage_seeds.json` holds these as machine-readable seeds for
`list_footage.py`. Add channels there as you find them.

### 2c. Esports (observer camera, 3v3 only)

Brawl Stars Esports on YouTube (https://www.youtube.com/@BrawlStarsEsports/streams)
and Liquipedia (https://liquipedia.net/brawlstars/, per-match VOD links and
picks). No player-POV archive of official matches was found. Use for meta and
macro analysis (picks, positioning), not for the perception model. [S]

---

## 3. Structured data

### 3a. Official Brawl Stars API

- Portal: https://developer.brawlstars.com. Base `https://api.brawlstars.com/v1`,
  `Authorization: Bearer <key>`. **Keys are bound to fixed IP addresses**
  (403 from anywhere else); max 10 keys per account. [G]
- **No rate limit is published**; 429 means throttled. One public pipeline
  budgets 20,000 requests per 40 minutes. Start slow, back off. [G]
- Endpoints: `/players/{tag}`, `/players/{tag}/battlelog` (last ~25 battles;
  new battles can take 30 min to appear), `/rankings/{country}/players`,
  `/rankings/{country}/brawlers/{id}`, `/brawlers`, `/events/rotation`,
  `/clubs/...`. [G]
- A battle record has: time, map, mode, type, duration, the owner's `rank`
  (Showdown placement) and `trophyChange`, and every participant's tag,
  brawler, power and trophies. **No positions, damage, kills or replays.** [G]
- Terms: Supercell ToS plus the portal's own terms (not readable from here;
  read them before a large crawl).

### 3b. Brawlify (no key)

- BrawlAPI https://brawlapi.com: brawlers, maps (id, mode, image URL),
  events, game modes. `tools/sync_brawlers.py` already uses
  `api.brawlapi.com/v1/brawlers`. [S]/[G]
- CDN https://cdn.brawlify.com (repo https://github.com/Brawlify/CDN, MIT):
  `maps/regular/{mapId}.png` renders of every map; "no traffic limit and no
  IP blocking"; must follow Supercell's ToS and Fan Content Policy. [G]
- Game-data tables (`https://api.brawlify.com/game`, `csv_logic/*`): the
  client's own numbers for speed, range, HP. Which tables are served was not
  checked. [G]

### 3c. Ready-made datasets

| Dataset | What | Tag |
|---|---|---|
| **EliF77/brawlstars-ranked** (Hugging Face), pipeline https://github.com/elixf7/brawlstars-data-pipeline (MIT) | >1M ranked 3v3 sets per season as Parquet (teams, brawlers, elo, map, result). No Showdown. Dataset-card licence unverified | [G]+[S] |
| Kaggle `sirwerto/brawlstars-battlelog` (2021), `robguilarr/brawl-stars-logs-and-metadata-2023` | Old battle logs incl. Showdown. Historical only | [S] |
| Brawl Time Ninja https://brawltime.ninja | Map/brawler win rates on the site; no open data access found (the code is now private) | [G]/[S] |

---

## 4. Computer-vision data and models

| Source | What | Licence | Tag |
|---|---|---|---|
| **AngelFireLA/BrawlStarsBotMaking** (PylaAI model zoo) | Published ONNX/PT weights, Nov 2026: entity detector V3 (player/teammate/enemy, ground keypoint), terrain **segmentation** V3/V4 (wall, bush, water, rope fence, pads, ... at 480x832), game-mode elements (ball, power cube, box). Training data not published | **No licence file**: ask the author before use | [G] |
| PylaAI https://github.com/PylaAI/PylaAI | The bot those models come from | CC BY-NC 4.0 | [G] |
| Roboflow Universe: bloxxy/brawl-stars-dataset (2,551 imgs, MIT), bounding-box-data-set (1,249, CC BY 4.0, player/wall/bush/enemy), brawl-stars-bot (108, CC BY 4.0, includes **poison**), brawl-stars-segment (77, CC BY 4.0, instance seg incl. box/cube/poison), ~25 more small sets | ~8,800 images in total with overlap; older UI; expect to relabel | per set | [S] |
| BarsamErshadi/brawl-map-segmentation | Synthetic wall/bush segmentation generator | no licence file | [G] |
| Brawlify CDN map renders | Top-down ground truth for walls/bushes per map | MIT repo, Supercell art | [G] |

Plan for detector training (Phase 1d): auto-label our own recordings with the
current HSV perception plus, if the author agrees, the Pyla segmentation
model; correct by hand; train a small YOLO; export to ONNX. The noise sweep in
the plan says detection accuracy is not the bottleneck, so this stays behind
capture speed and footage.

---

## 5. Mechanics references (for `sim/` calibration)

| Topic | Value | Tag |
|---|---|---|
| Showdown gas damage | 20% of max HP per second; +5% per tick after 5 s inside; heals reduced 50% (since Apr 2023; older footage follows a flat 1000/tick rule) | [S] |
| Gas start / shrink rate | Not found. Measure from footage (`fit_from_footage.py calibrate`) | - |
| Move speed | 750 points "normal", 855 "very fast"; points-to-tiles conversion disputed (≈ 2.5-2.8 tiles/s at 750). Measure from footage | [S] |
| Per-brawler HP, damage, range | Fandom wiki pages; Brawlify game-data tables; `tools/sync_brawler_abilities.py` | [S]/[G] |
| Showdown placement as "win" | Brawlify counts places 1-4 as a win | [S] |

---

## 6. Rules: footage, automation, data

None of the Supercell pages could be opened here; re-read them before relying
on this summary.

- **Fan Content Policy** (https://supercell.com/en/fan-content-policy/):
  allows non-commercial fan content (guides, apps, gameplay videos) with the
  notice "This material is unofficial and is not endorsed by Supercell", no
  implied endorsement, no fees without approval. It licenses *Supercell's* IP
  to fan creators; it gives no rights to **other creators' videos**. [S]
- **YouTube Terms**: no downloading except through YouTube's own features or
  with permission [U]. Hence `list_footage.py` lists metadata only and the
  manifest's `rights` column starts as "ask".
- **Supercell Terms of Service**: prohibit "cheats, exploits, automation
  software", and (per the snippet's summary) emulators and bots. Running an
  agent in live matchmaking violates them and risks bans. Training on
  recordings and in our simulator does not touch the service. Live tests stay
  in Friendly Games on a test account (`DISCLAIMER.md`). [S]
- **API data** is about other players: keep the crawl database private,
  publish aggregates only.

Cleanest routes to footage, best first:
1. Record your own games (`scrcpy --record match.mp4`, or the phone's
   screen recorder with "Show taps" on, which a later labeller can read), and ask
   the TE players to record theirs (plan section 5): exact HUD, exact device,
   no rights question.
2. Ask creators from the manifest for **raw** recordings. Unedited sessions
   without overlays are what the labeller wants anyway.
3. Publish only derived artefacts (labels, statistics, fitted settings), with
   the unofficial-content notice.

---

## 7. Open items to verify by hand

Fan Content Policy and ToS full text; developer-portal terms and real rate
limits; BrawlAPI auth; Hugging Face dataset licence; Pyla model-zoo licence;
identity of the unidentified uploaders (the 100k Showdown-only pusher first);
video durations and overlays (the manifest fills durations); gas timing and
the points-to-tiles conversion.
