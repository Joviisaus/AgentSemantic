
"""
VLM prompt templates:
- Three-way classification (full match / partial match / no match)
- When partial, estimate seed count (for spectral clustering splits)

VLM responses must be strict JSON for reliable parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Literal, Optional, Sequence, Tuple

LABEL_FULL: Final = "full"
LABEL_PARTIAL: Final = "partial"
LABEL_NONE: Final = "none"

VALID_LABELS: Final = (LABEL_FULL, LABEL_PARTIAL, LABEL_NONE)

MediaType = Literal["mesh", "pointcloud"]

LABEL_FULL: Final = "full"
LABEL_PARTIAL: Final = "partial"
LABEL_NONE: Final = "none"

VALID_LABELS: Final = (LABEL_FULL, LABEL_PARTIAL, LABEL_NONE)

# ---------------------------------------------------------------------------
# Input modality guide + judgment overview (injected into stage prompts)
# ---------------------------------------------------------------------------

# {n_views} ???????= VLM ?????????????? _render_guide()?
RENDER_GUIDE_TEMPLATE: Final[Dict[str, str]] = {
    "mesh": (
        "[What the image shows]\n "
        "A {n_views}-view collage of the same mesh block. In each view:\n"
        "  - **Bright red/orange** = the **single block** under evaluation (triangle face coloring);\n "
        "  - **Light gray** = the rest of the model, for spatial context only, **not** part of this block.\n " "Red may be visible in only one or two views due to occlusion ? **check every view**."
    ),
    "pointcloud": (
        "[What the image shows]\n "
        "A {n_views}-view collage of the same point-cloud block. In each view:\n"
        "  - **Bright red/orange dots/spheres** = sampled points of the **single block** under evaluation;\n "
        "  - **Light gray points** = the rest of the model, for context only, **not** part of this block.\n " "Point clouds are sparser than meshes; red may appear as clusters or streaks ? "
        "**find all red points across views** before concluding.\n "
        "Point-cloud risk: small held/worn props (weapons, tools) may appear as a tiny red cluster "
        "in an arm bend ? **do not write a vague label like \"arm\" and miss the held object**."
    ),
}

# ?? 12 ???VLM ???????? / mesh ????????
DEFAULT_N_VIEWS: Final[int] = 12


def _render_guide(media_type: str, n_views: int = DEFAULT_N_VIEWS) -> str:
    """???????? RENDER_GUIDE??????????????"?? 12 ?"?????"""
    tmpl = RENDER_GUIDE_TEMPLATE.get(media_type, RENDER_GUIDE_TEMPLATE["mesh"])
    return tmpl.format(n_views=max(1, int(n_views)))


# ???????? RENDER_GUIDE[...] ?????? 12 ????
RENDER_GUIDE: Final[Dict[str, str]] = {
    k: v.format(n_views=DEFAULT_N_VIEWS) for k, v in RENDER_GUIDE_TEMPLATE.items()
}

JUDGMENT_OVERVIEW_TEMPLATE: Final = """[How the VLM assigns full / partial / none]
This system uses **two stages** to reduce bias from seeing the user semantic too early:

  Stage 1 ? **Pure observation (with image, no semantic)**
    ? Report only which body parts/regions the red highlight **actually covers** ? output covered_parts

  Stage 2 ? **Pure logic (no image, text only)**
    ? Compare covered_parts to the user target ?{semantic}? via **set comparison** ? output label

Label meanings for the pipeline:
  - **full**    : Red coverage **lies entirely within** ?{semantic}? (including sub-parts, e.g. ears under "head") ? no further split
  - **partial** : Red spills into **independent major parts outside** the semantic (e.g. head block includes chest) ? trigger geometric re-split
  - **none**    : Red is **completely unrelated** to ?{semantic}? ? discard block

**Forbidden**: skip observation and guess label; **forbidden** to peek at / infer user semantic during observation.
"""

FORCE_TERMINATE_NOTE: Final = """
[Forced termination note]
This block **cannot be split further** (too few points or max split depth reached).
You must still assign label per rules; `committed_if_forced` means:
"If the pipeline must stop now, does this block finally count as the semantic?" ? **only full or none**.
Choose full only if you are very sure the red bulk is ?{semantic}?; any non-semantic parts mixed in ? none.
"""

MEMORY_BLOCK_TEMPLATE: Final = """
[Memory of earlier verdicts on this SAME model & target ?{semantic}?]
The blocks below are other regions of the **same model** already judged in previous iterations
(parents that were split, and sibling/cousin leaves). Use them **only for consistency**, not to override what you see:
  - Do **not** re-claim as semantic a region that another leaf already took as **full** ? avoid double-counting.
  - If parents/siblings establish that the target concentrates in one area, treat red far from that area as **extra/none**.
  - Stay consistent with settled verdicts unless the current observation gives a clear reason to differ (note it in `reason`).
Earlier verdicts (most recent last):
{memory_entries}
"""

PROP_SEMANTIC_NOTE_TEMPLATE: Final = """
[Semantic type: independent prop / held object ?{semantic}?]
The observer **may have listed only limb names** (arm, torso) and missed the prop embedded in them ? common with point clouds.
If covered or dominant_region suggests the prop may be in a grip / shoulder-carry region, **do not** assign none directly;
assign **partial** so geometry can peel the prop off the limb.
"""

SEMANTIC_COVERAGE_DISCIPLINE = """Semantic coverage discipline (**mandatory** ? avoids losing parts after split and expensive VLM re-selection)
-----------------------------------
matched_parts / covered_parts on this block list semantics that **must still be claimable by child blocks** after splitting.
After geometric split into seed_count children, the **union** of child matched sets must cover **every item** above.

**Typical failure (forbidden)**: parent matched=[head, body, arm], after split union is only [body, arm] ? **head lost**.

When choosing seed_count / strategy:
1. **One semantic per piece**: N items in matched_parts ? seed_count must be **? N** (when seed_count is fixed, strategy must yield **? seed_count geometrically distinct** sub-regions).
2. **Protect small parts**: head, hand, finger, ear, horn, ornament ? small area, easily merged ? **do not** default to **cvt** or uniform **geodesic_fps** on multi-semantic composite blocks.
   Prefer **graph_cut** (grooves/seams, high lambda_concave), **skeleton** (limb branching),
   **spectral** (distant/joint separation, high concavity_weight).
   When **concavities are obvious** (armpit, groin, finger gaps, neck base, armor seams, skirt folds):
   prefer graph_cut or spectral; spectral default concavity_weight=0.70 is already high;
   raise further to **0.75~0.85** (or graph_cut lambda_concave?0.9).
3. **Strategy cautions**:
   - **cvt**: only when matched items are similar and you need equal area; **not** for head+torso+limbs composites.
   - **sdf**: coarse-to-fine by thickness; **cannot** guarantee one block per semantic item.
   - **geodesic_fps**: fallback for ambiguous composites with small parts, not first choice.
4. **Reason self-check**: briefly state which child each matched_part should land in; if a small part may merge into a neighbor, **change strategy** and note in reason.
5. Goal: later multi-label on children can **check off each item** without coverage repair re-selection."""

MULTI_LABEL_COVERAGE_DISCIPLINE = """Multi-label coverage discipline (avoid losing parts after split)
-----------------------------------
This block may come from a parent geometric split. When checking matched_parts:
1. **Label small parts too**: head, hand, finger, ear ? small area, but if red highlight **actually covers** that geometry, **must** include in matched_parts; do not skip as "too small".
2. **Only this block's geometry**: check an item only if bright red/orange highlight **actually includes** that part; do not copy sibling labels.
3. **Union awareness**: sibling blocks from the same parent should union to cover all parent semantics; if this block has geometry for a missing semantic, **must claim it**.
4. Missing labels trigger expensive VLM coverage re-selection ? prefer conservatively labeling all visible independent components on this block."""

MULTI_LABEL_PARENT_INHERITANCE_DISCIPLINE = """Parent semantic inheritance discipline (**mandatory**)
-----------------------------------
This block was split from a parent. Parent's prior semantic list is **parent_matched** above.

1. **Closed to parent**: matched_parts **should only be chosen from parent_matched**.
   Do **not** label semantics the parent **did not** list.
2. **High-confidence exception**: only if you are **very sure** (?85%) across views that red covers a part not in parent_matched,
   you may add it, with confidences **?0.85** for that item.
3. **When uncertain**: pick 1~3 items from parent_matched that best fit this block; do not introduce distant semantics the parent did not claim.
4. dominant must be in matched_parts and prefer the largest share among parent_matched."""

MULTI_LABEL_MAX_DEPTH_DISCIPLINE = """Max split depth discipline (**at limit ? final decision required**)
-----------------------------------
This block reached **maximum geometric split depth** (current depth={current_depth}, max_depth={max_depth}).
**No further geometric split** ? you must give a **single, definitive** final assignment.

1. **matched_parts must have exactly 1 item** ? pick the candidate that best fits the red highlight.
2. Even if red spans multiple regions (e.g. torso+thigh junction), choose only **dominant**
   (largest area / main geometry); **forbidden** to list 2+ items.
3. When unsure, pick the most geometrically relevant candidate; dominant must match matched_parts[0].
4. confidences for the chosen item **?0.80** ? this is the final call.
5. The system accepts this and stops splitting ? last round, be decisive."""

CLASSIFY_SYSTEM_PROMPT = (
    "You are a rigorous 3D model semantic analysis assistant. "
    "The user provides a multi-view collage: only **one** block is highlighted (bright red/orange);"
    "other regions are light gray for context only. "
    "You must **check every view**, **enumerate all parts covered by red**, then compare semantics, then assign label. " "Reply in strict JSON only ? no Markdown fences, no extra text."
)

# ---------------------------------------------------------------------------
# Two-stage classification: Stage 1 pure observation ? no semantic (avoid confirmation bias)
# ---------------------------------------------------------------------------

OBSERVE_SYSTEM_PROMPT = (
    "You are a strict, restrained 3D visual observer. "
    "In the multi-view collage, report **only** which object parts/regions are **actually** covered by **bright red/orange** highlight. " "No explanation, no guessing, no semantic inference; "
    "**light gray / beige / white / any non-bright-red-orange must not be listed**. " "**In practice, a single block's red usually covers only 1~3 adjacent parts** ? "
    "if you list more than 3, you are **likely inventing**; keep only parts you can **point to red in at least one view**. " "Strict JSON only ? no Markdown, no extra text."
)

OBSERVE_CONTEXT_HINT = """Object context (naming reference only ? **does not imply** red covers any listed part)
-----------------------------------
- This object is a **{category}**. {parts_line}
- When naming covered parts, **prefer** names from the candidate list; if red covers something outside the list, use a concise noun.
- Candidate list is a **vocabulary only** ? **do not** assume red covers a part just because it appears on the list.
"""

OBSERVE_ATTACHED_PROPS_BLOCK: Final = """
**Attached / held objects on limbs (easily missed ? inspect separately)**
-----------------------------------
If red covers arm, palm, or torso side, **do not stop at a vague limb name**:
Step A. Along the limb: palm, fingers, forearm end, shoulder/under-arm ? any **separate cluster** of finer red points?
Step B. If you see protruding geometry unlike the limb (barrel, rod, bottle, blade, bag, helmet?), **list it separately**,
     named by shape (e.g. "held elongated object", "rod on shoulder") ? **forbidden** to write only "right arm".
Step C. If limb and object points merge and boundaries are unclear, use compound items:
     "right arm and held object", "torso with shoulder-carried object" ? **do not omit the object**.
"""

OBSERVE_USER_TEMPLATE = """{render_guide}
{attached_props_block}

{judgment_overview}

==================================
Stage 1 task: pure observation (**no user semantic in this stage**)
==================================

**Check every view** in the {n_views}-panel image; list parts/regions **actually** covered by red highlight.

**How to observe (mandatory order)**
-----------------------------------
Step 1. Scan all {n_views} views left-to-right, top-to-bottom; note per view "is there bright red/orange here?"
Step 2. **Merge** all red locations across views into one part list (each part once).
Step 3. For each listed part, ask: "Can I point to red on **one specific** panel?" ? if not ? **remove** it.
Step 4. Write `dominant_region`: where red concentrates on the model (one sentence).

Observation discipline (violations are errors)
-----------------------------------
1. **Real blocks usually have red on only 1~3 adjacent parts**. If you list 4+, delete uncertain items first.
2. **Only bright red/orange**; light gray, white, beige **do not count** as red.
3. Red barely grazes an edge without clear coverage ? **does not count**.
4. Granularity guide (adapt to model type):
   - Humanoid/animal: head/face/neck/chest/back/waist/upper arm/forearm/hand/thigh/calf/foot ?
   - Props/mechanical: barrel/grip/trigger/magazine/wheel/wing/base ? (name shapes **you see**)
5. If **no view** has bright red ? `covered_parts` = [], explain in notes.

Output format (minimal)
-----------------------------------
{{
  "covered_parts": ["1~3 items; if >3, self-check and delete"],
  "dominant_region": "e.g. 'concentrated on mid-barrel', 'front of both legs', 'narrow ring at waist'",
  "notes": "optional observation details"
}}
"""

OBSERVE_USER_PROMPT = """This is a 12-view collage of the same 3D object.
In each view only **one** block is **bright red/orange**; all other geometry is light gray.
{context_block}
Check every view and **concisely** list parts **actually** covered by red.

Observation discipline (violations are errors)
-----------------------------------
1. **Real blocks usually cover only 1~3 adjacent parts**. E.g. "shell + top cover", "base + stand", "two adjacent segments of one rod".
   If you list 4+ parts, pause ? you **almost certainly** imagined some. **Remove uncertain items**.
2. **List only parts where you can point to red on a specific panel**.
   - Light gray, white, beige **are not** red.
   - Gray part you "think should be included" ? **cannot** list.
   - Red passes edge of a part without clear coverage ? does not count.
3. You do not know the user's task ? **do not** pre-guess which part it "should be",
   or add extra parts to seem thorough.
4. Granularity (**forbidden** left/right/front/back/up/down): use concise nouns
   (base / stand / panel / handle / wheel / blade / pipe / shell / top cover / leg / seat / backrest, etc.;
   prefer candidate list if provided). Symmetric pairs: one generic name (e.g. "leg" not "left leg").
5. If **no bright red anywhere**, `covered_parts` = [] and explain in notes.

Output format (minimal)
-----------------------------------
{
  "covered_parts": ["1~3 items; >3 is wrong ? delete first"],
  "dominant_region": "One sentence where red concentrates, e.g. 'base + stand', 'whole shell', 'narrow edge ring'",
  "notes": "optional details"
}
"""

REASON_SYSTEM_PROMPT = (
    "You are a semantic classifier. You will not see any image. "
    "The user provides: (1) covered_parts from the visual observer (parts covered by red);"
    "(2) the target semantic. Apply set-comparison rules and output JSON only."
)

REASON_USER_TEMPLATE_POINTCLOUD = """{judgment_overview}
{force_terminate_note}
{prop_semantic_note}
{memory_block}
==================================
Stage 2 task: pure logic (**no image**, text only)
==================================

Stage 1 observation:
  covered_parts = {covered_parts_json}
  dominant_region = {dominant_region_json}

Target semantic: ?{semantic}?

**Reasoning steps (mandatory order; write intermediate results to JSON fields)**
-----------------------------------
Step 0 ? Parse the target semantic into a boundary contract:
  Before comparing with covered_parts, extract:
  - `target_model`: what object/model category is implied (airplane, chair, guitar, etc.; "unknown" if absent).
  - `target_location`: where on that model the requested region is located (front/rear/top/underside/left/right,
    attachment area, around a joint, etc.; "unknown" if absent).
  - `included_semantics`: structures at that location that are INCLUDED in ?{semantic}?.
  - `excluded_neighbors`: nearby / attached / visually connected structures at the same location that are NOT included.

  **Location is not semantics**: if target says "engine at rear of airplane, excluding fuselage/tail",
  then red on the rear fuselage or tail is **extra**, not semantic, even though it is at the same location.
  Conversely, if target says "airplane body/fuselage", wings/engines/wheels attached to the body are usually extra.

  **Use explicit include/exclude words in the semantic if present**:
  - "including / include / belongs to / target" → included_semantics.
  - "excluding / exclude / not / without / nearby" → excluded_neighbors.
  These exclusions are hard constraints unless covered_parts clearly refer only to the included region.

Step 1 ? `semantic_parts`:
  Decide which parts on this model correspond to ?{semantic}? (usually 1~3 items).
  - Humanoid: semantic="hair" ? ["scalp hairstyle"]; semantic="belt" ? ["waist band"]
  - Prop: semantic="water gun" ? ["gun body/barrel"] or finer ["grip","barrel"] (match covered_parts granularity)
  - **Inclusive completion**: covered=["head overall"], semantic="hair" ? semantic_parts still lists hair (hair within head)
  - When covered is **unrelated** to semantic region (semantic="hair", covered=["calf"]) ? semantic_parts = []

Step 2 ? `extra_parts` (**only truly out-of-scope parts; sub-parts are NOT extra**):
  extra_parts = items in covered_parts **outside** semantic scope.

  **Semantic inclusion / sub-part rule (critical ? violations cause useless re-splits)**:
  - User semantics are often **coarse**; finer sub-parts that **belong to** the semantic must merge in, **not** go to extra.
  - semantic="head": ears, face, forehead, skull, chin, nose, eyes, mouth, cheeks, hair
    **are part of head** ? extra must be [] (even if observer listed "head" and "ears" separately)
  - semantic="face": eyes, nose, mouth, forehead, cheeks, chin **belong to face** ? not extra
  - semantic="hair": scalp/back hair belongs; **face, ears, neck do not** ? may be extra
  - semantic="water gun": barrel, grip, trigger, magazine **all belong to water gun** ? not extra to each other
  - semantic="belt": waist band is semantic; abdomen, back, legs **overflow** ? extra
  - semantic="airplane engine / engine nacelle": engine nacelle/intake/exhaust **belong**;
    fuselage/body/tail/wing/wheel **do not** unless explicitly included
  - semantic="airplane body / fuselage": fuselage/body **belong**;
    engine/wheel/wing/tail stabilizer **do not** unless explicitly included

  **Only** when covered has **independent major parts outside** semantic scope, fill extra and assign partial.
  Ex: semantic="head", covered=["head","ears"] ? extra=[] ? **full**
  Ex: semantic="head", covered=["head","neck","chest"] ? extra=["neck","chest"] ? partial
  Ex: semantic="hair", covered=["hair","face"] ? extra=["face"] ? partial
  Ex: semantic="rear airplane engine excluding fuselage/tail", covered=["rear fuselage","tail"] ?
      extra=["rear fuselage","tail"], missing_overlap=true or partial only if engine is also covered
  Ex: semantic="airplane body/fuselage", covered=["body","engine nacelle"] ?
      semantic_parts=["body/fuselage"], extra=["engine nacelle"], label="partial"

  When coarse part must split: covered=["head overall"], semantic="hair" ? extra as "head region excluding hair"

Step 3 ? `missing_overlap`:
  covered and semantic regions have **no intersection** ? true; else false

  **Held / worn / independent prop semantics** (gun, sword, rod, bottle, bag, hat, tool?):
  - Observer listed only "torso"/"arm" **without prop name** ? not necessarily unrelated!
  - Props often **embed in** arm/palm/shoulder red clusters; if covered includes **hand/arm/palm/finger/wrist**
    or dominant_region suggests "extended, gripping, shoulder-carry, object at end" ? **missing_overlap must be false**,
    semantic_parts write ?{semantic} (possibly embedded in limb coverage, pending split)?? label tends **partial**
  - Only if covered is **entirely** unrelated distal parts (semantic="water gun", covered=["left calf/tail"]) ? true
  Ex: semantic="water gun", covered=["torso","right arm"] ? **not none**; missing_overlap=false, partial
  Ex: semantic="water gun", covered=["held elongated object","right arm"] ? missing_overlap=false, partial or full
  Ex: semantic="water gun", covered=["left calf"] ? missing_overlap=true, none

Step 4 ? `semantic_area_ratio` (0~1, share of red that is semantic):
  Almost all semantic ? ?0.85; half-half ? ?0.5; mostly extra ? ?0.2
  **Be honest**: large face, small hair ? low ratio

Step 5 ? `dominant_side` (from ratio):
  ratio ? 0.7 ? "semantic"; ratio ? 0.3 ? "extra"; middle ? "balanced"

Step 6 ? `label` (whether to keep splitting):
  ?? covered empty ????????????????????????????? "none"
  ?? missing_overlap=true (truly unrelated)????? "none"
  ?? extra_parts non-empty (overflow or limb+prop mix) ? "partial"
  ?? extra empty and covered intersects semantic ?? "full"

  **Do not** assign partial because observer split one semantic into sub-part names (head+ears, barrel+grip) ?
  when all sub-parts lie in semantic scope, assign **full** to avoid useless re-split.

Step 7 ? `committed_if_forced` (**strict** ? final promise when forced stop, full/none only):
  dominant_side="semantic" and ratio ? 0.7 ? "full"
  anything else (extra / balanced / uncertain) ? "none"
  **Prefer miss over false accept**: extra-dominant blocks must not promise full

**Core principle: merge sub-parts + split only on overflow**
- Sub-parts within semantic (ears under head, grip under water gun) **are not** extra ? **full**
- Only when red clearly overflows to major parts outside semantic ? **partial** for geometric re-split
- committed_if_forced stays strict: ratio ?0.7, extra empty ? full

Output **only** this JSON (missing fields = error):
{{
  "target_model": "...",
  "target_location": "...",
  "included_semantics": ["..."],
  "excluded_neighbors": ["..."],
  "semantic_parts": ["..."],
  "extra_parts": ["..."],
  "missing_overlap": true | false,
  "semantic_area_ratio": 0.0~1.0,
  "dominant_side": "semantic" | "extra" | "balanced",
  "label": "full" | "partial" | "none",
  "committed_if_forced": "full" | "none",
  "confidence": 0.0~1.0,
  "reason": "One English sentence: what red covered, semantic share, why this label/committed_if_forced"
}}
"""

REASON_USER_TEMPLATE = """The visual observer listed parts covered by red highlight:
covered_parts = {covered_parts_json}
{context_block}
Target semantic: ?{semantic}?

Reason in order and fill JSON.

**Core rule: lenient label + strict commitment**
- Classification (label) **lenient**: if covered_parts **contains or may contain** the semantic region, allow partial
  so the pipeline can keep splitting until clean.
- Share / forced commitment (dominant_side / committed_if_forced) **strict**:
  if unsure semantic dominates, promise "none" ? avoid accepting a base-heavy block with a sliver of stand as "stand" at forced stop.

-----------------------------------
1. **Decide which parts correspond to ?{semantic}?** on this object (usually 1~3), fill `semantic_parts`.
   - Prefer **pure semantic parts**: semantic="wheel" ? "wheel / tire / caster";
     semantic="handle" ? "handle / grip".
   - **Inclusive completion**: if covered_parts only has a **coarse part containing** the semantic without separating it,
     still write semantic_parts (treat as implicit). E.g. semantic="wheel", covered=["chassis overall"]
     ? semantic_parts = ["wheel (within chassis overall)"], do not leave empty and assign none.
   - But: when covered_parts is **completely unrelated** (semantic="wheel", covered=["top cover","panel"]),
     semantic_parts must be empty.

2. **`extra_parts`**: covered_parts minus semantic_parts.
   - List neighboring non-semantic parts honestly (shell / base / stand / panel / top cover?).
   - For coarse parts ("chassis overall"), the **non-semantic portion** counts as extra ("chassis excluding wheels", etc.).

3. **`missing_overlap`**: are covered_parts **completely unrelated** to semantic region?
   - Completely unrelated ? true (? label=none)
   - Even "coarse part implicitly contains semantic" ? false

4. **Rough share of red that is semantic**, fill `semantic_area_ratio` (0.0~1.0 float).
   - Red almost all semantic (e.g. only hair cluster)          ? ? 0.85
   - Half semantic, half non-semantic                         ? ? 0.5
   - Red mostly extra (full face+neck, tiny hair on top)      ? ? 0.2
   - **Be honest on ratio** ? do not inflate to keep the block.

5. **`dominant_side`**:
   - semantic_area_ratio ? 0.7 ? "semantic"
   - semantic_area_ratio ? 0.3 ? "extra"
   - in between              ? "balanced"

6. **`label` (pipeline iteration, **lenient**)**:
   - covered_parts empty or missing_overlap=true                   ? "none"
   - extra_parts non-empty                                          ? "partial"
   - extra_parts empty and semantic_parts non-empty                 ? "full"

7. **`committed_if_forced` (forced final promise, **strict**)**:
   - dominant_side = "semantic" and semantic_area_ratio ? 0.7 ? "full"
   - any other case (extra / balanced / uncertain)              ? "none"
   - **Prefer dropping a balanced block over letting extra-dominant into final mask.**

Output **only** this JSON (missing fields = error):
{{
  "semantic_parts": ["..."],
  "extra_parts": ["..."],
  "missing_overlap": true | false,
  "semantic_area_ratio": 0.0~1.0,
  "dominant_side": "semantic" | "extra" | "balanced",
  "label": "full" | "partial" | "none",
  "committed_if_forced": "full" | "none",
  "confidence": 0.0~1.0,
  "reason": "One English sentence: coverage, semantic share, why label/committed_if_forced."
}}
"""

CLASSIFY_USER_TEMPLATE = """User-specified semantic region:
?{semantic}?

==================================
Task: classify the **red-highlighted block** vs ?{semantic}? as full / partial / none.
==================================

Important observation discipline (**mandatory**)
-----------------------------------
1. Do not conclude from one glance ? **check every panel**.
2. If red appears on different body regions in different views (e.g. crown + chest in one, waist in another),
   **all** count as part of the highlighted block ? **do not pick one**.
3. Highlight often spans multiple anatomical parts (hair + face + torso) ? usually **partial**.
4. You have confirmation bias: seeing red makes you say "yes, that's the semantic". **Counter it**:
   **list all parts red covers first**, then compare to ?{semantic}?.

Criteria
-----------------------------------
- "full": every part touched by red belongs to ?{semantic}?. May cover only a sub-region of the semantic,
  but **no geometry outside** the semantic.
  Ex: semantic="face", red only on "eyes" ? full.
- "partial": red intersects ?{semantic}? but **also clearly covers non-semantic parts**
  (adjacent or distant ? face/neck/chest/back/waist/abdomen/arm/forearm/hand/thigh/calf/foot, etc.).
  Ex: semantic="hair", red on "hair + face + chest band" ? partial.
  Ex: semantic="belt", red on "belt + shoes + wristband" ? partial.
- "none": red has **no intersection** with ?{semantic}?.
  Ex: semantic="face", red on "legs" ? none.

Output format (all fields required, in order)
-----------------------------------
{{
  "covered_parts": [
    "List every part covered by red, one item each;",
    "Granularity e.g. 'crown hair', 'full face', 'neck', 'chest vertical strip', 'upper arm', 'waist ring', 'both shoe tops' (no directional words).",
    "Do not duplicate the same region seen from another angle."
  ],
  "semantic_parts": [
    "List parts on this character corresponding to '{semantic}' (usually 1~3).",
    "Ex: semantic='hair' ? ['scalp hairstyle']; semantic='belt' ? ['waist narrow band']."
  ],
  "extra_parts": [
    "covered_parts minus semantic_parts ? red outside semantic.",
    "If empty, []; if non-empty, label must be partial or none."
  ],
  "missing_overlap": "true / false string: true if covered_parts and semantic_parts have no intersection",
  "label": "full | partial | none",
  "confidence": 0.0~1.0,
  "reason": "One English sentence: what red covered and why this label."
}}

Label decision table (**derive strictly, not by feel**)
-----------------------------------
- extra_parts empty and missing_overlap=false ? "full"
- extra_parts non-empty and missing_overlap=false ? "partial"
- missing_overlap=true                       ? "none"

Prefer conservative partial to trigger re-split over optimistic full.
"""

SEED_SYSTEM_PROMPT = (
    "You are a 3D model semantic analysis assistant. "
    "You receive a 12-view collage plus prior covered_parts / extra_parts from classification. " "The red block only **partially** matches the user semantic and must be split into sub-blocks. "
    "Your seed_count and strategy must ensure **no observed part is lost after split** (avoid expensive re-selection). " "Reply in strict JSON only."
)

SEED_USER_TEMPLATE = """User semantic region: ?{semantic}?

Structured info from prior classification (**trust this, do not re-guess from image**):
- covered_parts (all parts red covers) = {covered_parts_json}
- semantic_parts (subset belonging to ?{semantic}?) = {semantic_parts_json}
- extra_parts (red outside semantic) = {extra_parts_json}
- current block face count ? {block_faces}

Task: estimate **how many sub-blocks** to cut so **at least one sub-block covers only semantic_parts**
and others cover extra_parts.

Open-vocabulary discipline (**mandatory**)
-----------------------------------
- covered_parts / semantic_parts / extra_parts come from **prior open observation** ?
  may be any short names (prop parts, animal parts, mechanical components); **do not** force a human template.
- seed_count and strategy serve **splitting observed parts**, not guessing from priors.
- If N distinct parts were observed, seed_count usually ? N (between N and N+1, max {max_seeds}).
  Ex: covered_parts=[chair back slats, seat, leg] ? seed_count=3 or 4.
- When extra_parts non-empty, often need +1 cut to separate semantic from excess.

{semantic_coverage_discipline}

Also choose geometric partition strategy and params:
- Pick strategy from **open semantic description** and **block_faces**, not always geodesic_fps.
- graph_cut: parts separated by gaps/folds/grooves (preferred when concavity obvious, lambda_concave?0.8);
  spectral: distant parts stuck together (default first choice); when spectral, provide geodesic_weight /
  curvature_weight / concavity_weight (**default favors concavity concavity_weight=0.70; raise to 0.75~0.85 when grooves obvious**);
  skeleton: tree/limb skeleton, split along axis branches (large meshes auto-pool);
  cvt: equal fragments; sdf: thickness hierarchy in description (large meshes auto-reduce num_rays);
  geodesic_fps: fallback when semantics unclear, not default for large face count.
- Large block_faces: lower lloyd_iter / num_rays while keeping best semantic strategy.

Parameter discipline: defaults below; **when unsure keep defaults (params may be omitted entirely)**,
override only when render/semantics clearly require it, and explain in reason.

Return **only** this JSON:
{{
  "seed_count": <int, min 2, max {max_seeds}>,
  "strategy": "graph_cut" | "spectral" | "cvt" | "sdf" | "skeleton" | "geodesic_fps",
  "reason": "English brief: why this seed_count and strategy; map each matched/covered part to expected child",
  "params": {{
    "lambda_smooth": <float, graph_cut, 0.1~3.0, default 1.0>,
    "lambda_concave": <float, graph_cut, 0.0~2.0, obvious concavity ?0.8, default 0.85>,
    "geodesic_weight": <float, spectral, 0.05~0.80, default 0.20>,
    "curvature_weight": <float, spectral, 0.05~0.80, default 0.15>,
    "concavity_weight": <float, spectral, 0.05~0.90, default 0.70, obvious grooves 0.75~0.85>,
    "lloyd_iter": <int, cvt/geodesic_fps/skeleton, 2~12, default cvt=4 / others=0>,
    "contraction_iters": <int, skeleton, 2~12, default 5>,
    "graph_vertex_budget": <int, spectral/skeleton/geodesic_fps, 4000~20000, optional, default auto-pool>,
    "num_rays": <int, sdf, 4~24, default 8>,
    "cone_angle_deg": <float, sdf, 15~120, default 60>
  }}
}}
(Omit params entirely when unsure; include only fields for chosen strategy.)
"""

# ---------------------------------------------------------------------------
# Split strategy selection: graph_cut / spectral / CVT / SDF / skeleton / geodesic_fps
# ---------------------------------------------------------------------------

SPLIT_STRATEGY_SYSTEM_PROMPT = (
    "You are a 3D mesh partitioning algorithm expert. "
    "You receive semantic context, face count, matched_parts, seed_count, and optional renders;"
    "balance **full semantic coverage** (each item lands in its own child), **semantic fit**, and **runtime**. "
    "Choose one of six strategies and parameters. " "Strict JSON only ? no Markdown, no extra text."
)

SPLIT_STRATEGY_USER_TEMPLATE = """This block needs further geometric subdivision.

Semantic context
-----------------------------------
- Task semantic / matched parts: {semantic_context}
- covered_parts (red coverage) = {covered_parts_json}
- semantic_parts (belonging to semantic) = {semantic_parts_json}
- extra_parts (overflow parts) = {extra_parts_json}
- matched_parts (multi-label hits, full-coverage mode) = {matched_parts_json}
- Planned seed_count = {seed_count}
- Current block face count ? {block_faces}
- Subdivision stage: {split_stage}
- CLI strategy hint (reference only; you decide): {strategy_hint}

Performance and scale (system auto-optimizes; do not default to geodesic_fps for large face count)
-----------------------------------
- Geodesic strategies (spectral / skeleton / geodesic_fps / cvt) **pool vertex graphs** on large blocks
  (~8k~12k nodes); Dijkstra cost controlled; with fixed seed_count spectral **skips eigendecomposition**.
- block_faces < 30_000: prefer **spectral** (good distant-part separation).
- block_faces 30_000~150_000: pick spectral / skeleton / graph_cut / sdf by semantics; system lowers lloyd_iter / rays.
- block_faces ? 150_000: spectral / skeleton still OK; lloyd_iter 2~3 in params;
  **geodesic_fps only as fallback** when semantics unclear or prior strategies fail.
- graph_cut / sdf ray cost grows linearly with faces: reduce num_rays or lloyd_iter on huge blocks, not swap to non-semantic strategy.

Available strategies (strategy field must be one of)
-----------------------------------
1. **graph_cut**
   - Good when boundaries should follow grooves/valleys; adjacent semantics separated by gaps/folds
     (finger gaps, armor seams, skirt folds, armpit, neck base).
   - **First choice when concavity is obvious** ? boundary follows concave seams.
   - Params: lambda_smooth (boundary smoothness, 0.3~2.0, default 1.0),
     lambda_concave (concavity pull, 0.5~2.0, **default 0.85**; obvious concavity **0.9~1.2**).

2. **spectral**
   - Good for large pose/topology change, quasi-isometric splits (limb-torso joints, bent limbs, repeated symmetry).
   - Edge weight = weighted sum (auto-normalized):
     geodesic_weight (geodesic distance, separates distant parts),
     curvature_weight (dihedral bend, general fold lines),
     concavity_weight (concavity/grooves at joints/seams).
   - **Default favors concavity** (concavity_weight=0.70): best along grooves/joints;
     **obvious grooves/seams** (visible concave lines, finger gaps, groin) ? raise concavity_weight **0.75~0.85**;
     smooth bends, few grooves ? raise curvature_weight;
     distant limbs stuck together ? raise geodesic_weight.
   - Default: geodesic_weight=0.20, curvature_weight=0.15, concavity_weight=0.70.

3. **cvt** (centroidal Voronoi)
   - Equal-size compact roundish pieces (even finger splits, comparable fragments).
   - Params: lloyd_iter (2~12, default 4; auto-scaled by face count).

4. **sdf** (shape diameter function)
   - Organic models by thickness hierarchy (thick torso vs thin arm vs thinner fingers; quadruped leg vs torso).
   - Params: num_rays (4~24, default 8; auto-reduced on large blocks), cone_angle_deg (15~120, default 60).

5. **geodesic_fps** (geodesic FPS + Voronoi)
   - Fallback when semantics mixed or other strategies fail; also fast uniform splits.
   - Params: lloyd_iter (0~12, default 0; 2~4 on large for uniformity), graph_vertex_budget (4000~20000, optional).

6. **skeleton** (Laplacian contraction skeleton + geodesic Voronoi)
   - Tree/limb skeleton semantics (humanoid, quadruped, multi-limb mech) ? separate limbs from torso along axis/branches.
   - **Also splits one long limb along its axis into segments** (thigh/calf/foot, upper arm/forearm/hand, neck/head)
     even when transitions are smooth with no obvious groove ? cuts equidistant along axis, perpendicular to length,
     more stable than spectral/graph_cut when no concave seam exists.
   - **Criterion**: block is clearly elongated (major axis >> cross-section) and child semantics are longitudinal segments on same limb ? skeleton.
   - Params: contraction_iters (2~12, default 5), lloyd_iter (0~12, default 0),
     graph_vertex_budget (4000~20000, optional); large meshes auto-pool.

Open-vocabulary discipline
-----------------------------------
- covered_parts / semantic_parts / extra_parts / matched_parts are **open vocabulary** ? not necessarily humanoid;
  choose strategy from described geometry.
- Multiple matched_parts with very different names ? spectral or graph_cut on small/medium blocks;
  **obvious grooves/seams in image ? prefer graph_cut (high lambda_concave) or spectral (concavity_weight 0.75~0.85)**.
- Description emphasizes thickness hierarchy ? sdf on smaller blocks.
- Description emphasizes skeleton/limbs/torso-limb chain ? skeleton (large meshes OK, auto-pool).
- **Longitudinal limb segments without obvious groove ? must use skeleton**: e.g. separate thigh from leg, calf/foot segments;
  even if image shows no concave line, do not fall back to spectral/graph_cut random cuts; skeleton cuts equidistant along axis.
- **Semantics over scale**: pick best semantic strategy on large blocks; system pools/reduces iterations.
- geodesic_fps is fallback, not default: only when semantics unclear or groove/thickness/skeleton strategies unsuitable.

{semantic_coverage_discipline}

Parameter discipline
-----------------------------------
- Each param shows **default and range**; defaults are good starting points.
- **When unsure, keep defaults**: omit params entirely or only override fields you need; system fills defaults.
- Override only when render/semantics clearly require (e.g. deep concavity ? raise concavity_weight / lambda_concave),
  and explain in reason; do not change params without basis.

Return **only** this JSON:
{{
  "strategy": "graph_cut" | "spectral" | "cvt" | "sdf" | "skeleton" | "geodesic_fps",
  "reason": "English brief: why this strategy; map each matched_parts item to expected child (avoid swallowing small parts); note any param overrides",
  "params": {{
    "lambda_smooth": <float, graph_cut, 0.1~3.0, default 1.0>,
    "lambda_concave": <float, graph_cut, 0.0~2.0, default 0.85, obvious concavity 0.9~1.2>,
    "geodesic_weight": <float, spectral, 0.05~0.80, default 0.20>,
    "curvature_weight": <float, spectral, 0.05~0.80, default 0.15>,
    "concavity_weight": <float, spectral, 0.05~0.90, default 0.70, obvious grooves 0.75~0.85>,
    "lloyd_iter": <int, cvt/geodesic_fps/skeleton, 0~12, default cvt=4 / geodesic_fps=0 / skeleton=0>,
    "contraction_iters": <int, skeleton, 2~12, default 5>,
    "graph_vertex_budget": <int, spectral/skeleton/geodesic_fps, 4000~20000, optional, auto-pool>,
    "num_rays": <int, sdf, 4~24, default 8>,
    "cone_angle_deg": <float, sdf, 15~120, default 60>
  }}
}}
(Omit params fields you do not override; system uses defaults.)
"""

SPLIT_STRATEGY_SYSTEM_PROMPT_POINTCLOUD = (
    "You are a 3D point-cloud partitioning algorithm expert. "
    "You receive semantic context, point count, matched_parts, seed_count, and optional renders;"
    "balance **full semantic coverage**, **semantic fit**, and **runtime**. "
    "Choose one of six point-cloud strategies and parameters. " "Geometry comes from kNN graph, normals, curvature, and concavity ? no mesh faces/triangles. "
    "Strict JSON only ? no Markdown, no extra text."
)

SPLIT_STRATEGY_USER_TEMPLATE_POINTCLOUD = """This point-cloud block needs further geometric subdivision.

Semantic context
------------------------------------
- Task semantic / matched parts: {semantic_context}
- covered_parts (red coverage) = {covered_parts_json}
- semantic_parts (belonging to semantic) = {semantic_parts_json}
- extra_parts (overflow parts) = {extra_parts_json}
- matched_parts (multi-label hits, full-coverage mode) = {matched_parts_json}
- Planned seed_count = {seed_count}
- Current block point count ? {block_faces}
- Subdivision stage: {split_stage}
- CLI strategy hint (reference only; you decide): {strategy_hint}

Point-cloud traits and performance (system auto-optimizes; do not default to geodesic_fps for large point count)
------------------------------------
- All geometry on kNN graph: edge weights from geodesic distance, normal angle (curvature), and concavity ? no triangles.
- Geodesic strategies auto **downsample/pool point graphs** on large blocks (~8k~12k nodes);
  with fixed seed_count spectral **skips eigendecomposition**.
- block_points < 30_000: prefer **spectral**.
- block_points 30_000~150_000: spectral / skeleton / graph_cut / sdf by semantics; system lowers iterations/rays.
- block_points ? 150_000: spectral / skeleton OK; small iteration counts (2~3);
  **geodesic_fps fallback only**.

Available point-cloud strategies (strategy field must be one of)
------------------------------------
1. **graph_cut** (kNN graph cut)
   - Boundaries along grooves/valleys; adjacent semantics separated by gaps (finger gaps, armor seams, armpit, neck base).
   - **First choice when concavity obvious**.
   - Params: lambda_smooth (0.1~3.0, default 1.0), lambda_concave (0.0~2.0, **default 0.85**; obvious **0.9~1.2**).

2. **spectral** (point-cloud spectral)
   - Large pose/topology change, quasi-isometric splits (limb-torso, bent limbs, symmetry).
   - kNN edge weights (auto-normalized):
     curvature_weight (normal angle/curvature, fold lines),
     concavity_weight (grooves at joints/seams),
     normal_weight (normal consistency ? higher keeps smooth surface patches together).
   - **Obvious grooves/seams** ? raise concavity_weight (e.g. 0.5~1.0);
     smooth bends ? raise curvature_weight; noisy normals ? moderate normal_weight.
   - Default: curvature_weight?1.0, concavity_weight=0.0, normal_weight=1.0 (deviate as needed).

3. **cvt** ? equal compact pieces; lloyd_iter (2~12, default 4).

4. **sdf** (point-cloud SDF) ? thickness hierarchy; num_rays (4~24, default 8), cone_angle_deg (15~120, default 60).

5. **geodesic_fps** ? fallback / fast uniform; lloyd_iter (0~12, default 0).

6. **skeleton** ? tree/limb along axis/branches; longitudinal limb segments even without grooves;
   contraction_iters (2~12, default 5), lloyd_iter (0~12, default 0).

Open-vocabulary discipline
------------------------------------
- Inputs are open vocabulary ? choose from geometry described in text.
- Diverse matched_parts ? spectral or graph_cut; obvious grooves ? graph_cut (high lambda_concave) or spectral (raise concavity_weight).
- Thickness hierarchy ? sdf; skeleton/limb chain ? skeleton.
- Longitudinal limb segments without groove ? **must use skeleton**.
- Semantics over scale; geodesic_fps fallback only.

{semantic_coverage_discipline}

Parameter discipline
------------------------------------
- Defaults are good starting points; omit params or override only what you need.
- Explain overrides in reason.

Return **only** this JSON:
{{
  "strategy": "graph_cut" | "spectral" | "cvt" | "sdf" | "skeleton" | "geodesic_fps",
  "reason": "English brief: why this strategy; map matched_parts to expected children; note param overrides",
  "params": {{
    "lambda_smooth": <float, graph_cut, 0.1~3.0, default 1.0>,
    "lambda_concave": <float, graph_cut, 0.0~2.0, default 0.85, obvious concavity 0.9~1.2>,
    "curvature_weight": <float, spectral, 0.0~2.0, default 1.0>,
    "concavity_weight": <float, spectral, 0.0~2.0, default 0.0, obvious grooves 0.5~1.0>,
    "normal_weight": <float, spectral, 0.0~2.0, default 1.0>,
    "lloyd_iter": <int, cvt/geodesic_fps/skeleton, 0~12, default cvt=4 / geodesic_fps=0 / skeleton=0>,
    "contraction_iters": <int, skeleton, 2~12, default 5>,
    "num_rays": <int, sdf, 4~24, default 8>,
    "cone_angle_deg": <float, sdf, 15~120, default 60>
  }}
}}
(Omit params fields you do not override.)
"""


@dataclass
class ClassifyPrompt:
    system: str
    user: str


@dataclass
class ObjectContext:
    """Optional object grounding for generic (non-humanoid) part segmentation.

    ``category`` is the object class (e.g. "airplane"); ``candidate_parts`` is
    the open-vocab list of part names that may exist on this shape. Both are
    used only as a *naming* hint and never imply the highlighted block covers
    any particular part.
    """

    category: str = ""
    candidate_parts: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.category.strip() or self.candidate_parts)

    def observe_block(self) -> str:
        if self.is_empty():
            return ""
        if self.candidate_parts:
            parts = ", ".join(str(p).strip() for p in self.candidate_parts if str(p).strip())
            parts_line = f"It is typically composed of these parts (candidate list): {parts}."
        else:
            parts_line = ""
        return "\n" + OBSERVE_CONTEXT_HINT.format(
            category=self.category.strip() or "unknown object", parts_line=parts_line
        )

    def reason_block(self) -> str:
        if self.is_empty():
            return ""
        bits = []
        if self.category.strip():
            bits.append(f"Object category: {self.category.strip()}")
        if self.candidate_parts:
            parts = ", ".join(str(p).strip() for p in self.candidate_parts if str(p).strip())
            bits.append(f"Candidate part list for this object: {parts}")
        return "\n(Object context: " + "; ".join(bits) + ")"


def _media_defaults(media_type: str) -> Dict[str, Any]:
    if media_type == "pointcloud":
        return {
            "size_unit": "points",
            "large_threshold": 3000,
        }
    return {
        "size_unit": "faces",
        "large_threshold": 3000,
    }


def build_classify_prompt(
    semantic: str, *, media_type: MediaType = "mesh", n_views: int = DEFAULT_N_VIEWS
) -> ClassifyPrompt:
    guide = _render_guide(media_type, n_views)
    overview = JUDGMENT_OVERVIEW_TEMPLATE.format(semantic=semantic.strip())
    return ClassifyPrompt(
        system=CLASSIFY_SYSTEM_PROMPT,
        user=CLASSIFY_USER_TEMPLATE.format(
            semantic=semantic.strip(),
            render_guide=guide,
            judgment_overview=overview,
        ),
    )


def _looks_like_portable_prop_semantic(semantic: str) -> bool:
    """Whether user semantic looks like a held/worn/independent prop (not a body part)."""
    s = semantic.strip()
    if not s:
        return False
    prop_hints = (
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
    )
    body_hints = (
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
        "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?",
    )
    if any(h in s for h in prop_hints):
        # e.g. "machine gunner" ends with ? but core is gun ? still treat as prop
        if s.endswith("?") and len(s) <= 4:
            return False
        return True
    # coined/English names: conservative ? no body-part chars ? do not inject
    return not any(h in s for h in body_hints)




def build_classify_prompt(semantic: str, context: Optional["ObjectContext"] = None) -> ClassifyPrompt:
    return ClassifyPrompt(
        system=CLASSIFY_SYSTEM_PROMPT,
        user=CLASSIFY_USER_TEMPLATE.format(semantic=semantic.strip()),
    )


def build_observe_prompt(
    context: Optional["ObjectContext"] = None,
    *,
    media_type: MediaType = "mesh",
    n_views: int = DEFAULT_N_VIEWS,
) -> ClassifyPrompt:
    """Stage 1: pure observation ? no semantic (avoid confirmation bias)."""
    if media_type == "pointcloud":
        sem_placeholder = "(Disclosed only in stage 2 ? do not guess in this stage)"
        guide = _render_guide(media_type, n_views)
        overview = JUDGMENT_OVERVIEW_TEMPLATE.format(semantic=sem_placeholder)
        return ClassifyPrompt(
            system=OBSERVE_SYSTEM_PROMPT,
            user=OBSERVE_USER_TEMPLATE.format(
                render_guide=guide,
                attached_props_block=OBSERVE_ATTACHED_PROPS_BLOCK,
                judgment_overview=overview,
                n_views=max(1, int(n_views)),
            ),
        )
    context_block = context.observe_block() if context is not None else ""
    return ClassifyPrompt(
        system=OBSERVE_SYSTEM_PROMPT,
        user=OBSERVE_USER_PROMPT.replace("{context_block}", context_block),
    )


def build_reason_prompt(
    semantic: str,
    covered_parts: list[str],
    context: Optional["ObjectContext"] = None,
    *,
    media_type: MediaType = "mesh",
    force_terminate: bool = False,
    dominant_region: str = "",
    memory_block: str = "",
) -> ClassifyPrompt:
    """Stage 2: pure-text logic comparing stage-1 covered_parts to semantic.

    `memory_block`: optional pre-formatted text digest of earlier-iteration verdicts on
    the same model; injected so the VLM keeps cross-iteration memory for consistency.
    """
    import json as _json

    sem = semantic.strip()
    mem_note = ""
    if memory_block and memory_block.strip():
        mem_note = MEMORY_BLOCK_TEMPLATE.format(
            semantic=sem, memory_entries=memory_block.strip()
        )
    if media_type == "pointcloud":
        force_note = (
            FORCE_TERMINATE_NOTE.format(semantic=sem)
            if force_terminate
            else ""
        )
        prop_note = (
            PROP_SEMANTIC_NOTE_TEMPLATE.format(semantic=sem)
            if _looks_like_portable_prop_semantic(sem)
            else ""
        )
        dom = (dominant_region or "").strip() or "(not provided)"
        return ClassifyPrompt(
            system=REASON_SYSTEM_PROMPT,
            user=REASON_USER_TEMPLATE_POINTCLOUD.format(
                judgment_overview=JUDGMENT_OVERVIEW_TEMPLATE.format(semantic=sem),
                force_terminate_note=force_note,
                prop_semantic_note=prop_note,
                memory_block=mem_note,
                semantic=sem,
                covered_parts_json=_json.dumps(covered_parts, ensure_ascii=False),
                dominant_region_json=_json.dumps(dom, ensure_ascii=False),
            ),
        )

    context_block = context.reason_block() if context is not None else ""
    return ClassifyPrompt(
        system=REASON_SYSTEM_PROMPT,
        user=REASON_USER_TEMPLATE.format(
            semantic=sem,
            covered_parts_json=_json.dumps(covered_parts, ensure_ascii=False),
            context_block=context_block,
        ),
    )


# ---------------------------------------------------------------------------
# Revive pick: when the whole model ended with ZERO full blocks, force the VLM
# to choose the single most-likely leaf block and revive it as full (best effort).
# ---------------------------------------------------------------------------

REVIVE_SYSTEM_PROMPT = (
    "You are the final arbiter for 3D semantic segmentation. "
    "The whole model was searched block by block and **no** block was accepted as the target ?{semantic}?. "
    "We refuse to return an empty result, so you must pick the **single** candidate leaf block that is "
    "**most likely** to be (or to contain the bulk of) ?{semantic}?, even if imperfect. "
    "Be decisive ? you must choose exactly one. Strict JSON only, no Markdown."
)

REVIVE_USER_TEMPLATE = """Target semantic: ?{semantic}?

Situation
-----------------------------------
Every block was judged `none` during iteration ? the final mask would be **empty**. To avoid returning nothing,
relax the earlier strict rules and pick the one block that best represents ?{semantic}? (closest match wins).

Candidate leaf blocks (each may also be attached as a multi-view highlight image; the block id is printed in the
image title). Earlier observations are listed below as the authoritative legend:
{candidates_block}

How to choose
-----------------------------------
1. Prefer a block whose covered parts / dominant region **overlap the target's location**, even partially.
2. Among overlapping ones, prefer the one with the **largest share** of the target and least unrelated overflow.
3. If none clearly overlaps, pick the **least-unrelated** one (closest in location/shape to the target).
4. You **must** return one of the listed block_ids exactly.

Return **only** this JSON:
{{
  "chosen_block_id": "<one of the listed block ids, copied exactly>",
  "confidence": 0.0~1.0,
  "reason": "One English sentence: why this block is the closest match to the target."
}}
"""


def build_revive_prompt(
    semantic: str,
    candidates: Sequence[Dict[str, Any]],
) -> ClassifyPrompt:
    """Prompt to pick the single most-likely leaf block to revive as full.

    `candidates`: list of dicts with keys block_id, points, covered_parts,
    dominant_region, extra_parts, semantic_area_ratio, prior_reason.
    """
    import json as _json

    sem = semantic.strip()
    lines: List[str] = []
    for c in candidates:
        bid = str(c.get("block_id"))
        pts = c.get("points")
        covered = c.get("covered_parts") or []
        dom = str(c.get("dominant_region") or "").strip()
        extra = c.get("extra_parts") or []
        ratio = c.get("semantic_area_ratio")
        reason = str(c.get("prior_reason") or "").strip()
        if len(reason) > 200:
            reason = reason[:200] + "?"
        bits = [f"block_id={bid}"]
        if pts is not None:
            bits.append(f"points={pts}")
        bits.append(f"covered={_json.dumps(covered, ensure_ascii=False)}")
        if dom:
            bits.append(f"dominant_region={_json.dumps(dom, ensure_ascii=False)}")
        if extra:
            bits.append(f"extra={_json.dumps(extra, ensure_ascii=False)}")
        if ratio is not None:
            bits.append(f"semantic_area_ratio={ratio}")
        if reason:
            bits.append(f"prior_reason={reason}")
        lines.append("- " + "; ".join(bits))
    candidates_block = "\n".join(lines) if lines else "(none)"
    return ClassifyPrompt(
        system=REVIVE_SYSTEM_PROMPT.format(semantic=sem),
        user=REVIVE_USER_TEMPLATE.format(
            semantic=sem,
            candidates_block=candidates_block,
        ),
    )



# ---------------------------------------------------------------------------
# Global self-review: after each iteration VLM views whole-model progress image
# ---------------------------------------------------------------------------

REVIEW_SYSTEM_PROMPT = (
    "You are the lead supervisor for 3D model semantic analysis. "
    "Across iterations the model was split into blocks labeled full (in semantic) / partial (split more) / none (discard). " "The image shows **current global state**: all faces colored by latest judgment ? "
    "**bright red = confirmed ?{semantic}?**; **bright yellow = partial, pending split**; " "**light gray = confirmed not semantic**; **medium gray = abandoned / depth cap**. "
    "Assess whether **red coverage matches ?{semantic}?** overall and **whether to keep iterating** (split yellow) or stop. " "Strict JSON only ? no Markdown, no extra text."
)

REVIEW_USER_TEMPLATE = """Target semantic: ?{semantic}?

Iteration context (**factual data ? hard constraints**)
-----------------------------------
- Completed {iter_done} iterations, hard cap {max_depth}
- Current red face count: {full_faces} / {total_faces} ({full_ratio:.1%})  ? total red you should see
- Pending yellow face count: {pending_faces} / {total_faces} ({pending_ratio:.1%})
- Abandoned medium-gray face count: {abandoned_faces}

**Iron rules ? prevent VLM from stopping on "feel"**
-----------------------------------
1. **Describe honestly**: in `red_covered_parts` list **only** parts where you **actually see red on a panel**.
   - If red ratio < 3% (e.g. full_ratio={full_ratio:.1%}), almost no red visible ?
     `red_assessment` **cannot** be "matches", must be "incomplete".
   - Do not invent "red on crown hair" without seeing red ? must be visible on image.
2. **Do not terminate while much yellow remains**:
   - If pending_ratio ? 10% and semantic not basically covered, `should_terminate` **must** be false;
   - When pending_ratio < 5% and red looks in semantic region ? may allow should_terminate=true.
3. **Before stopping ask**: if we stop, final mask is current **red**. Would ?{semantic}? be recognizable at a glance?
   - If not ? do not terminate.

Think in order and fill JSON
-----------------------------------
1. **Observe red coverage**: focus on red pixels. Where on the character? ? `red_covered_parts`.
   - Red ratio {full_ratio:.1%}: if <3%, likely [] or "scattered specks, no region".
2. **Red quality**:
   - matches    : red mostly on ?{semantic}? with reasonable share;
   - incomplete : red clearly insufficient (typical: full_ratio below reasonable range);
   - over_claimed: red includes non-semantic parts;
   - partial    : both missing and extra.
3. **Yellow blocks**: still worth splitting?
   - Yellow may still contain ?{semantic}? (e.g. hair/hat inside head) ? keep_splitting;
   - Yellow surely non-semantic (hands/feet) ? likely_not_semantic;
   - Yellow mostly edge fragments ? mostly_done.
4. **should_terminate**:
   - Combine above + pending_ratio + iron rules; if full_ratio tiny and pending high, **never** terminate.

Return **only** this JSON:
{{
  "red_assessment": "matches | over_claimed | incomplete | partial",
  "red_covered_parts": ["parts red currently covers ? honest; [] if no red visible"],
  "missing_regions": "if red is missing expected regions, describe; else empty string",
  "extra_regions": "if red over-claims, describe non-semantic red regions; else empty string",
  "yellow_assessment": "should_keep_splitting | likely_not_semantic | mostly_done",
  "should_terminate": true | false,
  "confidence": 0.0~1.0,
  "reason": "One English sentence: from full_ratio={full_ratio:.1%} / pending_ratio={pending_ratio:.1%} and actual red coverage, why should_terminate."
}}
"""


def build_review_prompt(
    semantic: str,
    *,
    iter_done: int,
    max_depth: int,
    full_faces: int,
    pending_faces: int,
    abandoned_faces: int,
    total_faces: int,
) -> ClassifyPrompt:
    """Global progress self-review prompt."""
    full_ratio = (full_faces / total_faces) if total_faces else 0.0
    pending_ratio = (pending_faces / total_faces) if total_faces else 0.0
    return ClassifyPrompt(
        system=REVIEW_SYSTEM_PROMPT.format(semantic=semantic.strip()),
        user=REVIEW_USER_TEMPLATE.format(
            semantic=semantic.strip(),
            iter_done=int(iter_done),
            max_depth=int(max_depth),
            full_faces=int(full_faces),
            pending_faces=int(pending_faces),
            abandoned_faces=int(abandoned_faces),
            total_faces=int(total_faces),
            full_ratio=full_ratio,
            pending_ratio=pending_ratio,
        ),
    )


# ---------------------------------------------------------------------------
# One-shot grouping: over-segmented N candidates ? K groups with labels
# ---------------------------------------------------------------------------

GROUP_SYSTEM_PROMPT = (
    "You are a 3D model semantic analysis assistant. "
    "The user **over-segmented** the current partial block into N small candidate regions;"
    "each candidate uses a **distinct bright color** in the 12-view collage, "
    "with a **rounded-rect numeric label** (1, 2, 3, ...) at its centroid. " "Other faces are **light gray** for context ? **do not** assign them to any group. "
    "Merge candidate IDs into K groups (2?K?candidate count);"
    "each group is either in ?{semantic}?, not, or still uncertain. "
    "Strict JSON only ? no Markdown, no extra text."
)

GROUP_USER_TEMPLATE = """Target semantic: ?{semantic}?

Candidate list ({n_candidates} total; each has unique color + rounded numeric label in 12-view collage):
{candidate_table}

Task
-----------------------------------
Merge the {n_candidates} candidate IDs into 2~{max_groups} groups; label each group:
  - **full** : all candidates in group cover **only** pure geometry of ?{semantic}?, **no** face/neck/torso neighbors.
  - **none** : all candidates unrelated to ?{semantic}? (e.g. limbs, torso, feet).
  - **partial**: at least one candidate still mixes semantic and non-semantic ? needs more split.

Full label **strict threshold** (violation ? use partial)
-----------------------------------
- semantic=?hair?: full group candidates must cover **only** crown hair / hair mass / hair body,
  **not**: full face, forehead, neck, chin, ears, non-hair head regions.
- semantic=?belt?: full group must cover **only** waist band,
  **not**: abdomen, back, hips, chest.
- Rule: **if any candidate shows non-semantic geometry, group is partial not full**.
- **Prefer putting borderline candidates in partial for another pipeline pass over optimistic full**.
  Full groups enter final mask without re-check ? errors persist.

Merge rules
-----------------------------------
1. **Judge each candidate alone**: pure semantic / pure non-semantic / mixed?
   - pure semantic ? yes; pure non-semantic ? no; mixed/uncertain ? mixed
2. **Group**:
   - all yes ? full group (topology may be scattered)
   - all no  ? none group
   - all mixed ? one or more partial groups (split by mesh adjacency for later split)
3. **Coverage rule**: every ID 1..{n_candidates} assigned to **exactly one** group.
4. K usually 2~{max_groups}.

Return JSON per strict threshold:
{{
  "candidates_assessment": [
    {{"id": 1, "verdict": "yes|no|mixed", "note": "One sentence: what this candidate actually covers"}},
    {{"id": 2, "verdict": "...", "note": "..."}},
    ... (all {n_candidates} candidates)
  ],
  "groups": [
    {{
      "group_id": "A",
      "candidate_ids": [<int>, ...],
      "label": "full" | "partial" | "none",
      "confidence": 0.0~1.0,
      "description": "English: which parts these candidates cover"
    }},
    ...
  ],
  "reason": "One English sentence: how groups were formed ? especially why full groups have no non-semantic bleed."
}}
"""


def build_group_prompt(
    semantic: str,
    *,
    n_candidates: int,
    candidate_table: str,
    max_groups: int = 4,
) -> ClassifyPrompt:
    """One-shot grouping prompt. `candidate_table` is VLM-friendly list,
    e.g. '- 1=red (228,26,28)\\n- 2=blue (55,126,184)\\n...'."""
    return ClassifyPrompt(
        system=GROUP_SYSTEM_PROMPT.format(semantic=semantic.strip()),
        user=GROUP_USER_TEMPLATE.format(
            semantic=semantic.strip(),
            n_candidates=int(n_candidates),
            max_groups=max(2, int(max_groups)),
            candidate_table=candidate_table,
        ),
    )


# ---------------------------------------------------------------------------
# Full-coverage mode
# ---------------------------------------------------------------------------

ENUMERATE_SYSTEM_PROMPT = (
    "You are a 3D model coarse semantic partitioner. "
    "View 12-panel whole-model collage (no highlight); output **short, full-coverage, non-overlapping** part list;"
    "merge symmetric repeats (write \"arm\" not \"left arm/right arm\"). "
    "Same-class appendages use **one generic name** (e.g. arthropod \"appendage\" not \"leg\" + \"pedipalp\" separately). "
    "Names 2~6 chars; no finger/pattern-level detail. " "Must give complete auditable **topology_constraints** (count, attach_to). "
    "No directional words (left/right/front/back/up/down) in names. JSON only."
)

ENUMERATE_USER_TEMPLATE = """12-view whole-model collage (muted, no highlight).

Task: coarse **full-coverage** partition ? items together cover whole subject, non-overlapping, non-redundant.
Initial draft, may refine later; **short accurate names** (e.g. head, torso, arm), not long descriptions.

Naming and granularity
-----------------------------------
- Each item 2~6 chars; no parentheses; no \"includes/and/+\" merging multiple items in one name.
- **Forbidden** directional words for symmetric instances: use \"arm\", \"forearm\", \"thigh\", \"calf\", \"limb\", etc.
- Humanoid typical 6~8: head, neck, torso, upper arm, forearm, thigh, calf (hand/foot may merge into forearm/calf).
- Quadruped: head, neck, torso, tail, limb (or two symmetric limb groups, count=2 each).
- **Arthropods** (shrimp/crab/spider/scorpion/myriapod): typical 3 ? cephalothorax, abdomen, **appendage**.
  All external appendages (legs, pedipalps, claws, antennae?) **must merge as \"appendage\"**;
  forbid listing \"leg\", \"pedipalp\", \"claw\" separately.
  appendage count = total external appendage instances (crab 10, spider 8); attach_to = [\"cephalothorax\"] or [\"abdomen\"] per skeleton.
  Bad: [\"cephalothorax\",\"abdomen\",\"leg\",\"pedipalp\"]; Good: [\"cephalothorax\",\"abdomen\",\"appendage\"].
- Simple props/mechanical: 2~4 items (seat, backrest, leg ?).
- Do not split fingers/toes/hair strands; do not merge whole leg with torso.
{hint_block}

Constraints
-----------------------------------
1. Full coverage: all subject geometry maps to some item; hard cases merge into nearest major part.
2. Non-overlapping: no synonyms (torso + upper body); no splitting same appendage class into near-duplicates.
3. Count 2~12 items; **no** background / other / unclassified catch-alls.
4. **Strong constraint**: each final item has ?1 face; do not list parts absent or uncuttable on model.

Topology constraints (topology_constraints) ? **core, must be consistent**
-----------------------------------
For each semantic_parts item give **expected connected component count** and **skeleton attachment** (audited later).
Infer topology from **semantic skeleton**, not view direction.

**count (number of disconnected geometric pieces)**
- Symmetric repeats: count = **instance count** on full model.
  - Humanoid upper arm, forearm, thigh, calf each count=2.
  - If list has \"limbs\" equivalent ? count=4.
  - Quadruped two symmetric limb groups count=2 each; single \"limb\" item ? count=4.
- Central/unique parts (torso, head, neck, tail, seat?): usually count=1.

**attach_to (must-adjacent parent parts, exact semantic_parts strings)**
- Body/root (torso, seat?): attach_to = [].
- **All limb classes must attach_to body**: upper arm/forearm/thigh/calf/limb/appendage/tail ? attach_to = [\"torso\"] (or body name; arthropod appendage ? cephalothorax or abdomen).
- **Chain skeleton**: child attaches to direct parent, e.g. forearm?upper arm?torso; calf?thigh?torso; backrest/legs?seat.
- **Consistency rules**:
  - If upper arm+forearm+thigh+calf listed, all four chains root to torso; total 4 limb instances adjacent to torso.
  - attach_to names must exist in semantic_parts; no dangling references.
  - If count>0, each component should share mesh boundary with at least one attach_to parent.

- Names must **exactly match** semantic_parts; one topology_constraints row per item.

Return JSON only:
{{
  "object_kind": "?12 chars model type (e.g. humanoid character, quadruped, chair)",
  "semantic_parts": ["short1", "short2", "..."],
  "topology_constraints": [
    {{"part": "torso", "count": 1, "attach_to": []}},
    {{"part": "upper arm", "count": 2, "attach_to": ["torso"]}},
    {{"part": "forearm", "count": 2, "attach_to": ["upper arm"]}},
    {{"part": "thigh", "count": 2, "attach_to": ["torso"]}},
    {{"part": "calf", "count": 2, "attach_to": ["thigh"]}},
    {{"part": "head", "count": 1, "attach_to": ["torso"]}}
  ],
  "reason": "?20 chars optional: partition and topology rationale"
}}
"""

MULTI_LABEL_SYSTEM_PROMPT = (
    "You are a 3D block multi-label classifier. "
    "12-view collage: one block bright red/orange, rest light gray. " "Candidates are **open semantic short names** from enumeration (humanoid, animal, prop, mechanical, etc.). "
    "Select only candidates **actually covered** by red; **at least 1**, no background. " "If parent_matched given, child **should only pick from parent_matched**; "
    "parts not on parent list only if very sure (confidences?0.85). " "At max_depth, **matched_parts must be exactly 1** ? final definitive assignment. "
    "Small parts (head, hand, finger?) must be labeled if red covers them ? missing labels trigger expensive re-selection. " "JSON only."
)

MULTI_LABEL_USER_TEMPLATE = """From the candidate set, label which parts the red-highlighted block covers.
{parent_context_block}
Candidates (closed set ? exact spelling only):
{candidate_table}

Rules
-----------------------------------
1. Only bright red/orange counts; gray/white/beige is not red.
2. **matched_parts at least 1**; if unclear pick largest or most relevant candidate ? never empty.
3. Usually 1~3 adjacent candidates; if ?4, delete uncertain items first.
4. Geometry not listed ? assign to **mesh-adjacent nearest** candidate; do not invent new words.
5. If only \"appendage\" in list, label \"appendage\" even if it looks like leg or claw ? no out-of-list names.

{multi_label_coverage_discipline}
{parent_inheritance_block}
{max_depth_block}

Return JSON only:
{{
  "matched_parts": ["exact candidate text", "..."],
  "dominant": "largest share among matched (required, must be in matched_parts)",
  "confidences": {{"exact candidate": 0.0~1.0, "...": "per matched item, optional but recommended"}}
}}

Note: {n_candidates} candidates; each matched_parts entry must match candidate text exactly.
"""

SPLIT_COVERAGE_REPAIR_SYSTEM_PROMPT = (
    "You are a 3D block multi-label classifier. "
    "After parent geometric split, union of child labels must cover all parent semantics. " "If parent semantics missing from sibling union, re-judge **this block's** red coverage from the image. "
    "Select only from candidate set; at least 1. JSON only."
)

SPLIT_COVERAGE_REPAIR_USER_TEMPLATE = """Parent block ?{parent_id}? labels before split: {parent_matched}

Sibling block labels (reference only; judge **this block** geometry from image):
{siblings_table}

**Missing from union ? must be claimed by some child**: {missing_parts}
(Each item should land in exactly one child; add to matched_parts only if **this** red highlight actually covers it)

Highlighted block on this image: **{block_id}**

Candidates (closed set ? exact spelling):
{candidate_table}

Rules
-----------------------------------
1. Judge **only this block** (bright red/orange); do not copy sibling labels.
2. matched_parts at least 1; if this block contains missing items, **must** include them.
3. Do not check missing items this block does not contain.
4. Usually 1~3 adjacent candidates; if ?4, delete uncertain items.
5. Small parts (head, hand, finger?) must be included if this block's red covers them.

Return JSON only:
{{
  "matched_parts": ["exact candidate text", "..."],
  "dominant": "largest among matched (required, in matched_parts)"
}}
"""

EMPTY_SEMANTIC_REPAIR_SYSTEM_PROMPT = (
    "You are a 3D full-coverage semantic auditor. "
    "Every enumeration item must occupy at least 1 mesh face ? zero-face semantics forbidden. " "For zero-face items choose exactly one: "
    "split=carve from existing block; delete=remove from list. "
    "JSON only."
)

EMPTY_SEMANTIC_REPAIR_USER_TEMPLATE = """Current segmentation (12-view semantic color map) in attached image.

Semantic list and face counts (**0 = empty, must fix**):
{counts_table}

Zero-face semantics (one repair each):
{empty_list}

Donor blocks for split (block_id from table below; must be large enough to split):
{donor_table}

Task
-----------------------------------
For each zero-face semantic give **one** repair:
- **split**: carve geometry from a donor block.
  Must set `donor_block_id` (from table); prefer block most likely containing that part.
- **delete**: part absent or covered by other semantics ? remove from list.
  Omit `donor_block_id`.

Return JSON only:
{{
  "repairs": [
    {{
      "empty_semantic": "must match zero-face list exactly",
      "action": "split | delete",
      "donor_block_id": "required for split; omit or empty for delete",
      "reason": "?30 chars English"
    }}
  ]
}}

Requirement: repairs cover all zero-face semantics, exactly one row each; none missing.
"""

# ---------------------------------------------------------------------------
# Topology constraint inference (when user supplied semantic_parts, skip enumerate)
# ---------------------------------------------------------------------------

TOPOLOGY_INFER_SYSTEM_PROMPT = (
    "You are a 3D model skeleton topology analyst. "
    "From whole-model render and given semantic part list, infer connected component counts and skeleton attachments. " "When symmetric parts merged in list, count = total instances. "
    "Infer attach_to chains from semantic skeleton; all limbs must connect to body. " "No directional words in output. JSON only."
)

TOPOLOGY_INFER_USER_TEMPLATE = """12-view whole-model collage (muted, no highlight).

Given semantic part list (closed set ? do not add/rename):
{parts_table}

Task: topology_constraints for each item (from **semantic skeleton**, not view direction).

**count**
- Symmetric repeats: **instance count** on full model (e.g. upper arm=2, limb=4).
- Unique/central parts (torso, head, neck, tail, seat?): usually =1.
- If list has upper arm+forearm+thigh+calf, total limb instances should be 4.

**attach_to** (must-adjacent parent, list exact names; body/root = [])
- All limb classes attach_to body (torso/seat?).
- Chain: forearm?upper arm?torso, calf?thigh?torso, backrest?seat.
- **Rules**: each limb chain root adjacent to body; attach_to references only listed parts.

Return JSON only:
{{
  "topology_constraints": [
    {{"part": "exact list name", "count": 1, "attach_to": []}},
    ...
  ]
}}

Requirement: cover **every** list item; part names exact; count and attach_to consistent.
"""

# ---------------------------------------------------------------------------
# Topology violation repair (audit count / attach_to after segmentation)
# ---------------------------------------------------------------------------

TOPOLOGY_REPAIR_SYSTEM_PROMPT = (
    "You are a 3D full-coverage semantic topology auditor. "
    "From render, expected topology, and actual component stats, find violations and propose fixes. " "Too many blocks ? relabel; too few ? geometric_split donor with your seed_count and per-child semantics. "
    "Maintain skeleton: correct symmetric instance counts, limb chains rooted at body. " "Actions: reassign_component, geometric_split, cut_adjacency, cut_bridge, boundary_refine. "
    "No directional words. JSON only."
)

TOPOLOGY_REPAIR_USER_TEMPLATE = """Current segmentation (12-view semantic color map) in attached image.

Expected topology constraints:
{constraints_table}

Actual connected component counts:
{component_table}

Adjacency (parts sharing boundary):
{adjacency_table}

Detected topology violations (fix each):
{violations_table}

Donor blocks for carve/subdivide (block_id / current semantic / face count):
{donor_table}

Repair decisions (by violation type)
-----------------------------------
**count_high (too many / redundant)**:
  Which part are extra components really? Judge from image.
  ? Prefer **reassign_component**: relabel from from_semantic to to_semantic, component_index=N.
    May also fix a count_low elsewhere.

**count_low (too few)**:
  Missing blocks merged, mislabeled, or embedded?
  Case A ? same semantic merged (two arms in one block):
    ? **geometric_split**: child_assignments may repeat same semantic.
  Case B ? block mislabeled elsewhere:
    ? **reassign_component**.
  Case C ? multiple semantics in one block:
    ? **geometric_split**: you choose seed_count (2~{max_split}),
      child_assignments for each piece (labels 1..K in collage order); not all same semantic.
      Ex: torso block with two arms ? 3 pieces [arm, arm, torso].
  Case D ? limb instances short, should all attach to torso:
    ? geometric_split from donor; ensure attach_to chain correct.

**geometric_split notes**:
  - seed_count: pieces to cut (?2).
  - child_assignments: length = seed_count; index 1..K matches collage numeric labels;
    pre-assign by topology (symmetric instances may repeat semantic name); pipeline re-confirms visually.
  - donor_block_id: leaf block to cut.

**missing_attachment**:
  Part should attach_to parent but no shared boundary ? reassign or geometric_split to restore limb-root adjacency to body.

**illegal_adjacency (e.g. head?arm)**:
  ? **cut_adjacency**: from_semantic=A, to_semantic=B; pipeline uses spectral + concave geodesic cut.

**pseudo_bridge / handle_loop (e.g. legs fused)**:
  ? **cut_bridge**: from_semantic=A, to_semantic=B; pipeline cuts handle loop and relabels cores.

**boundary_refine (jagged post-surgery boundary)**:
  ? **boundary_refine**: adjacent pair to smooth via concavity-aware graph cut.

Forbidden: casual merge into neighbor semantic; no delete_part.

If image matches stats and no real violation, return empty repairs.

Return JSON only:
{{
  "topology_ok": true | false,
  "repairs": [
    {{
      "violation": "brief violation",
      "action": "reassign_component | geometric_split | cut_adjacency | cut_bridge | boundary_refine",
      "from_semantic": "wrong label for reassign",
      "to_semantic": "correct label for reassign",
      "component_index": 1,
      "donor_block_id": "required for geometric_split",
      "seed_count": 2,
      "child_assignments": [
        {{"index": 1, "semantic": "arm"}},
        {{"index": 2, "semantic": "arm"}}
      ],
      "reason": "?30 chars English"
    }}
  ]
}}
"""

GEOMETRIC_SPLIT_ASSIGN_SYSTEM_PROMPT = (
    "You are a 3D block semantic assigner. "
    "Donor block was geometrically split into numbered sub-blocks (1..K, colors + numeric labels); rest is light gray. " "Assign each number a semantic from the candidate set. JSON only."
)

GEOMETRIC_SPLIT_ASSIGN_USER_TEMPLATE = """Donor block {donor_block_id} current semantic: ?{donor_semantic}?
After geometric split: {child_count} pieces labeled 1..{child_count} on collage.

Candidate semantics (closed set ? exact spelling):
{candidate_table}

Topology repair pre-assignment (reference ? trust geometry you see):
{planned_table}

Task
-----------------------------------
For each index 1..{child_count} assign semantic (exact candidate text).
Different indices may share or differ in semantic.

Return JSON only:
{{
  "assignments": [
    {{"index": 1, "semantic": "exact candidate"}},
    ... must cover every index 1..{child_count}
  ]
}}
"""


def build_enumerate_prompt(*, hint: str = "") -> ClassifyPrompt:
    """Build full-coverage semantic enumeration prompt. `hint` injects extra constraints
    (e.g. \"quadruped cat, short animal skeleton names\")."""
    hint_block = (
        f"- User note: {hint.strip()}"
        if hint and hint.strip()
        else "- No extra note: pick appropriate template from image, short full-coverage names."
    )
    return ClassifyPrompt(
        system=ENUMERATE_SYSTEM_PROMPT,
        user=ENUMERATE_USER_TEMPLATE.format(hint_block=hint_block),
    )


def build_multi_label_prompt(
    candidate_parts: list[str],
    *,
    block_id: str = "",
    parent_id: str = "",
    parent_matched: Sequence[str] | None = None,
    at_max_depth: bool = False,
    current_depth: int = 0,
    max_depth: int = 0,
) -> ClassifyPrompt:
    """Build multi-label prompt ? which candidate parts the block hits."""
    parts = [str(x).strip() for x in candidate_parts if str(x).strip()]
    if not parts:
        raise ValueError("candidate_parts must not be empty")
    candidate_table = "\n".join(f"- {i+1}. {p}" for i, p in enumerate(parts))

    parent_list = [str(x).strip() for x in (parent_matched or []) if str(x).strip()]
    if parent_list:
        parent_context_block = (
            f"\nParent semantic constraint (labels before parent split)\n"
            f"-----------------------------------\n"
            f"Block **{block_id or '?'}** split from parent **{parent_id or '?'}**.\n"
            f"Parent parent_matched (child should pick only from these): {parent_list}\n"
        )
        parent_inheritance_block = MULTI_LABEL_PARENT_INHERITANCE_DISCIPLINE
    else:
        parent_context_block = ""
        parent_inheritance_block = ""

    if at_max_depth:
        max_depth_block = MULTI_LABEL_MAX_DEPTH_DISCIPLINE.format(
            current_depth=int(current_depth),
            max_depth=int(max_depth),
        )
    else:
        max_depth_block = ""

    return ClassifyPrompt(
        system=MULTI_LABEL_SYSTEM_PROMPT,
        user=MULTI_LABEL_USER_TEMPLATE.format(
            parent_context_block=parent_context_block,
            candidate_table=candidate_table,
            n_candidates=len(parts),
            multi_label_coverage_discipline=MULTI_LABEL_COVERAGE_DISCIPLINE,
            parent_inheritance_block=parent_inheritance_block,
            max_depth_block=max_depth_block,
        ),
    )


def build_split_coverage_repair_prompt(
    *,
    parent_id: str,
    block_id: str,
    parent_matched: Sequence[str],
    sibling_matched: Dict[str, Sequence[str]],
    missing_parts: Sequence[str],
    candidate_parts: Sequence[str],
) -> ClassifyPrompt:
    """After split, missing semantic coverage ? re-prompt VLM to re-check matched_parts."""
    parts = [str(x).strip() for x in candidate_parts if str(x).strip()]
    if not parts:
        raise ValueError("candidate_parts must not be empty")
    candidate_table = "\n".join(f"- {i+1}. {p}" for i, p in enumerate(parts))
    siblings_table = "\n".join(
        f"- {bid}: {list(matched)}"
        for bid, matched in sorted(sibling_matched.items())
    ) or "(none)"
    return ClassifyPrompt(
        system=SPLIT_COVERAGE_REPAIR_SYSTEM_PROMPT,
        user=SPLIT_COVERAGE_REPAIR_USER_TEMPLATE.format(
            parent_id=str(parent_id).strip(),
            block_id=str(block_id).strip(),
            parent_matched=list(parent_matched),
            siblings_table=siblings_table,
            missing_parts=list(missing_parts),
            candidate_table=candidate_table,
        ),
    )


def build_topology_infer_prompt(
    semantic_parts: Sequence[str],
) -> ClassifyPrompt:
    """When user already supplied semantic_parts, infer topology constraints."""
    parts = [str(x).strip() for x in semantic_parts if str(x).strip()]
    parts_table = "\n".join(f"- {i+1}. {p}" for i, p in enumerate(parts))
    return ClassifyPrompt(
        system=TOPOLOGY_INFER_SYSTEM_PROMPT,
        user=TOPOLOGY_INFER_USER_TEMPLATE.format(parts_table=parts_table),
    )


def build_topology_repair_prompt(
    *,
    topology_constraints: Sequence[Dict[str, Any]],
    component_counts: Dict[str, int],
    component_details: Sequence[Dict[str, Any]],
    adjacency_pairs: Sequence[Tuple[str, str]],
    violations: Sequence[Dict[str, Any]],
    donor_blocks: Sequence[Dict[str, Any]],
    max_split: int = 8,
) -> ClassifyPrompt:
    """Topology violation repair: VLM chooses reassign / geometric_split."""
    constraints_table = "\n".join(
        f"- {c.get('part', '?')}: count={c.get('count', '?')}, "
        f"attach_to={c.get('attach_to', [])}"
        for c in topology_constraints
    ) or "(no constraints)"
    component_table = "\n".join(
        f"- {p}: {int(component_counts.get(p, 0))} components"
        for p in sorted(component_counts.keys())
    )
    detail_lines: List[str] = []
    for d in component_details:
        detail_lines.append(
            f"  - {d.get('semantic', '?')} #{d.get('index', '?')}: "
            f"{int(d.get('faces', 0))} faces"
        )
    component_table += (
        "\nPer-component detail:\n" + "\n".join(detail_lines) if detail_lines else ""
    )
    adjacency_table = (
        "\n".join(f"- {a} ? {b}" for a, b in adjacency_pairs)
        if adjacency_pairs
        else "(no adjacency)"
    )
    violations_table = (
        "\n".join(
            f"- [{v.get('kind', '?')}] {v.get('part', '?')}: {v.get('detail', '')}"
            for v in violations
        )
        if violations
        else "(no auto-detected violations ? judge from attached image)"
    )
    if donor_blocks:
        donor_table = "\n".join(
            f"- {b['block_id']}: {b['semantic']}, {int(b['faces'])} faces"
            for b in donor_blocks
        )
    else:
        donor_table = "(no available donor blocks)"
    return ClassifyPrompt(
        system=TOPOLOGY_REPAIR_SYSTEM_PROMPT,
        user=TOPOLOGY_REPAIR_USER_TEMPLATE.format(
            constraints_table=constraints_table,
            component_table=component_table,
            adjacency_table=adjacency_table,
            violations_table=violations_table,
            donor_table=donor_table,
            max_split=max(2, int(max_split)),
        ),
    )


def build_geometric_split_assign_prompt(
    *,
    donor_block_id: str,
    donor_semantic: str,
    child_count: int,
    candidate_semantics: Sequence[str],
    planned_assignments: Sequence[Dict[str, Any]],
) -> ClassifyPrompt:
    """After geometric split: VLM assigns semantic to each numbered child."""
    candidates = [str(x).strip() for x in candidate_semantics if str(x).strip()]
    candidate_table = "\n".join(f"- {i+1}. {p}" for i, p in enumerate(candidates))
    if planned_assignments:
        planned_table = "\n".join(
            f"- #{int(a.get('index', 0))}: {a.get('semantic', '?')}"
            for a in planned_assignments
        )
    else:
        planned_table = "(no pre-assignment)"
    return ClassifyPrompt(
        system=GEOMETRIC_SPLIT_ASSIGN_SYSTEM_PROMPT,
        user=GEOMETRIC_SPLIT_ASSIGN_USER_TEMPLATE.format(
            donor_block_id=str(donor_block_id).strip(),
            donor_semantic=str(donor_semantic).strip() or "(unknown)",
            child_count=max(1, int(child_count)),
            candidate_table=candidate_table,
            planned_table=planned_table,
        ),
    )


def build_empty_semantic_repair_prompt(
    *,
    semantic_parts: Sequence[str],
    face_counts: Dict[str, int],
    empty_parts: Sequence[str],
    donor_blocks: Sequence[Dict[str, Any]],
) -> ClassifyPrompt:
    """Zero-face semantic repair: VLM chooses split (from donor) or delete."""
    counts_table = "\n".join(
        f"- {p}: {int(face_counts.get(p, 0))} faces"
        for p in semantic_parts
    )
    empty_list = "\n".join(f"- {p}" for p in empty_parts)
    if donor_blocks:
        donor_table = "\n".join(
            f"- {b['block_id']}: {b['semantic']}, {int(b['faces'])} faces"
            for b in donor_blocks
        )
    else:
        donor_table = "(no available donor blocks ? delete only)"
    return ClassifyPrompt(
        system=EMPTY_SEMANTIC_REPAIR_SYSTEM_PROMPT,
        user=EMPTY_SEMANTIC_REPAIR_USER_TEMPLATE.format(
            counts_table=counts_table,
            empty_list=empty_list,
            donor_table=donor_table,
        ),
    )


# ---------------------------------------------------------------------------
# Disconnected components of same part: e.g. one \"limb\" ? 4 blocks ? instance names
# ---------------------------------------------------------------------------

COMPONENT_NAME_SYSTEM_PROMPT = (
    "You are a 3D part naming assistant. "
    "One semantic part broke into multiple disconnected components;"
    "12-view collage colors each with numeric label (1..N), rest light gray. "
    "From **adjacency/skeleton role** vs body, decide per-block labels"
    "(e.g. limb-1 / limb-2, wing-1 / wing-2) or keep original name if fragments/indistinguishable. "
    "No directional words in labels. JSON only."
)

COMPONENT_NAME_USER_TEMPLATE = """Original semantic part: ?{part}?. Geometrically split into {n} disconnected components.
Collage labels them 1..{n} (unique color + centroid numeric label).

Per-component topology reference (body connection, relative scale, by numeric label):
{component_table}

Task
-----------------------------------
1. Inspect each numbered block in 12 views + adjacency table.
2. Assign each `label`:
   - If **symmetric/repeated same-class parts** (four limbs, wing pair, wheels?),
     use **instance suffix**: \"{part}-1\", \"{part}-2\"? or semantic subtype (no directional words).
   - If **fragments of one part** or instances indistinguishable ? label = original ?{part}?.
3. Labels short (?8 chars); no unjustified instance names ? when unsure keep original.
   **Different blocks** get different names only when truly distinct.

Return JSON only:
{{
  "split": true | false,
  "components": [
    {{"id": 1, "label": "limb-1", "reason": "independent component adjacent to torso"}},
    ... one row per index 1..{n}
  ]
}}
"""


def build_component_name_prompt(
    part: str,
    n: int,
    component_table: str,
) -> ClassifyPrompt:
    """Build prompt for naming disconnected components of the same semantic part."""
    return ClassifyPrompt(
        system=COMPONENT_NAME_SYSTEM_PROMPT,
        user=COMPONENT_NAME_USER_TEMPLATE.format(
            part=str(part).strip(),
            n=int(n),
            component_table=component_table,
        ),
    )


def build_split_strategy_prompt(
    *,
    semantic_context: str = "",
    seed_count: int = 2,
    covered_parts: list[str] | None = None,
    semantic_parts: list[str] | None = None,
    extra_parts: list[str] | None = None,
    matched_parts: list[str] | None = None,
    block_faces: int = 0,
    split_stage: str = "main",
    strategy_hint: str = "",
    media_type: MediaType = "mesh",
) -> ClassifyPrompt:
    """Build VLM split-strategy selection prompt."""
    import json as _json

    hint = str(strategy_hint or "").strip()
    if not hint or hint == "auto":
        hint_text = "none (you decide from semantics and block_faces)"
    else:
        hint_text = f"user prefers {hint}, but if it conflicts with block_faces scale, scale wins"

    if media_type == "pointcloud":
        _system = SPLIT_STRATEGY_SYSTEM_PROMPT_POINTCLOUD
        _user_template = SPLIT_STRATEGY_USER_TEMPLATE_POINTCLOUD
    else:
        _system = SPLIT_STRATEGY_SYSTEM_PROMPT
        _user_template = SPLIT_STRATEGY_USER_TEMPLATE

    return ClassifyPrompt(
        system=_system,
        user=_user_template.format(
            semantic_context=str(semantic_context or "").strip() or "(unspecified)",
            seed_count=max(2, int(seed_count)),
            covered_parts_json=_json.dumps(
                covered_parts or [], ensure_ascii=False
            ),
            semantic_parts_json=_json.dumps(
                semantic_parts or [], ensure_ascii=False
            ),
            extra_parts_json=_json.dumps(extra_parts or [], ensure_ascii=False),
            matched_parts_json=_json.dumps(
                matched_parts or [], ensure_ascii=False
            ),
            block_faces=int(block_faces),
            split_stage=str(split_stage or "main"),
            strategy_hint=hint_text,
            semantic_coverage_discipline=SEMANTIC_COVERAGE_DISCIPLINE,
        ),
    )


SEED_USER_TEMPLATE_POINTCLOUD = """{render_guide}

User semantic region: ?{semantic}?

Structured results from prior classification (**prior facts** ? trust, do not overturn from image):
- covered_parts (all red-covered parts) = {covered_parts_json}
- semantic_parts (subset belonging to ?{semantic}?) = {semantic_parts_json}
- extra_parts (red outside semantic) = {extra_parts_json}
- current block size ? {block_size} {size_unit}

==================================
Task: estimate seed_count (how many sub-blocks to cut)
==================================

**Goal**: after cut, **at least one sub-block covers only semantic_parts**, others on extra_parts.

**How to estimate seed_count (think in order)**
-----------------------------------
Step 1. Count **distinct parts** in covered_parts ? baseline = max(2, part_count + 1)
Step 2. If extra_parts includes large torso/head/whole limb ? at least 4 (hard to cut in one)
Step 3. Larger block (>{large_threshold} {size_unit}) ? prefer more pieces; complex ? 4~5
Step 4. Only when covered_parts is one small region ? seed_count=2 allowed
Step 5. Cap {max_seeds}

Ex: covered=[crown hair, full face, neck] ? 3 parts ? suggest 4 (margin)
Ex: covered=[barrel, grip], semantic=[barrel] ? suggest 3~4 to peel grip off

Return JSON only:
{{
  "seed_count": <int, usually 3 or 4, complex 5; min 2, max {max_seeds}>,
  "reason": "English: part count and why this seed_count"
}}
"""


def build_seed_prompt(
    semantic: str,
    *,
    max_seeds: int = 8,
    covered_parts: list[str] | None = None,
    semantic_parts: list[str] | None = None,
    extra_parts: list[str] | None = None,
    block_faces: int = 0,
    block_points: int | None = None,
    media_type: MediaType = "mesh",
    n_views: int = DEFAULT_N_VIEWS,
) -> ClassifyPrompt:
    """Build seed_count estimation prompt."""
    import json as _json

    if media_type == "pointcloud":
        defaults = _media_defaults(media_type)
        block_size = int(block_points if block_points is not None else block_faces)
        guide = _render_guide(media_type, n_views)
        return ClassifyPrompt(
            system=SEED_SYSTEM_PROMPT,
            user=SEED_USER_TEMPLATE_POINTCLOUD.format(
                render_guide=guide,
                semantic=semantic.strip(),
                max_seeds=max(2, int(max_seeds)),
                covered_parts_json=_json.dumps(covered_parts or [], ensure_ascii=False),
                semantic_parts_json=_json.dumps(semantic_parts or [], ensure_ascii=False),
                extra_parts_json=_json.dumps(extra_parts or [], ensure_ascii=False),
                block_size=block_size,
                size_unit=defaults["size_unit"],
                large_threshold=defaults["large_threshold"],
            ),
        )

    return ClassifyPrompt(
        system=SEED_SYSTEM_PROMPT,
        user=SEED_USER_TEMPLATE.format(
            semantic=semantic.strip(),
            max_seeds=max(2, int(max_seeds)),
            covered_parts_json=_json.dumps(covered_parts or [], ensure_ascii=False),
            semantic_parts_json=_json.dumps(semantic_parts or [], ensure_ascii=False),
            extra_parts_json=_json.dumps(extra_parts or [], ensure_ascii=False),
            block_faces=int(block_faces),
            semantic_coverage_discipline=SEMANTIC_COVERAGE_DISCIPLINE,
        ),
    )


# ---------------------------------------------------------------------------
# Semantic tree: VLM builds hierarchical partition tree from whole-model render
# ---------------------------------------------------------------------------

SEMANTIC_TREE_SYSTEM_PROMPT = (
    "You are a 3D hierarchical semantic partitioner. "
    "View 12-panel whole-model collage (no highlight); output a **coarse-to-fine semantic tree**:"
    "root is whole object; each level splits parent into non-overlapping children covering parent fully. "
    "Merge symmetric repeats at same level (\"arm\" not \"left arm/right arm\"). "
    "Names 2~6 chars; deeper levels may split fingers/ornament but not over-fragment. " "No directional words in names. JSON only."
)

SEMANTIC_TREE_USER_TEMPLATE = """12-view whole-model collage (muted, no highlight).

Task: **hierarchical semantic tree** ? from root \"whole\", each level splits parent into child parts.
Pipeline will geometrically split level by level; you only provide **reasonable hierarchy**.

Example (humanoid ? structure only, do not copy names blindly):
  whole ? upper body / lower body
  upper body ? head / torso / arm
  arm ? upper arm / forearm / hand
  hand ? palm / wrist / fingers
  lower body ? thigh / calf / foot

Naming and granularity
-----------------------------------
- Each item 2~6 chars; no parentheses; no \"includes/and/+\" in one name.
- **No** directional words.
- Level 1 (root children) coarse: 2~4 items (upper/lower body, or head/torso/limbs).
- Deeper levels refine; leaves may be fingers, shoe top, etc.
- Children must **fully cover** parent geometry, non-overlapping.
- Suggested tree depth 2~{max_depth} levels (including root); do not pad depth with meaningless splits.
{hint_block}

Constraints
-----------------------------------
1. Root name fixed \"whole\"; children non-empty.
2. Each non-leaf has 2~8 children.
3. Leaves have children = [].
4. Same semantic name must not appear on different branches.
5. No background / other / unclassified catch-alls.

Return JSON only:
{{
  "object_kind": "?12 chars model type (e.g. humanoid character, quadruped, chair)",
  "tree": {{
    "name": "whole",
    "children": [
      {{
        "name": "child1",
        "children": [
          {{"name": "finer", "children": []}}
        ]
      }}
    ]
  }},
  "reason": "?30 chars English: hierarchy rationale"
}}
"""


def build_semantic_tree_prompt(
    *,
    hint: str = "",
    max_depth: int = 5,
) -> ClassifyPrompt:
    """Build hierarchical semantic tree prompt."""
    hint_block = (
        f"- User note: {hint.strip()}"
        if hint and hint.strip()
        else "- No extra note: pick appropriate hierarchy template from image."
    )
    return ClassifyPrompt(
        system=SEMANTIC_TREE_SYSTEM_PROMPT,
        user=SEMANTIC_TREE_USER_TEMPLATE.format(
            hint_block=hint_block,
            max_depth=max(2, int(max_depth)),
        ),
    )


# ---------------------------------------------------------------------------
# Post-processing: geometric cleanup of the FINAL semantic mask
# ---------------------------------------------------------------------------

POSTPROCESS_SYSTEM_PROMPT = (
    "You are a 3D segmentation post-processing expert. The pipeline has produced a FINAL "
    "selection (a region labelled as the user semantic). You are given (a) geometric quality "
    "metrics of that selection and (b) a multi-view render with the selection highlighted in "
    "bright red. Decide which geometric cleanup operations (if any) improve the selection so it "
    "matches the semantic region more cleanly. Operate ONLY on geometry/topology of the existing "
    "selection: you may drop islands/outliers, smooth jagged boundaries, fill enclosed holes, or "
    "keep the largest connected piece. You must NOT invent new regions or change semantics. "
    "If the selection is already clean and faithful, return an empty operations list. "
    "Strict JSON only, no Markdown, no extra text."
)

POSTPROCESS_USER_TEMPLATE = """{render_guide}

User semantic region: ?{semantic}?
The bright-red points/faces are the pipeline's FINAL answer for this semantic. Judge whether its
geometry needs cleanup.

Geometric quality metrics of the current selection (computed on the {unit} graph)
------------------------------------
{metrics_json}

Metric meaning (read before deciding)
------------------------------------
- selected_fraction: selection size / whole model.
- num_components / largest_component_fraction: connectivity. Many components or a small
  largest fraction => the selection is fragmented (scattered islands).
- num_small_islands / island_point_fraction: count and mass of tiny disconnected blobs
  (likely spillover or noise far from the main region).
- boundary_ratio: fraction of selected elements on the boundary. High => thin / jagged / shredded.
- boundary_roughness: jaggedness of the boundary (variance of local selected-ratio). High => ragged edge.
- edge_smoothness: 0~1 composite, higher = smoother cleaner edge.
- hole_fraction: unselected elements fully enclosed by the selection (interior holes to fill).
- outlier_fraction: isolated selected elements far from their nearest selected neighbour.
- solidity: selection size / elements inside its convex hull. Low (<~0.5) => the convex hull
  bridges over OTHER parts (selection is hollow or wraps a different component). High (~1) = compact, solid.
- mean_curvature / boundary_mean_curvature: surface variation; a clean part boundary usually
  sits in a concave seam (boundary curvature notably higher than interior).

Available cleanup operations (operations[].name must be one of)
------------------------------------
1. "keep_largest_component" ? drop everything except the single largest connected piece.
   Use ONLY when small islands are clearly spurious AND the true region is one connected piece.
2. "remove_small_components" ? drop connected pieces smaller than min_fraction x largest.
   Use when there are a few scattered islands but the main mass is correct. params: {{ "min_fraction": 0.01~0.5 (default 0.1) }}
3. "remove_outliers" ? drop isolated far-away selected elements. params: {{ "std_ratio": 1.5~6 (default 3.0) }}
4. "smooth_boundary" ? morphological smoothing of a jagged/ragged edge.
   params: {{ "mode": "open"|"close"|"both" (open=trim spikes, close=fill nicks, both=overall smooth; default "both"),
             "iters": 1~3 (default 1) }}
5. "fill_holes" ? add interior unselected elements enclosed by the selection.
   params: {{ "iters": 1~4 (default 2), "threshold": 0.5~0.95 (default 0.7) }}

Decision discipline
------------------------------------
- Be conservative. Only choose an operation when a metric AND the render agree it helps.
- Prefer removing obvious noise (small islands / outliers) before aggressive boundary smoothing.
- Do NOT smooth_boundary just because boundary_ratio is high on a genuinely thin part (e.g. a strap, antenna).
- If metrics look clean (few components, high edge_smoothness, high solidity, low outlier/island/hole), return [].
- Order matters: list operations in the order they should run (e.g. remove islands -> remove outliers -> smooth).

Return ONLY this JSON:
{{
  "operations": [
    {{ "name": "<op name>", "params": {{ ... only overrides ... }} }}
  ],
  "reason": "English brief: which metrics + visual evidence drove each op, or why no cleanup needed"
}}
"""


def build_postprocess_prompt(
    semantic: str,
    *,
    metrics: Dict[str, Any],
    media_type: MediaType = "pointcloud",
    n_views: int = DEFAULT_N_VIEWS,
) -> ClassifyPrompt:
    """Build the final-mask geometric post-processing decision prompt."""
    import json as _json

    unit = "point" if media_type == "pointcloud" else "face"
    return ClassifyPrompt(
        system=POSTPROCESS_SYSTEM_PROMPT,
        user=POSTPROCESS_USER_TEMPLATE.format(
            render_guide=_render_guide(media_type, n_views),
            semantic=semantic.strip(),
            unit=unit,
            metrics_json=_json.dumps(metrics, ensure_ascii=False, indent=2),
        ),
    )


COMPONENT_SPLIT_SYSTEM_PROMPT = (
    "You are a 3D segmentation post-processing expert. The pipeline's FINAL selection for a user "
    "semantic has broken into several LARGE disconnected pieces (connected components). "
    "IMPORTANT: these components are already proven **geometrically separate** on the 3D "
    "point/face adjacency graph — they share NO continuous path. Your job is NOT to decide "
    "whether they are connected; they are not. Your job is to decide which disconnected "
    "pieces truly belong to the user semantic and which are spillover into a different part. "
    "Each large component is drawn in a distinct solid color in a multi-view collage of the "
    "SAME 3D object from different camera angles; non-selected points/faces are dim grey. "
    "Because each panel is a 2D projection, two physically separate 3D components can LOOK "
    "fused or touching in some views (especially side/oblique views where depth stacks) while "
    "being clearly apart in others (front/back/top). You MUST reason across ALL views and must "
    "NOT treat 'looks connected in one view' as proof they are the same part. "
    "Keep ALL components when they are merely disjoint pieces of the SAME semantic (e.g. left/right "
    "symmetric parts, a part split by a sampling gap). Keep only a SUBSET when one large piece "
    "clearly sits on a DIFFERENT adjacent part than the semantic. "
    "Do NOT invent regions or change semantics. Strict JSON only, no Markdown, no extra text."
)

COMPONENT_SPLIT_USER_TEMPLATE = """{render_guide}

[Color scheme for THIS task — overrides the red-highlight description above]
Each **large disconnected component** has its own solid color ({color_legend}).
These are NOT "one red block" — compare **color vs color** and **color vs grey context**
across views. Dim grey = unselected model context.

User semantic region: «{semantic}»
The pipeline's FINAL selection for this semantic split into {n_components} large connected components.
Each is shown in a distinct color: {color_legend}. All non-selected {unit}s are dim grey.

Per-component sizes (fraction = component {unit}s / whole selection)
------------------------------------
{components_json}

Other geometric metrics of the whole selection
------------------------------------
{metrics_json}

Critical: same 3D object, many 2D views (read before deciding)
------------------------------------
The image is a {n_views}-view collage of the **same** 3D model from different cameras
(elev/az labels on each panel). This is NOT {n_views} different objects.

**Geometric fact (already computed):** C0, C1, … are **topologically disconnected**
connected components — there is no continuous adjacency path between them in 3D.
Do NOT reverse this because one panel makes two colors look merged.

**Projection trap (very common):** In side / oblique views, two separate 3D parts often
**overlap in the image** and look like one blob (depth stacking). In front / back / top
views the **same two colors** are often clearly separated by background grey or a visible gap.
→ A side view that looks "connected" does **NOT** mean the pieces belong together.
→ If **any** view shows clear color separation AND one color sits on a different body part,
  that is strong evidence to DROP that component even when another view looks fused.

Multi-view decision procedure (**mandatory**)
------------------------------------
Step 1 — Scan **every** panel (all {n_views} views). For each component color, note:
  - In which views is it visible?
  - In which views is it **clearly separated** from the other colored component(s)?
  - In which views does it **only look** fused because of projection / depth overlap?

Step 2 — Pick the **best-separation view(s)** per component pair: views where two colors
  are most obviously distinct (largest gap, least overlap, different body regions visible).
  Prefer front/back/top over pure side views when they disagree.

Step 3 — In those best-separation views, for **each** component ask:
  "Does this colored region lie on «{semantic}» (or a sub-part of it), or on a clearly
   DIFFERENT adjacent part (spillover across a seam)?"

Step 4 — Aggregate across views:
  - DROP a component if in its clearest-separation views it consistently sits on a part
    **outside** «{semantic}», even if some oblique view makes it look attached to another color.
  - KEEP a component if in clearest-separation views it lies on «{semantic}», even if some
    side view makes it look merged with another color.
  - KEEP **both/all** when every large component, judged in clearest-separation views,
    is part of «{semantic}» (e.g. left+right symmetric pieces, two blobs of the same wing).

Typical cases
------------------------------------
- **DROP spillover:** C0 (red) = main «{semantic}» mass; C1 (blue) = neighbour part that
  touches in a side view but is clearly separate in front view → keep [0], drop C1.
- **KEEP symmetric / same part:** C0 + C1 = left and right pieces of «{semantic}»; separated
  in front view but both clearly on the semantic → keep [0, 1].
- **KEEP sampling gap:** two blobs of the same part with a thin missing bridge; both colors
  map to «{semantic}» in best views → keep both.

Decision discipline
------------------------------------
- Trust **geometric disconnection** + **best-separation views** over a single fused-looking panel.
- Use color identity consistently: C0=first/largest component, C1=second largest, etc.
- Fractions only tell you each piece is large (not a tiny island); they do not tell you semantics.
- When genuinely ambiguous in ALL views, prefer to KEEP (conservative; do not over-prune).
- In ``reason``, cite **which view(s)** showed separation vs fusion and why each component
  belongs to «{semantic}» or a neighbour.

Return ONLY this JSON:
{{
  "keep_components": [<component ids to KEEP, e.g. 0, 1>],
  "reason": "English brief: per component, keep/drop with view evidence (e.g. 'front view: C0 on wing, C1 on fuselage; side view overlap ignored')"
}}
If every large component belongs to the semantic, return all ids (no split).
"""


def build_component_split_prompt(
    semantic: str,
    *,
    components: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    media_type: MediaType = "pointcloud",
    n_views: int = DEFAULT_N_VIEWS,
) -> ClassifyPrompt:
    """Build the prompt asking the VLM whether a multi-large-component selection should be split.

    ``components`` is a list of ``{"id", "fraction", "size"}`` ordered by size (id 0 = largest);
    each id maps to a fixed render color via the color legend.
    """
    import json as _json

    unit = "point" if media_type == "pointcloud" else "face"
    n_components = len(components)
    color_names = ("red", "blue", "green", "orange", "purple", "cyan")
    color_legend = ", ".join(
        f"C{i}={color_names[i % len(color_names)]}" for i in range(n_components)
    )
    return ClassifyPrompt(
        system=COMPONENT_SPLIT_SYSTEM_PROMPT,
        user=COMPONENT_SPLIT_USER_TEMPLATE.format(
            render_guide=_render_guide(media_type, n_views),
            semantic=semantic.strip(),
            unit=unit,
            n_views=max(1, int(n_views)),
            n_components=n_components,
            color_legend=color_legend,
            components_json=_json.dumps(components, ensure_ascii=False, indent=2),
            metrics_json=_json.dumps(metrics, ensure_ascii=False, indent=2),
        ),
    )
