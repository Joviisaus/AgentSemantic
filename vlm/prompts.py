"""
VLM 提示模板：
- 三分类（完全符合 / 局部符合 / 完全不符合）
- 局部符合时估计种子数（用于谱聚类分裂）

约定 VLM 必须返回严格的 JSON，便于稳定解析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

LABEL_FULL: Final = "full"
LABEL_PARTIAL: Final = "partial"
LABEL_NONE: Final = "none"

VALID_LABELS: Final = (LABEL_FULL, LABEL_PARTIAL, LABEL_NONE)

# 分裂前 VLM 决策：避免切分后语义丢失、触发昂贵的覆盖重选
SEMANTIC_COVERAGE_DISCIPLINE = """语义覆盖纪律（**必须遵守**，避免分裂后丢部位、触发昂贵 VLM 重选）
-----------------------------------
当前块 matched_parts / covered_parts 列出的是**切分后必须仍能被子块认领**的语义清单。
几何切分产生 seed_count 个子块后，各子块 matched 的**并集**须覆盖上述**每一项**。

**典型失败（禁止）**：父块 matched=[头, 身体, 手臂]，切完子块并集只剩 [身体, 手臂] → **头消失**。

选 seed_count / strategy 时：
1. **一项一语义一块**：matched_parts 有 N 项 → seed_count 须 **≥ N**（系统给定 seed_count 时，策略须产出 **≥ seed_count 个几何可区分** 子区域）。
2. **保护小部位**：头、手、指、耳、角、装饰等占面小、最易被吞并——**禁止**对含多项语义的复合块首选 **cvt** 或「均匀」**geodesic_fps**。
   优先 **graph_cut**（凹槽/接缝，lambda_concave 取高）、**skeleton**（肢类分叉）、
   **spectral**（远距/关节分离，concavity_weight 取高）。
   **凹陷明显**（腋窝、腹股沟、指缝、颈根、装甲接缝、裙摆褶皱等）时：
   首选 graph_cut 或 spectral；spectral 默认 concavity_weight=0.70 已偏高，
   此时再上调到 **0.75~0.85**（或 graph_cut 的 lambda_concave≥0.9）。
3. **策略忌口**：
   - **cvt**：仅当 matched 各项语义相近、仅需均分面积；**不得**用于头+躯干+四肢等综合块。
   - **sdf**：按粗细分层，**不能保证**每项语义独占一块。
   - **geodesic_fps**：语义复合且含小部位时仅作兜底，非首选。
4. **reason 自检**：简述每项 matched_part 切分后预期落在哪个子块；若某小部位可能被并入邻块，**换 strategy** 并在 reason 写明。
5. 目标是让后续子块 multi-label 能**逐项勾选**，无需触发覆盖修复重选。"""

MULTI_LABEL_COVERAGE_DISCIPLINE = """语义覆盖标注纪律（避免分裂后丢部位）
-----------------------------------
本块可能由父块几何切分而来。勾选 matched_parts 时：
1. **小部位也要标**：头、手、指、耳等占面小，但若红色高亮区**确实覆盖**该部位几何，**必须**纳入 matched_parts，禁止因「太小」漏标。
2. **只标本块几何**：仅当鲜艳红/橙高亮**实际包含**该部位才勾选；勿照搬兄弟子块标注。
3. **并集意识**：同一父块分裂出的兄弟子块 matched 并集应覆盖父块全部语义；本块若有某缺失语义的几何主体，**必须认领**。
4. 漏标会导致系统触发昂贵的 VLM 覆盖重选——宁可保守多标本块真实可见部位，也不要漏掉独立几何组件。"""

MULTI_LABEL_PARENT_INHERITANCE_DISCIPLINE = """父块语义继承纪律（**必须遵守**）
-----------------------------------
本块由父块几何分裂而来。父块上一轮已标注的语义清单见上文 **parent_matched**。

1. **封闭于父块**：matched_parts **原则上只能从 parent_matched 中勾选**。
   父块**未标注**的语义，**不要**标到本块。
2. **高把握例外**：仅当你在多视角中**非常确定**（主观 ≥85%）红色高亮区确实覆盖父块未列出的部位时，
   才可将其加入 matched_parts，并在 confidences 中为该项填写 **≥0.85**。
3. **不确定时**：从 parent_matched 里选 1~3 项与本块几何最贴合的；勿引入父块未认领的远处语义。
4. dominant 须在 matched_parts 内，且优先选 parent_matched 中占比最大的一项。"""

MULTI_LABEL_MAX_DEPTH_DISCIPLINE = """最大切分深度纪律（**已达上限，必须给出最终定论**）
-----------------------------------
⚠️ 本块已达几何切分**最大深度**（当前 depth={current_depth}，上限 max_depth={max_depth}）。
**不可能再几何分裂**，你必须给出**唯一、确切**的最终归属。

1. **matched_parts 只能有 1 项**——从候选中选出红色高亮区**最贴合**的一项。
2. 即使红色看似跨多个部位（如躯干+大腿连接处），也必须只选 **dominant**
   （面积占比最大、几何主体所在）一项；**禁止**标 2 项及以上。
3. 犹豫时在候选中选几何最相关的一项；dominant 与 matched_parts[0] 必须一致。
4. confidences 为选定项填 **≥0.80**，表示这是最终定论。
5. 系统会直接接受此项并停止切分——这是最后一轮，请给出明确划分。"""

CLASSIFY_SYSTEM_PROMPT = (
    "你是一个严谨的 3D 模型语义分析助手。"
    "用户给你 12 个不同视角拼接的渲染图：图中只有**一个**分块被高亮（鲜艳红/橙色），"
    "其它区域是淡灰色，仅作为定位参考。"
    "你必须先**逐视角观察**、**枚举所有红色覆盖到的部位**，再进行语义比较，最后才给出 label。"
    "严格按 JSON 格式回答，禁止 Markdown 代码块、禁止任何额外文字。"
)


# ---------------------------------------------------------------------------
# 两阶段分类：第 1 阶段「纯观察」——不出现语义，避免确认偏差
# ---------------------------------------------------------------------------

OBSERVE_SYSTEM_PROMPT = (
    "你是一个严格、克制的 3D 模型视觉观察员。"
    "在 12 视角拼图里，**只**报告颜色为**鲜艳红/橙**的高亮区域**实际**覆盖到的物体部件/区域。"
    "不要解释、不要猜测、不要联想任何语义；"
    "**淡灰色 / 米色 / 白色 / 任何非鲜艳红橙色的部位都不能列入**。"
    "**真实情况下，单个分块的红色通常只覆盖 1~3 个相邻部件**——"
    "如果你列出了超过 3 个部件，你**很可能在编造**，请重新只保留你**百分百在某张图里直接看到红色**的那些。"
    "严格按 JSON 格式回答，禁止 Markdown，禁止额外文字。"
)

# 物体上下文块（可选）：物体类别 + 候选部件清单。由 build_observe_prompt /
# build_reason_prompt 在已知时注入；为空时整段省略，prompt 退化为通用开放词表。
OBSERVE_CONTEXT_HINT = """物体上下文（仅用于命名参考，**不暗示**红色一定覆盖其中任何一项）
-----------------------------------
- 该物体是一个 **{category}**。{parts_line}
- 描述红色覆盖部件时**优先**从上述候选名里选用；若红色覆盖到清单外的区域，可自行用简洁名词命名。
- ⚠️ 候选清单只是命名词表，**不要**因为某部件在清单里就臆断红色覆盖到它。
"""

OBSERVE_USER_PROMPT = """这是同一个 3D 物体的 12 个视角拼接渲染。
图中每个视角里只有**一个**分块被涂成**鲜艳红/橙色**，其他几何全部是淡灰色。
{context_block}
请逐视角看一遍，把红色**实际**覆盖到的物体部件**精简地**列出来。

观察纪律（违反即视为错误）
-----------------------------------
1. **真实分块的红色通常只在 1~3 个相邻部件**。例如「外壳 + 顶盖」、「底座 + 支架」、「同一根细长杆的相邻两段」。
   如果你列出了 4 个以上的部件，先停下来——**几乎一定**有部件你其实没看到红色，
   只是被自己想象进去了。**回去删掉那些不确定的**。
2. **只列你能在某张图里直接指出"这块红色在这里"的部件**。
   - 淡灰、白色、米色都**不算**红色。
   - 部件看起来是灰色但你"觉得它应该被涵盖" → **不能**列入。
   - 红色"路过"某个部件的边缘但没明显覆盖 → 不算。
3. 不知道用户在做什么任务，**不要**预先脑补"它应该是哪个部件"，
   也不要为了让答案"显得严谨"而多写几个部件。
4. 颗粒度参考（**禁止**用左/右/前/后/上/下等方位词）：用简洁名词描述部件
   （如 底座 / 支架 / 面板 / 把手 / 轮子 / 刀刃 / 管道 / 外壳 / 顶盖 / 腿 / 座面 / 靠背 等；
   若上面给了候选部件清单，优先从中选用）。成对对称部件只写一次统称（如「腿」不写「左腿」）。
5. 如果整张图**完全没有**鲜艳红色，`covered_parts` 写 `[]`，并在 notes 里说明。

输出格式（极简）
-----------------------------------
{
  "covered_parts": ["1~3 项；多于 3 项一定是错的，自己先删"],
  "dominant_region": "用一句话说明红色集中在物体哪个部位：例如 '集中在底座 + 支架'、'分布于整个外壳'、'只在一圈窄边'",
  "notes": "可选：补充任何观察细节"
}
"""


# 第 2 阶段「纯逻辑」——只用文字，不带图，决定 full / partial / none
REASON_SYSTEM_PROMPT = (
    "你是一个语义分类器。你不会看到任何图片。"
    "用户会告诉你：(1) 视觉观察员之前列出的 covered_parts（红色覆盖的物体部件）；"
    "(2) 用户要找的语义。你只需要按规则做集合比较并输出 JSON。"
)

REASON_USER_TEMPLATE = """前面的视觉观察员已经把"红色高亮覆盖的物体部件"列出来：
covered_parts = {covered_parts_json}
{context_block}
用户要找的语义：「{semantic}」

请按顺序推理，并把结果填到 JSON。

**核心规则：宽进 + 严判**
- 分类阶段（label）**宽**：只要 covered_parts 中**含有或可能包含**语义所在区域，就允许判 partial，
  让 pipeline 继续切，把语义切干净。
- 占比 / 强制承诺阶段（dominant_side / committed_if_forced）**严**：
  把握不到主导是语义就承诺 "none"，避免把"以底座为主、夹一点支架"的块在收尾时被错认为「支架」。

-----------------------------------
1. **想清楚「{semantic}」在该物体上对应哪些部件**（通常 1~3 项），填 `semantic_parts`。
   - 优先写**纯净的语义部件**：例如语义=「轮子」→ "轮子 / 轮胎 / 脚轮"；
     语义=「把手」→ "把手 / 手柄"。
   - **包容性补全**：如果 covered_parts 里只有**包含语义的笼统部件**而没有明确分出语义，
     依然要把语义部件写进 semantic_parts（视作 covered_parts 隐含包括它）。例如：
     语义=「轮子」，covered_parts = ["底盘整体"] → semantic_parts = ["轮子（包在底盘整体里）"]，
     不要因为没单独写出"轮子"就把 semantic_parts 留空判 none。
   - 但是：covered_parts 跟语义所在区域**完全无关**时（如语义=「轮子」，covered_parts=["顶盖","面板"]），
     semantic_parts 必须留空。

2. **`extra_parts`**：covered_parts 减去 semantic_parts 之后剩下的部件。
   - 把诸如 "外壳 / 底座 / 支架 / 面板 / 顶盖..." 这类**近邻但不属于语义**的部件老实写进去。
   - 笼统部件（如"底盘整体"）拆开后，**语义以外的那部分**也要算 extra（写成"底盘除轮子外区域"等）。

3. **`missing_overlap`**：covered_parts 跟语义所在区域是否**完全无关**。
   - 完全无关 → true（→ label=none）
   - 哪怕只是"covered_parts 笼统部位里包含语义" → false

4. **粗估"语义部分占整片红色面积"的占比**，填 `semantic_area_ratio`（0.0~1.0 浮点）。
   - 红色基本全是语义部位（如只覆盖一团头发）          → ≥ 0.85
   - 红色一半是语义、一半是非语义                       → ≈ 0.5
   - 红色绝大部分是 extra（如整张脸+颈，仅头顶一点头发）→ ≤ 0.2
   - **占比要诚实：脸大头发小，比例就要小**，不要因为"想留下这个块"而虚高。

5. **`dominant_side`**：
   - semantic_area_ratio ≥ 0.7 → "semantic"
   - semantic_area_ratio ≤ 0.3 → "extra"
   - 介于其间                  → "balanced"

6. **`label`（用于 pipeline 继续迭代的判定，**宽**）**：
   - covered_parts 为空 或 missing_overlap=true                   → "none"
   - extra_parts 非空                                              → "partial"
   - extra_parts 为空 且 semantic_parts 非空                       → "full"

7. **`committed_if_forced`（强制收尾时的最终承诺，**严**）**：
   - dominant_side = "semantic" 且 semantic_area_ratio ≥ 0.7 → "full"
   - 其它任何情况（extra / balanced / 不确定）              → "none"
   - **宁愿漏掉一个 balanced 块，也绝不能让 extra-dominant 的块溜进最终掩码。**

只能输出如下 JSON（缺字段视为错误）：
{{
  "semantic_parts": ["..."],
  "extra_parts": ["..."],
  "missing_overlap": true | false,
  "semantic_area_ratio": 0.0~1.0,
  "dominant_side": "semantic" | "extra" | "balanced",
  "label": "full" | "partial" | "none",
  "committed_if_forced": "full" | "none",
  "confidence": 0.0~1.0,
  "reason": "用一句中文总结：覆盖到了什么、语义占比大概多少、为什么 label/committed_if_forced 给这个值。"
}}
"""

CLASSIFY_USER_TEMPLATE = """用户给定的语义区域：
「{semantic}」

==================================
任务：判断**红色高亮分块**相对于「{semantic}」是 full / partial / none。
==================================

⚠️ 重要观察纪律（必须严格遵守）
-----------------------------------
1. 不要"扫一眼某个视角就下结论"。你必须**逐张图**检查红色覆盖区域。
2. 如果不同视角下红色出现在身体的不同部位（比如一视角红在头顶 + 胸口，另一视角红在腰部），
   这些**全部**都算高亮分块的一部分，**不能只挑一处**作答。
3. 高亮往往会跨越多个解剖部位（例如同时覆盖「头发 + 面部 + 躯干」）——
   这种情况几乎一定是 partial。
4. 你天生有"确认偏差"倾向：看到红色就想说"是的就是用户问的语义"。**对抗这种倾向**：
   先把红色覆盖到的部位**全部列出来**，再和「{semantic}」对照。

判定标准
-----------------------------------
- "full"：红色覆盖的每一处部位都属于「{semantic}」。允许只覆盖语义的一小段，
  但**不能多出任何一寸**不属于语义的几何。
  例：语义=「面部」，红色仅覆盖「眼睛」 → full。
- "partial"：红色与「{semantic}」有交集，但**同时还覆盖了明显不属于语义的部位**
  （不论是相邻部位还是远端零件——脸 / 颈 / 胸 / 背 / 腰 / 腹 / 臂 / 前臂 / 手 / 大腿 / 小腿 / 脚 等）。
  例：语义=「头发」，红色覆盖「头发 + 面部 + 胸前条带」 → partial。
  例：语义=「腰带」，红色覆盖「腰带 + 鞋子 + 腕带」 → partial。
- "none"：红色与「{semantic}」完全没有交集。
  例：语义=「面部」，红色覆盖「腿部」 → none。

输出格式（必须包含全部字段，按顺序填写）
-----------------------------------
{{
  "covered_parts": [
    "用中文列出红色覆盖到的所有部位，每个部位单独一项；",
    "颗粒度示例：'头顶尖发'、'整张脸'、'颈部'、'胸前竖条带'、'上臂'、'腰部一圈'、'双脚鞋面'（禁止方位词）。",
    "若某视角下红色仅是上一项部位的同一区域换角度，不重复列。"
  ],
  "semantic_parts": [
    "用中文列出 '{semantic}' 在该角色身上对应的部位（通常 1~3 项）。",
    "例：语义='头发' → ['头顶发型']；语义='腰带' → ['腰部一圈细条带']。"
  ],
  "extra_parts": [
    "covered_parts 减去 semantic_parts 之后剩下的部位，即'红色多出来不属于语义的部分'。",
    "如果空，写 []；只要非空，label 必须是 partial 或 none。"
  ],
  "missing_overlap": "true / false 字符串：若 covered_parts 与 semantic_parts 完全无交集为 true，否则 false",
  "label": "full | partial | none",
  "confidence": 0.0~1.0,
  "reason": "用一句中文总结：具体红色覆盖到了什么，为什么得到这个 label。"
}}

label 决策表（**严格按此推导，不要凭感觉**）
-----------------------------------
- extra_parts 为空 且 missing_overlap=false → "full"
- extra_parts 非空 且 missing_overlap=false → "partial"
- missing_overlap=true                       → "none"

切记：宁可保守判 partial 触发再分裂，也不要乐观判 full。
"""


SEED_SYSTEM_PROMPT = (
    "你是一个 3D 模型语义分析助手。"
    "用户给你 12 个视角拼接的渲染图 + 上一轮分类得到的 covered_parts / extra_parts，"
    "当前红色分块只『局部符合』用户语义，需要再切成若干子块继续判断。"
    "你的 seed_count 与 strategy 必须保证切分后**不丢失任何已观察到的部位**（避免昂贵重选）。"
    "请严格按 JSON 格式回答。"
)

SEED_USER_TEMPLATE = """用户给定的语义区域：「{semantic}」

上一轮分类已经得到的结构化信息（请优先采信，不要凭图片再臆测）：
- covered_parts（红色覆盖的所有部位）= {covered_parts_json}
- semantic_parts（其中真正属于「{semantic}」的部位）= {semantic_parts_json}
- extra_parts（红色多出来不属于语义的部位）= {extra_parts_json}
- 当前块面数 ≈ {block_faces}

任务：估计要把它**切成几个子块**，才能让其中**至少一个子块只覆盖 semantic_parts**，
其它子块只覆盖 extra_parts。

开放语义纪律（必须遵守）
-----------------------------------
- covered_parts / semantic_parts / extra_parts 是**上一阶段 VLM 开放观察**的结果，
  可能包含任意中文短名（道具零件、动物部位、机械组件等），**不要**套固定人体模板。
- seed_count 与 strategy 都应服务于「把这些**实际观察到**的部位拆开」，而不是凭常识臆测。
- 若观察列了 N 个不同部位，seed_count 通常 ≥ N（可在 N~N+1 之间，上限 {max_seeds}）。
  例：covered_parts=[椅背竖条, 椅面, 椅腿] → seed_count=3 或 4。
- extra_parts 非空时，往往要多 1 刀把语义与多余区域分开。

{semantic_coverage_discipline}

同时选择几何分块策略（strategy）与 params：
- 根据**当前块的开放语义描述**与**面数规模 block_faces** 综合选策略，不是一律 geodesic_fps。
- graph_cut：观察部位被细缝/折线/凹槽分开（凹陷明显时首选，lambda_concave≥0.8）；
  spectral：多个远距离部位粘在同块（默认首选）；选 spectral 时须给出 geodesic_weight /
  curvature_weight / concavity_weight（**默认就偏向凹度 concavity_weight=0.70；凹槽/接缝明显时再上调到 0.75~0.85**）；
  skeleton：树状/肢类骨架结构，沿中轴分叉切分（大图自动池化）；
  cvt：需均匀切同等碎片；sdf：描述里强调粗细/厚度差（大图自动减少 num_rays）；
  geodesic_fps：语义不明或其它策略不适用时的兜底，不是面数大的默认选择。
- block_faces 很大时：降低 lloyd_iter / num_rays，仍保持语义最匹配的策略。

参数纪律：下面每个字段都给了默认值；**不确定就沿用默认（可整段省略 params）**，
只在渲染图/语义明显需要时才覆盖某个参数。覆盖时也要在 reason 里说明依据。

只能返回如下 JSON：
{{
  "seed_count": <int, 最少 2，最多 {max_seeds}>,
  "strategy": "graph_cut" | "spectral" | "cvt" | "sdf" | "skeleton" | "geodesic_fps",
  "reason": "中文简述：为什么这个 seed_count 和 strategy；逐项说明各 matched/covered 部位切分后如何落到不同子块",
  "params": {{
    "lambda_smooth": <float, graph_cut, 0.1~3.0, 默认 1.0>,
    "lambda_concave": <float, graph_cut, 0.0~2.0, 凹陷明显时≥0.8, 默认 0.85>,
    "geodesic_weight": <float, spectral, 0.05~0.80, 默认 0.20>,
    "curvature_weight": <float, spectral, 0.05~0.80, 默认 0.15>,
    "concavity_weight": <float, spectral, 0.05~0.90, 默认 0.70, 凹陷明显时 0.75~0.85>,
    "lloyd_iter": <int, cvt/geodesic_fps/skeleton, 2~12, 默认 cvt=4 / 其它=0>,
    "contraction_iters": <int, skeleton, 2~12, 默认 5>,
    "graph_vertex_budget": <int, spectral/skeleton/geodesic_fps, 4000~20000, 可选, 默认自动池化>,
    "num_rays": <int, sdf, 4~24, 默认 8>,
    "cone_angle_deg": <float, sdf, 15~120, 默认 60>
  }}
}}
（params 只填所选策略相关字段；不确定就省略，系统会用上面的默认值。）
"""


# ---------------------------------------------------------------------------
# 分块策略选择：图割 / 谱聚类 / CVT / SDF
# ---------------------------------------------------------------------------

SPLIT_STRATEGY_SYSTEM_PROMPT = (
    "你是一个 3D 网格分块算法专家。"
    "用户会给你当前分块的语义上下文、面数规模、matched_parts 与 seed_count，以及可选的渲染图；"
    "你需要在**语义覆盖完整**（切分后每项语义都能落到独立子块）、**语义适配**与**运行耗时**之间权衡，"
    "从六种分块策略中选最合适的一种，并给出参数。"
    "严格按 JSON 格式回答，禁止 Markdown，禁止额外文字。"
)

SPLIT_STRATEGY_USER_TEMPLATE = """当前分块需要继续几何细分。

语义上下文
-----------------------------------
- 任务语义 / 命中部位：{semantic_context}
- covered_parts（红色覆盖）= {covered_parts_json}
- semantic_parts（属于语义）= {semantic_parts_json}
- extra_parts（多出来的部位）= {extra_parts_json}
- matched_parts（多标签命中，全覆盖模式）= {matched_parts_json}
- 计划切分块数 seed_count = {seed_count}
- 当前块面数 ≈ {block_faces}
- 细分阶段：{split_stage}
- CLI 策略提示（仅供参考，你有最终裁量权）：{strategy_hint}

性能与规模（系统会自动优化，勿因面数大而一律选 geodesic_fps）
-----------------------------------
- 测地类策略（spectral / skeleton / geodesic_fps / cvt）在大块上会自动**池化顶点图**
  （约 8k~12k 图节点），Dijkstra 复杂度可控；已给定 seed_count 时 spectral **不做特征分解**。
- block_faces < 30_000：优先 **spectral**（语义远距分离质量好）。
- block_faces 30_000~150_000：按语义选 spectral / skeleton / graph_cut / sdf；系统自动降 lloyd_iter / 射线数。
- block_faces ≥ 150_000：仍可选 spectral / skeleton，但 params 中 lloyd_iter 建议 2~3；
  **geodesic_fps 仅作语义不明或前述策略失败时的兜底**，不是默认首选。
- graph_cut / sdf 射线开销随面数线性涨：超大块时减少 num_rays 或 lloyd_iter，而非换无语义策略。

可选分块策略（strategy 字段只能取以下之一）
-----------------------------------
1. **graph_cut**（图割法）
   - 适合：边界需沿凹槽/凹谷走、相邻语义被细缝/折线分开（如手指缝、装甲接缝、裙摆褶皱、腋窝、颈根）。
   - **凹陷明显时首选 graph_cut**，边界会沿凹缝走，切分效果最好。
   - 参数：lambda_smooth（边界平滑，0.3~2.0，默认 1.0）、
     lambda_concave（凹缝吸引，0.5~2.0，**默认 0.85**；凹陷明显时 **0.9~1.2**）。

2. **spectral**（谱聚类）
   - 适合：整体拓扑/姿态变化大、需等距不变形切分（如四肢与躯干连接、弯曲肢体、对称重复结构）。
   - 边权 = 三项能量加权和（系统会自动归一化）：
     geodesic_weight（测地距离，远距部位分离）、
     curvature_weight（二面角弯曲，沿一般折线/弯曲处走界）、
     concavity_weight（凹度/凹槽，关节/接缝/沟槽处走界）。
   - **默认就以凹度为主导**（concavity_weight=0.70）：沿凹槽/接缝/关节走界的切分质量最好，
     **凹槽/接缝/沟槽明显**（附图可见凹线、指缝、腹股沟等）→ 再上调 concavity_weight 到 **0.75~0.85**；
     平滑弯曲、少凹槽 → 提高 curvature_weight；
     远距肢体粘连 → 提高 geodesic_weight。
   - 默认：geodesic_weight=0.20, curvature_weight=0.15, concavity_weight=0.70（三项会自动归一化）。

3. **cvt**（变分质心 Voronoi）
   - 适合：需要大小均匀、形状紧凑圆润的子块（如均匀切手指、把大块均分成可比碎片）。
   - 参数：lloyd_iter（Lloyd 迭代次数，2~12，默认 4；系统按面数自动缩放）。

4. **sdf**（形状直径函数）
   - 适合：有机生物模型按"粗细/厚度"区分肢体（躯干厚、手臂细、手指更细；四足动物的腿 vs 躯干）。
   - 参数：num_rays（射线数，4~24，默认 8；大块自动减少）、cone_angle_deg（锥角，15~120，默认 60）。

5. **geodesic_fps**（测地 FPS + Voronoi）
   - 适合：语义混杂难判断、或其它策略失败后的**兜底**；也适合需快速均匀切分的场景。
   - 参数：lloyd_iter（0~12，默认 0；大块用 2~4 提升均匀度）、graph_vertex_budget（4000~20000，可选，默认自动）。

6. **skeleton**（Laplacian 收缩骨架 + 测地 Voronoi）
   - 适合：树状/肢类骨架语义（人形、四足动物、多肢机械），沿中轴/分叉把肢体与躯干分开。
   - **同样适合把一条细长肢体沿长轴切成多段**（如 大腿/小腿/脚、上臂/前臂/手、脖子/头），
     即使分段之间**过渡平滑、没有明显凹缝或关节**——切口沿中轴等距、垂直于长轴，
     比 spectral/graph_cut 在「找不到凹缝」时乱切要稳得多。
   - **判据**：当前块明显细长（长轴远大于横截面）、且子语义是同一肢体上的纵向分段 → 选 skeleton。
   - 参数：contraction_iters（2~12，默认 5）、lloyd_iter（0~12，默认 0）、
     graph_vertex_budget（4000~20000，可选，默认自动）；大图自动池化图。

开放语义纪律
-----------------------------------
- 输入的 covered_parts / semantic_parts / extra_parts / matched_parts 均为**开放词表**，
  勿假设一定是人形；按文字描述的几何关系选策略。
- 多个 matched_parts 名称差异大、像不同组件 → 小中块用 spectral 或 graph_cut；
  **附图凹陷/接缝明显 → 优先 graph_cut（高 lambda_concave）或 spectral（concavity_weight 上调到 0.75~0.85）**。
- 描述强调整体粗细/厚度层次 → 小块用 sdf。
- 描述强调骨架/肢类/躯干-四肢链式结构 → skeleton（大图亦可用，系统自动池化）。
- **细长肢体的纵向分段（无明显凹缝/关节）→ 必须选 skeleton**：例如把大腿从腿上分出来、
  把小腿/脚分段，附图看不到凹线也不要退回 spectral/graph_cut 乱切；skeleton 会沿中轴等距切。
- **语义优先于规模**：大块仍选最合语义的策略；系统会用池化/降迭代保证可完成。
- geodesic_fps 是兜底而非默认：只有语义不明或凹谷/厚度/骨架策略都不合适时才选。

{semantic_coverage_discipline}

参数纪律
-----------------------------------
- 下面每个参数都标注了**默认值与取值范围**；这些默认值已是大多数情况下的较优起点。
- **不确定就沿用默认**：可以整段省略 params，或只填你确实想改的那几个字段，系统会对缺省项使用默认值。
- 只在**渲染图/语义明显需要**时才覆盖默认值（例如凹陷特别深 → 上调 concavity_weight / lambda_concave），
  并在 reason 里说明覆盖依据；不要无依据地乱改参数。

只返回如下 JSON：
{{
  "strategy": "graph_cut" | "spectral" | "cvt" | "sdf" | "skeleton" | "geodesic_fps",
  "reason": "中文简述：为什么选这个策略；逐项说明 matched_parts 各项切分后预期落在哪个子块（避免小部位被吞并）；若覆盖了任何默认参数，说明依据",
  "params": {{
    "lambda_smooth": <float, graph_cut, 0.1~3.0, 默认 1.0>,
    "lambda_concave": <float, graph_cut, 0.0~2.0, 默认 0.85, 凹陷明显时 0.9~1.2>,
    "geodesic_weight": <float, spectral, 0.05~0.80, 默认 0.20>,
    "curvature_weight": <float, spectral, 0.05~0.80, 默认 0.15>,
    "concavity_weight": <float, spectral, 0.05~0.90, 默认 0.70, 凹陷明显时 0.75~0.85>,
    "lloyd_iter": <int, cvt/geodesic_fps/skeleton, 0~12, 默认 cvt=4 / geodesic_fps=0 / skeleton=0>,
    "contraction_iters": <int, skeleton, 2~12, 默认 5>,
    "graph_vertex_budget": <int, spectral/skeleton/geodesic_fps, 4000~20000, 可选, 默认自动池化>,
    "num_rays": <int, sdf, 4~24, 默认 8>,
    "cone_angle_deg": <float, sdf, 15~120, 默认 60>
  }}
}}
（params 里只填所选策略相关的、且你想偏离默认的字段；其余一律省略，系统用默认值。）
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
            parts = "、".join(str(p).strip() for p in self.candidate_parts if str(p).strip())
            parts_line = f"它通常由这些部件构成（候选清单）：{parts}。"
        else:
            parts_line = ""
        return "\n" + OBSERVE_CONTEXT_HINT.format(
            category=self.category.strip() or "未知物体", parts_line=parts_line
        )

    def reason_block(self) -> str:
        if self.is_empty():
            return ""
        bits = []
        if self.category.strip():
            bits.append(f"物体类别：{self.category.strip()}")
        if self.candidate_parts:
            parts = "、".join(str(p).strip() for p in self.candidate_parts if str(p).strip())
            bits.append(f"该物体的候选部件清单：{parts}")
        return "\n（物体上下文：" + "；".join(bits) + "）"


def build_classify_prompt(semantic: str, context: Optional["ObjectContext"] = None) -> ClassifyPrompt:
    return ClassifyPrompt(
        system=CLASSIFY_SYSTEM_PROMPT,
        user=CLASSIFY_USER_TEMPLATE.format(semantic=semantic.strip()),
    )


def build_observe_prompt(context: Optional["ObjectContext"] = None) -> ClassifyPrompt:
    """阶段 1：纯观察，不出现 semantic（避免确认偏差），但可注入物体类别 + 候选部件。"""
    context_block = context.observe_block() if context is not None else ""
    return ClassifyPrompt(
        system=OBSERVE_SYSTEM_PROMPT,
        user=OBSERVE_USER_PROMPT.replace("{context_block}", context_block),
    )


def build_reason_prompt(
    semantic: str,
    covered_parts: list[str],
    context: Optional["ObjectContext"] = None,
) -> ClassifyPrompt:
    """阶段 2：拿阶段 1 的 covered_parts 与语义做纯文本逻辑判定。"""
    import json as _json
    context_block = context.reason_block() if context is not None else ""
    return ClassifyPrompt(
        system=REASON_SYSTEM_PROMPT,
        user=REASON_USER_TEMPLATE.format(
            semantic=semantic.strip(),
            covered_parts_json=_json.dumps(covered_parts, ensure_ascii=False),
            context_block=context_block,
        ),
    )


# ---------------------------------------------------------------------------
# 全局自评：每轮迭代后让 VLM 看「整模型当前进度图」，自己决定要不要继续 / 修正
# ---------------------------------------------------------------------------

REVIEW_SYSTEM_PROMPT = (
    "你是一个 3D 模型语义分析的总督导。"
    "用户在多轮迭代里把模型切成若干分块，每块判定为 full（属于语义） / partial（局部符合，待再切） / none（不属于）。"
    "现在你拿到的图是**当前全局状态**：所有 face 都按当前最新判定染色——"
    "**鲜红 = 已确认属于「{semantic}」**；**亮黄 = 待继续切分的 partial**；"
    "**浅灰 = 已确认不属于**；**中灰 = 已被舍弃 / 触顶**。"
    "你的任务是：对照「{semantic}」语义，**整体审视红色覆盖区域是否准确**，"
    "并决定**是否还需要继续迭代**（继续切黄色块），还是当前的红色已经足够好可以收工。"
    "严格按 JSON 格式回答，禁止 Markdown，禁止额外文字。"
)

REVIEW_USER_TEMPLATE = """目标语义：「{semantic}」

迭代上下文（这些是**事实数据**，不是估计，必须当作硬约束）
-----------------------------------
- 当前已迭代 {iter_done} 轮，硬上限 {max_depth} 轮
- 当前红色面数：{full_faces} / {total_faces}（{full_ratio:.1%}）  ← 这是你能看到的红色总量
- 待分裂的黄色面数：{pending_faces} / {total_faces}（{pending_ratio:.1%}）
- 已舍弃的中灰面数：{abandoned_faces}

**铁律 — 防止 VLM 凭"语感"瞎终止**
-----------------------------------
1. **如实描述**：在 `red_covered_parts` 里**只**列你**真的能在某张图上看到红色**的部位。
   - 如果红色比例 < 3%（例如本轮 full_ratio={full_ratio:.1%}），意味着图上几乎看不见红色——
     这种情况下 `red_assessment` **不允许** = "matches"（因为没什么红色可"匹配"），
     必须 = "incomplete"。
   - 不要因为"我觉得已经差不多了"就编造"红色覆盖了头顶尖发"——你必须在图上**真的看到红色**。
2. **黄色块还多就别终止**：
   - 如果 pending_ratio ≥ 10% 且语义还没基本覆盖，`should_terminate` **必须** = false；
   - 当 pending_ratio < 5% 且红色看起来确实在语义所在区域 → 才允许 should_terminate=true。
3. **终止前再问自己**：如果终止，最终掩码就是当前**红色**部分。它能让人一眼认出「{semantic}」吗？
   - 不能 → 不要终止。

请按以下顺序思考并填入 JSON
-----------------------------------
1. **观察红色覆盖**：先把眼睛聚焦到红色像素本身。它在角色身上的哪里？写进 `red_covered_parts`。
   - 红色比例 {full_ratio:.1%}：如果 <3%，这一项几乎一定写 [] 或像 "零星几片，未形成区域"。
2. **判断红色质量**：
   - matches    ：红色基本就在「{semantic}」位置且占比合理；
   - incomplete ：红色明显不够（典型：full_ratio < 该语义合理范围）；
   - over_claimed：红色多出非语义部位；
   - partial    ：既缺又多。
3. **判断黄色块的去留**：黄色还有没有继续切的必要？
   - 黄色覆盖了仍可能含「{semantic}」的部位（如头部内还有发/帽待分） → keep_splitting；
   - 黄色全是肯定不属于语义的部位（手脚） → likely_not_semantic；
   - 黄色都是边角碎片 → mostly_done。
4. **终止决定 should_terminate**：
   - 综合上面两项 + pending_ratio + 上面的「铁律」给出 true / false；
   - 当前 pending_ratio={pending_ratio:.1%}，full_ratio={full_ratio:.1%}：如果 full_ratio 极小且 pending 还很多，**绝不**终止。

只能返回如下 JSON：
{{
  "red_assessment": "matches | over_claimed | incomplete | partial",
  "red_covered_parts": ["red 当前覆盖到的部位（如实写，看不到红就写空 []）"],
  "missing_regions": "如果 red 缺漏，描述应该被红色覆盖但目前不是红色的部位；否则空字符串",
  "extra_regions": "如果 red 过度，描述被涂红但其实不属于语义的部位；否则空字符串",
  "yellow_assessment": "should_keep_splitting | likely_not_semantic | mostly_done",
  "should_terminate": true | false,
  "confidence": 0.0~1.0,
  "reason": "用一句中文说明：根据 full_ratio={full_ratio:.1%} / pending_ratio={pending_ratio:.1%} 与红色实际覆盖，为什么得到这个 should_terminate。"
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
    """全局进度自评提示。"""
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
# 一次性分组：把过分割的 N 个候选区一次性分成 K 组，每组带 label
# 这是「VLM 选源点」的最直接形式 —— VLM 通过点选候选编号来定义最终切分
# ---------------------------------------------------------------------------

GROUP_SYSTEM_PROMPT = (
    "你是一个 3D 模型语义分析助手。"
    "用户已经把当前 partial 分块**过分割**成 N 个小候选区，"
    "每个候选区在 12 视角拼图里用一个**独立的鲜艳颜色**绘制，"
    "并在它的几何中心**画了一个圆角矩形数字标签**（1, 2, 3, ...）。"
    "上下文里其它面是**淡灰色**，仅用于定位，**不要**把它们划入任何组。"
    "你的任务：把这些候选编号**一次性合并**成 K 组（2≤K≤候选数），"
    "每组要么属于「{semantic}」要么不属于（或仍不确定）。"
    "严格按 JSON 格式回答，禁止 Markdown，禁止额外文字。"
)

GROUP_USER_TEMPLATE = """目标语义：「{semantic}」

候选区列表（共 {n_candidates} 个，每个候选区在 12 视角拼图里有独立颜色 + 圆角矩形数字标签）：
{candidate_table}

任务
-----------------------------------
请把上面 {n_candidates} 个候选编号**一次性**合并成 2~{max_groups} 个组，每组打 label：
  - **full** ：该组的候选**全部**只覆盖「{semantic}」对应的纯净几何，**没有任何**面部/颈部/躯干等近邻部位。
  - **none** ：该组的候选**全部**与「{semantic}」无关（如四肢、躯干、脚等）。
  - **partial**：该组里至少有一个候选**仍混着**语义和非语义（无法干净切开），需要再切。

⚠️ Full 标签的**严格门槛**（违反就该用 partial）
-----------------------------------
- 「{semantic}」=「头发」时，full 组的所有候选必须**只**覆盖头顶尖发 / 发束 / 头发本体，
  **不能包括**：整张脸、面部、额头、颈部、下巴、耳朵、头部除发型外的部分。
- 「{semantic}」=「腰带」时，full 组的所有候选必须**只**覆盖腰部一圈条带，
  **不能包括**：腹部、背部、臀部、胸前。
- 总原则：**只要一个候选里看得到任何非语义部位，这一组就该是 partial 而不是 full**。
- **宁可把整片都丢进 partial 让 pipeline 再切一遍，也不要把"差不多"的候选放进 full 组。**
  full 组一旦确定，pipeline 不会再验证它，会直接进最终掩码——错就一直错。

合并规则
-----------------------------------
1. **先逐候选独立判断**：候选 i 单独看：纯语义 / 纯非语义 / 混着？
   - 纯语义     → 候选标记 yes
   - 纯非语义   → 候选标记 no
   - 混着 / 不确定 → 候选标记 mixed
2. **合组**：
   - 所有 yes 候选 → full 组（拓扑上分散也无所谓）
   - 所有 no  候选 → none 组
   - 所有 mixed 候选 → 一个或多个 partial 组（按网格邻接关系拆，便于后续分裂）
3. **覆盖率铁律**：所有 1..{n_candidates} 编号必须被分配且仅分配到**一组**。
4. K（组数）通常 2~{max_groups}。

请按上面的**严格门槛**给出 JSON：
{{
  "candidates_assessment": [
    {{"id": 1, "verdict": "yes|no|mixed", "note": "一句话说明这个候选实际覆盖到哪里"}},
    {{"id": 2, "verdict": "...", "note": "..."}},
    ... (所有 {n_candidates} 个候选都列出)
  ],
  "groups": [
    {{
      "group_id": "A",
      "candidate_ids": [<int>, ...],
      "label": "full" | "partial" | "none",
      "confidence": 0.0~1.0,
      "description": "中文描述这组的候选覆盖了哪些部位"
    }},
    ...
  ],
  "reason": "中文一句：怎么得到上面的分组——尤其是 full 组凭什么不夹带任何非语义部位。"
}}
"""


def build_group_prompt(
    semantic: str,
    *,
    n_candidates: int,
    candidate_table: str,
    max_groups: int = 4,
) -> ClassifyPrompt:
    """生成一次性分组提示。`candidate_table` 是 VLM 友好的候选清单，
    形如 '- 1=红 (228,26,28)\\n- 2=蓝 (55,126,184)\\n...'。"""
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
# 全覆盖（Full-coverage）模式
#
# 思路：先枚举**短名、粗粒度、全覆盖**部位清单（初稿，后续可细化；对称实例合并命名）。
# 再把清单作为封闭候选集，逐块 multi-label：每个分块必须从候选中至少选 1 项，
#   - 1 个 → 该块直接归到该部位
#   - ≥2 个 → 分裂成 len(matched) 个种子继续切（套用 CC → segmat → FPS 级联）
# 直到每个叶分块只命中 1 个语义为止（不允许 background / 未归类面）。
# ---------------------------------------------------------------------------


ENUMERATE_SYSTEM_PROMPT = (
    "你是 3D 模型粗粒度语义分区员。"
    "看 12 视角整模型拼图（无高亮），输出**短名、全覆盖、互不重叠**的部位清单；"
    "对称重复部件合并为一项（写「手臂」不写「左臂/右臂」）。"
    "同类外附肢用**一个统称**（如节肢动物写「附肢」不写「步足」「触肢」并列）。"
    "名称宜简（2~6 字），不拆手指/花纹级细节。"
    "必须给出完整、可审计的**拓扑约束**（连通块数量 count、骨架连接 attach_to）。"
    "禁止在名称或说明中使用左/右/前/后/上/下等方位词。只输出 JSON。"
)


ENUMERATE_USER_TEMPLATE = """12 视角整模型拼图（淡色、无高亮）。

任务：粗粒度**全覆盖**分区——各项叠起来包住整个主体，彼此不重叠、不重复。
这是初稿，后续会细化；名称**短而准**（如「头」「躯干」「手臂」），勿写长描述。

命名与颗粒度
-----------------------------------
- 每项 2~6 字，不用括号、不用「含/及/+」合并多项。
- **禁止**用左/右/前/后/上/下等方位词区分对称实例：写「手臂」「前臂」「大腿」「小腿」「肢」等统称。
- 人形常见 6~8 项：头、颈、躯干、上臂、前臂、大腿、小腿（手/脚可并入前臂/小腿）。
- 四足：头、颈、躯干、尾、肢（或拆成两组对称肢，各 count=2）。
- **节肢动物**（虾/蟹/蜘蛛/蝎/多足类）：典型 3 项——头胸、腹部、**附肢**。
  所有外附肢（步足、触肢、螯、须等）**必须合并为一项「附肢」**；禁止同时列出「步足」「触肢」「螯」等细分名。
  附肢 count = 模型上外附肢实例总数（如蟹 10、蜘蛛 8）；附肢 attach_to = ["头胸"] 或 ["腹部"]（依模型骨架，须与主体邻接）。
  反例（禁止）：["头胸","腹部","步足","触肢"]；正例：["头胸","腹部","附肢"]。
- 简单道具/机械：2~4 项（椅面、椅背、椅腿 …）。
- 不拆手指/脚趾/发丝；不要把整条腿与躯干合成一项。
{hint_block}

约束
-----------------------------------
1. 全覆盖：主体几何都要能归入某一项；难归类则并入最近大块。
2. 互不重叠：不要同时列「躯干」与「上身」这类同义重复；也不要把同一类外附肢拆成多个近义项（如「步足」+「触肢」）。
3. 数量 2~12 项；**不得**出现 background / 其它 / 未分类 等兜底项。
4. **强约束**：最终每项至少 1 个 face；模型上不存在或无法切出的部位不要列入（收尾阶段会 delete 或 split 补救）。

拓扑约束（topology_constraints）——**核心，必须自洽**
-----------------------------------
为每个 semantic_parts 项给出**期望的连通块数量**与**骨架连接关系**（后续分割会按此审计修复）。
拓扑从**语义骨架**推断，不靠视角方位。

**count（互不相连几何块数）**
- 对称重复部件：count = 该语义在完整模型上的**实例数**。
  - 人形「上臂」「前臂」「大腿」「小腿」各 count=2（对称实例各一，但名称不写实例方位）。
  - 若清单含「四肢」或等价统称 → count=4。
  - 四足两组对称肢各 count=2；若只列一项「肢」→ count=4。
- 中心/唯一部件（躯干、头、颈、尾、椅面等）：通常 count=1。

**attach_to（必须邻接连接的上级部位，用 semantic_parts 原文）**
- 主体/根节点（躯干、椅面等）：attach_to = []。
- **所有肢类必须 attach_to 主体**：上臂/前臂/大腿/小腿/肢/附肢/尾 等 → attach_to = ["躯干"]（或对应主体名，节肢动物附肢→头胸或腹部）。
- **链式骨架**：子段连直接上级，例如 前臂→上臂→躯干；小腿→大腿→躯干；椅背/椅腿→椅面。
- **一致性铁律**：
  - 若拆出上臂+前臂+大腿+小腿（或等价四肢语义），四条肢链的**根**都必须 attach_to 躯干；
    合计应有 4 条独立肢链（或 count 之和 = 4 个肢实例）全部与躯干邻接。
  - attach_to 中的名称必须在 semantic_parts 中存在；不得指向未列出的部位。
  - 若某部位 count>0，其每个连通块都应至少与 attach_to 中某一上级共享网格边界。

- 名称必须与 semantic_parts **完全一致**；每项恰好 1 条 topology_constraints。

只返回 JSON：
{{
  "object_kind": "≤12 字，模型类型（如「类人角色」「四足动物」「椅子」）",
  "semantic_parts": ["短名1", "短名2", "..."],
  "topology_constraints": [
    {{"part": "躯干", "count": 1, "attach_to": []}},
    {{"part": "上臂", "count": 2, "attach_to": ["躯干"]}},
    {{"part": "前臂", "count": 2, "attach_to": ["上臂"]}},
    {{"part": "大腿", "count": 2, "attach_to": ["躯干"]}},
    {{"part": "小腿", "count": 2, "attach_to": ["大腿"]}},
    {{"part": "头", "count": 1, "attach_to": ["躯干"]}}
  ],
  "reason": "≤20 字，可选，说明分区与拓扑依据"
}}
"""


MULTI_LABEL_SYSTEM_PROMPT = (
    "你是 3D 分块多标签判别员。"
    "12 视角拼图里仅一块为鲜艳红/橙高亮，其余淡灰。"
    "候选集是枚举阶段给出的**开放语义短名**（可能是人形、动物、道具、机械等任意部件名）。"
    "只能从候选集中勾选红色**实际覆盖**的部位；**至少选 1 项**，禁止 background。"
    "若给出父块 parent_matched，子块**原则上只从 parent_matched 中选**；"
    "父块未列出的语义仅在你非常确定（confidences≥0.85）时才可勾选。"
    "若提示已达 max_depth，**matched_parts 只能 1 项**，必须给出最终确切归属，禁止多项。"
    "小部位（头、手、指等）若高亮区确有该几何也必须勾选，漏标会导致昂贵覆盖重选。"
    "只输出 JSON。"
)


MULTI_LABEL_USER_TEMPLATE = """从候选集中标出红色高亮块覆盖了哪些部位。
{parent_context_block}
候选（封闭集，只能从中选，须**原样拼写**）：
{candidate_table}

规则
-----------------------------------
1. 只认鲜艳红/橙；灰/白/米色不算红。
2. **matched_parts 至少 1 项**；看不清时选占比最大或几何最相关的候选，禁止留空。
3. 通常 1~3 个相邻候选；≥4 项先删减不确定项。
4. 候选未列出的几何 → 归入**网格邻接上最近**的候选，勿自造新词（保持封闭候选集与开放枚举词表一致）。
5. 若候选仅有「附肢」，即使高亮区像步足或螯，也只标「附肢」，勿写「步足」「触肢」等候选外名称。

{multi_label_coverage_discipline}
{parent_inheritance_block}
{max_depth_block}

只返回 JSON：
{{
  "matched_parts": ["候选原文", "..."],
  "dominant": "matched 中占比最大的一项（必填，且须在 matched_parts 内）",
  "confidences": {{"候选原文": 0.0~1.0, "...": "每项 matched 的把握度，可选但建议填写"}}
}}

说明：共 {n_candidates} 个候选；matched_parts 每项须与候选完全一致。
"""


SPLIT_COVERAGE_REPAIR_SYSTEM_PROMPT = (
    "你是 3D 分块多标签判别员。"
    "父块几何分裂后，各子块标注的并集须覆盖父块分裂前的全部语义。"
    "若某些父级语义在兄弟子块并集中缺失，请结合附图**重新判断本块**红色高亮区域实际覆盖的部位。"
    "只能从候选集中勾选；至少选 1 项。只输出 JSON。"
)


SPLIT_COVERAGE_REPAIR_USER_TEMPLATE = """父块「{parent_id}」分裂前标注：{parent_matched}

各兄弟子块当前标注（仅供参考；以你看到的**本块**几何为准）：
{siblings_table}

**并集中缺失、须由某个子块认领的语义**：{missing_parts}
（每项应恰好落入一个子块；仅当本图红色高亮区**确实覆盖**该部位时才加入 matched_parts）

本图高亮块：**{block_id}**

候选（封闭集，只能从中选，须原样拼写）：
{candidate_table}

规则
-----------------------------------
1. 只判断**本块**（鲜艳红/橙高亮）实际几何，勿照搬兄弟标注。
2. matched_parts 至少 1 项；若本块含缺失列表中的部位，**必须**纳入 matched_parts。
3. 本块不含的缺失项不要勾选。
4. 通常 1~3 个相邻候选；≥4 项先删减不确定项。
5. 小部位（头、手、指等）若本块高亮区确有该几何，**必须**纳入 matched_parts，禁止漏标。

只返回 JSON：
{{
  "matched_parts": ["候选原文", "..."],
  "dominant": "matched 中占比最大的一项（必填，且须在 matched_parts 内）"
}}
"""


EMPTY_SEMANTIC_REPAIR_SYSTEM_PROMPT = (
    "你是 3D 全覆盖语义审计员。"
    "枚举清单中每一项最终都必须至少占 1 个 mesh face，不允许 0 face 的空语义。"
    "对 0 face 的项，你只能二选一："
    "split=从已有块中分裂 carve 出该部位；delete=从清单删除该项。"
    "只输出 JSON。"
)


EMPTY_SEMANTIC_REPAIR_USER_TEMPLATE = """当前分割结果（12 视角按语义上色图）见附图。

语义清单与各部位 face 数（**0 表示空语义，必须处理**）：
{counts_table}

0 face 的语义（必须逐项给出 repair）：
{empty_list}

可供 split 时选取的 donor 块（只能从下表 block_id 里选；须足够大以便分裂）：
{donor_table}

任务
-----------------------------------
对每个 0 face 语义给出 **唯一** repair：
- **split**：从 donor 块里 carve 出该语义的几何。
  必须填 `donor_block_id`（上表之一），优先选几何上最可能包含该部位的块。
- **delete**：模型上不存在或已被其它语义覆盖，从清单移除该项。
  不要填 `donor_block_id`。

只能返回 JSON：
{{
  "repairs": [
    {{
      "empty_semantic": "须与上面 0 face 列表完全一致",
      "action": "split | delete",
      "donor_block_id": "split 时必填 block_id；delete 时省略或空字符串",
      "reason": "≤30 字"
    }}
  ]
}}

要求：repairs 必须覆盖全部 0 face 语义，每项恰好 1 条；不得遗漏。
"""


# ---------------------------------------------------------------------------
# 拓扑约束推断（用户已给定 semantic_parts、跳过枚举时补拓扑）
# ---------------------------------------------------------------------------

TOPOLOGY_INFER_SYSTEM_PROMPT = (
    "你是 3D 模型骨架拓扑分析员。"
    "根据整模型渲染图和给定的语义部位清单，推断各部位的连通块数量与骨架连接关系。"
    "对称重复部件在清单中已合并为一项时，count 写其实例总数。"
    "必须从语义骨架推断 attach_to 链，确保所有肢类最终连接主体。"
    "禁止在输出中使用左/右/前/后/上/下等方位词。只输出 JSON。"
)

TOPOLOGY_INFER_USER_TEMPLATE = """12 视角整模型拼图（淡色、无高亮）。

已给定的语义部位清单（封闭集，不得增删改名）：
{parts_table}

任务：为每项给出 topology_constraints（从**语义骨架**推断，不靠视角方位）。

**count**
- 对称重复部件：写完整模型上的**实例数**（如「上臂」=2，「肢」=4）。
- 唯一/中心部件（躯干、头、颈、尾、椅面等）：通常 =1。
- 若清单同时含上臂+前臂+大腿+小腿（或等价四肢语义），四肢实例合计应为 4。

**attach_to**（必须邻接连接的上级，用清单原文；主体/根写 []）
- 所有肢类必须 attach_to 主体（躯干/椅面等）。
- 链式骨架：子段连直接上级（前臂→上臂→躯干，小腿→大腿→躯干，椅背→椅面）。
- **铁律**：每条肢链的根都必须与主体邻接；attach_to 只能引用清单中已有部位。

只返回 JSON：
{{
  "topology_constraints": [
    {{"part": "清单原文", "count": 1, "attach_to": []}},
    ...
  ]
}}

要求：topology_constraints 必须覆盖清单中的**每一项**，且 part 与清单完全一致；count 与 attach_to 须互相自洽。
"""


# ---------------------------------------------------------------------------
# 拓扑违规修复（分割完成后审计 count / attach_to）
# ---------------------------------------------------------------------------

TOPOLOGY_REPAIR_SYSTEM_PROMPT = (
    "你是 3D 全覆盖语义拓扑审计员。"
    "根据渲染图、期望拓扑约束与实际连通块统计，找出违规并给出修复方案。"
    "核心原则：块数多了→relabel 到正确语义；块数少了→几何分裂 donor 块，"
    "由你决定切成几块、每块归哪个语义（不必全归同一语义）。"
    "修复须满足语义骨架：对称部件实例数正确、所有肢链根节点与主体邻接。"
    "修复动作："
    "reassign_component=改标已有连通块；"
    "geometric_split=几何切分 donor 块为 seed_count 块并按 child_assignments 归语义。"
    "禁止在说明中使用左/右/前/后/上/下等方位词。只输出 JSON。"
)

TOPOLOGY_REPAIR_USER_TEMPLATE = """当前分割结果（12 视角按语义上色图）见附图。

期望拓扑约束：
{constraints_table}

实际各部位连通块数（几何上互不相连的块数）：
{component_table}

邻接关系（实际哪些部位共享边界）：
{adjacency_table}

检测到的拓扑违规（须逐项修复）：
{violations_table}

可供 carve / subdivide 时选取的 donor 块（block_id / 当前语义 / 面数）：
{donor_table}

修复决策（按违规类型）
-----------------------------------
**count_high（块数过多 / 冗余）**：
  多出来的连通块**实际**是哪个部位？看图判断，不要简单删掉。
  → 优先 **reassign_component**：把多余块从 from_semantic 改标到它真正对应的 to_semantic。
    例：躯干 2 块，其中一块其实是头 → from=躯干, to=头, component_index=2。
    这可能同时修复另一个 count_low 的部位。

**count_low（块数不足）**：
  缺失的块去哪了？先判断是「合并了」还是「标错了」还是「嵌在别的块里」。
  情形 A — 同语义内多块合并（如两臂合成一块）：
    → **geometric_split**：切开 donor，child_assignments 可多个归同一语义。
  情形 B — 缺失块被标到别的语义：
    → **reassign_component**：from_semantic=错标处, to_semantic=缺块处。
  情形 C — 一块里混了多个语义、需拆开：
    → **geometric_split**：你决定 seed_count（2~{max_split}），
      child_assignments 列出切分后每块（按拼图数字标签 1..K 顺序）应归哪个语义；
      **不必全归同一语义**，例：躯干块里嵌着两条手臂 → 切成 3 块 [手臂,手臂,躯干]。
  情形 D — 四肢实例不足且应全部连接躯干：
    → 优先 geometric_split 从 donor 切出缺失肢实例，并确保切出块 attach_to 链正确。

**geometric_split 说明**：
  - seed_count：几何切分块数（≥2，按实际需要，可 >2）。
  - child_assignments：长度须等于 seed_count；index 从 1 起，与拼图上数字标签一一对应；
    按**语义拓扑**预分配（同类对称实例可重复同一语义名）；管线切分后会再让 VLM 看图确认编号对应。
  - donor_block_id：被切的叶块（block_id）。

**missing_attachment**：
  某部位应有 attach_to 上级但实际无共享边界 → 优先 reassign 到正确语义，或 geometric_split 把嵌套块剥离并改标，
  使每条肢链根节点与主体（躯干/椅面等）恢复邻接。

**illegal_adjacency（非法邻接，如头↔手臂）**：
  → **cut_adjacency**：from_semantic=部位A, to_semantic=部位B。
  管线将用拉普拉斯谱分解 + 凹缝测地切割，强制在几何上断开伪连接。

**pseudo_bridge / handle_loop（伪桥梁，如两腿粘连）**：
  → **cut_bridge**：from_semantic=部位A, to_semantic=部位B。
  管线将定位 handle loop 并切开桥梁，再按各自核心区域重标。

**boundary_refine（手术后边界锯齿）**：
  → **boundary_refine**：from_semantic/to_semantic 为需平滑的相邻部位对。
  管线将用凹角感知图割能量最小化精修边界。

禁止：不要随便 merge 进邻近语义；不要 delete_part。

若附图与统计一致、你认为**无实质违规**，返回空 repairs 列表。

只返回 JSON：
{{
  "topology_ok": true | false,
  "repairs": [
    {{
      "violation": "简述违规",
      "action": "reassign_component | geometric_split | cut_adjacency | cut_bridge | boundary_refine",
      "from_semantic": "reassign 时：错标语义",
      "to_semantic": "reassign 时：正确语义",
      "component_index": 1,
      "donor_block_id": "geometric_split 时必填",
      "seed_count": 2,
      "child_assignments": [
        {{"index": 1, "semantic": "手臂"}},
        {{"index": 2, "semantic": "手臂"}}
      ],
      "reason": "≤30 字"
    }}
  ]
}}
"""


GEOMETRIC_SPLIT_ASSIGN_SYSTEM_PROMPT = (
    "你是 3D 分块语义指派员。"
    "12 视角拼图里 donor 块已被几何切成若干编号子块（1..K，不同颜色+数字标签），"
    "其余几何为淡灰参考。请为每个编号指定应归的语义（只能从候选集中选）。"
    "只输出 JSON。"
)

GEOMETRIC_SPLIT_ASSIGN_USER_TEMPLATE = """donor 块 {donor_block_id} 当前语义：「{donor_semantic}」
几何切分后共 {child_count} 块，拼图上以数字 1..{child_count} 标出。

候选语义（封闭集，必须原样拼写）：
{candidate_table}

拓扑修复阶段建议的预分配（仅供参考，以你看到的几何为准）：
{planned_table}

任务
-----------------------------------
为每个编号 1..{child_count} 指定 semantic（候选原文）。
不同编号可以归同一语义，也可以归不同语义。

只返回 JSON：
{{
  "assignments": [
    {{"index": 1, "semantic": "候选原文"}},
    ... 必须覆盖 1..{child_count} 每个编号
  ]
}}
"""


def build_enumerate_prompt(*, hint: str = "") -> ClassifyPrompt:
    """构建「枚举全覆盖语义部位」提示。`hint` 注入额外约束
    （例如「四足猫，按动物骨架短名分区」）。"""
    hint_block = (
        f"- 用户补充：{hint.strip()}"
        if hint and hint.strip()
        else "- 无额外说明：按上图形态自选合适模板，保持短名全覆盖。"
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
    """构建「多标签 — 命中候选集中的哪些部位」提示。"""
    parts = [str(x).strip() for x in candidate_parts if str(x).strip()]
    if not parts:
        raise ValueError("candidate_parts 不能为空")
    candidate_table = "\n".join(f"- {i+1}. {p}" for i, p in enumerate(parts))

    parent_list = [str(x).strip() for x in (parent_matched or []) if str(x).strip()]
    if parent_list:
        parent_context_block = (
            f"\n父块语义约束（上一轮分裂前父块标注）\n"
            f"-----------------------------------\n"
            f"本块 **{block_id or '?'}** 由父块 **{parent_id or '?'}** 几何分裂而来。\n"
            f"父块 parent_matched（子块原则上只能从中选）：{parent_list}\n"
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
    """分裂后语义覆盖缺失：提醒 VLM 结合上下文重新勾选本块 matched_parts。"""
    parts = [str(x).strip() for x in candidate_parts if str(x).strip()]
    if not parts:
        raise ValueError("candidate_parts 不能为空")
    candidate_table = "\n".join(f"- {i+1}. {p}" for i, p in enumerate(parts))
    siblings_table = "\n".join(
        f"- {bid}: {list(matched)}"
        for bid, matched in sorted(sibling_matched.items())
    ) or "（无）"
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
    """用户已给定 semantic_parts 时，补推断拓扑约束。"""
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
    """拓扑违规修复：VLM 决定 reassign / geometric_split。"""
    constraints_table = "\n".join(
        f"- {c.get('part', '?')}: count={c.get('count', '?')}, "
        f"attach_to={c.get('attach_to', [])}"
        for c in topology_constraints
    ) or "（无约束）"
    component_table = "\n".join(
        f"- {p}: {int(component_counts.get(p, 0))} 块"
        for p in sorted(component_counts.keys())
    )
    detail_lines: List[str] = []
    for d in component_details:
        detail_lines.append(
            f"  - {d.get('semantic', '?')} #{d.get('index', '?')}: "
            f"{int(d.get('faces', 0))} face"
        )
    component_table += (
        "\n各块明细：\n" + "\n".join(detail_lines) if detail_lines else ""
    )
    adjacency_table = (
        "\n".join(f"- {a} ↔ {b}" for a, b in adjacency_pairs)
        if adjacency_pairs
        else "（无邻接）"
    )
    violations_table = (
        "\n".join(
            f"- [{v.get('kind', '?')}] {v.get('part', '?')}: {v.get('detail', '')}"
            for v in violations
        )
        if violations
        else "（无自动检测违规，请结合附图自行判断）"
    )
    if donor_blocks:
        donor_table = "\n".join(
            f"- {b['block_id']}: {b['semantic']}, {int(b['faces'])} face"
            for b in donor_blocks
        )
    else:
        donor_table = "（无可用 donor 块）"
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
    """几何切分后：VLM 为每个编号子块指定语义。"""
    candidates = [str(x).strip() for x in candidate_semantics if str(x).strip()]
    candidate_table = "\n".join(f"- {i+1}. {p}" for i, p in enumerate(candidates))
    if planned_assignments:
        planned_table = "\n".join(
            f"- #{int(a.get('index', 0))}: {a.get('semantic', '?')}"
            for a in planned_assignments
        )
    else:
        planned_table = "（无预分配）"
    return ClassifyPrompt(
        system=GEOMETRIC_SPLIT_ASSIGN_SYSTEM_PROMPT,
        user=GEOMETRIC_SPLIT_ASSIGN_USER_TEMPLATE.format(
            donor_block_id=str(donor_block_id).strip(),
            donor_semantic=str(donor_semantic).strip() or "（未知）",
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
    """0 face 语义补救：VLM 决定 split（从 donor 块 carve）或 delete。"""
    counts_table = "\n".join(
        f"- {p}: {int(face_counts.get(p, 0))} face"
        for p in semantic_parts
    )
    empty_list = "\n".join(f"- {p}" for p in empty_parts)
    if donor_blocks:
        donor_table = "\n".join(
            f"- {b['block_id']}: {b['semantic']}, {int(b['faces'])} face"
            for b in donor_blocks
        )
    else:
        donor_table = "（无可用 donor 块 → 只能 delete）"
    return ClassifyPrompt(
        system=EMPTY_SEMANTIC_REPAIR_SYSTEM_PROMPT,
        user=EMPTY_SEMANTIC_REPAIR_USER_TEMPLATE.format(
            counts_table=counts_table,
            empty_list=empty_list,
            donor_table=donor_table,
        ),
    )


# ---------------------------------------------------------------------------
# 同部位不连通块细分命名：一个「肢」断成 4 块 → 让 VLM 决定是否拆成
# 多个实例名（如 肢-1/肢-2），依据拓扑角色而非视角方位。
# ---------------------------------------------------------------------------

COMPONENT_NAME_SYSTEM_PROMPT = (
    "你是 3D 模型部位命名助手。"
    "某个语义部位在几何上断成了多个互不相连的块；"
    "12 视角拼图里每个块用**不同颜色 + 数字标签 (1..N)** 标出，其余几何为淡灰仅作参考。"
    "请结合各块与主体的**邻接/骨架角色**，决定是否给每个块一个更具体的名字"
    "（如 肢-1 / 肢-2、翼-1 / 翼-2）；"
    "若它们本就是同一部位的碎片、或无法可靠区分实例，则沿用原名。"
    "禁止在 label 中使用左/右/前/后/上/下等方位词。只输出 JSON。"
)

COMPONENT_NAME_USER_TEMPLATE = """原语义部位：「{part}」。它在几何上断成了 {n} 个互不相连的块。
拼图里这些块按数字标签 1..{n} 标出（每块一个独立颜色 + 重心数字标签）。

各块拓扑参考（与主体的连接关系、相对尺度，已按数字标签编号）：
{component_table}

任务
-----------------------------------
1. 逐个数字块看清它在主体骨架中的角色（结合 12 视角 + 上表邻接信息）。
2. 给每个编号一个 `label`：
   - 若这些块其实是**对称 / 重复的同类部件**（四条肢、一对翅膀、多个轮…），
     用**实例后缀**区分：「{part}-1」「{part}-2」… 或语义子类名（仍禁止方位词）。
   - 若它们应当是**同一个部位的碎片**、或实例无法可靠区分 → label 直接写原名「{part}」。
3. label 要短（≤8 字）；不要给毫无依据的实例名（拿不准就沿用原名）。
   同一原部位下，**真正不同的块**才用不同名字。

只返回 JSON：
{{
  "split": true | false,
  "components": [
    {{"id": 1, "label": "肢-1", "reason": "独立连通块，与躯干邻接"}},
    ... 必须为 1..{n} 每个编号各一条
  ]
}}
"""


def build_component_name_prompt(
    part: str,
    n: int,
    component_table: str,
) -> ClassifyPrompt:
    """构建「同部位不连通块细分命名」提示。"""
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
) -> ClassifyPrompt:
    """生成 VLM 分块策略选择提示。"""
    import json as _json

    hint = str(strategy_hint or "").strip()
    if not hint or hint == "auto":
        hint_text = "无（由你根据语义与 block_faces 全权决定）"
    else:
        hint_text = f"用户倾向 {hint}，但若与 block_faces 规模冲突，以规模约束为准"

    return ClassifyPrompt(
        system=SPLIT_STRATEGY_SYSTEM_PROMPT,
        user=SPLIT_STRATEGY_USER_TEMPLATE.format(
            semantic_context=str(semantic_context or "").strip() or "（未指定）",
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


def build_seed_prompt(
    semantic: str,
    *,
    max_seeds: int = 8,
    covered_parts: list[str] | None = None,
    semantic_parts: list[str] | None = None,
    extra_parts: list[str] | None = None,
    block_faces: int = 0,
) -> ClassifyPrompt:
    """生成 seed_count 询问提示。如果传入了 covered/extra 等上下文，
    VLM 就不用再凭图片臆测，能更稳地给出更高的 seed_count。
    """
    import json as _json

    return ClassifyPrompt(
        system=SEED_SYSTEM_PROMPT,
        user=SEED_USER_TEMPLATE.format(
            semantic=semantic.strip(),
            max_seeds=max(2, int(max_seeds)),
            covered_parts_json=_json.dumps(
                covered_parts or [], ensure_ascii=False
            ),
            semantic_parts_json=_json.dumps(
                semantic_parts or [], ensure_ascii=False
            ),
            extra_parts_json=_json.dumps(extra_parts or [], ensure_ascii=False),
            block_faces=int(block_faces),
            semantic_coverage_discipline=SEMANTIC_COVERAGE_DISCIPLINE,
        ),
    )


# ---------------------------------------------------------------------------
# 语义树：VLM 根据整模型渲染图生成层级分区树
# ---------------------------------------------------------------------------

SEMANTIC_TREE_SYSTEM_PROMPT = (
    "你是 3D 模型层级语义分区员。"
    "看 12 视角整模型拼图（无高亮），输出一棵**由粗到细**的语义树："
    "根节点代表整体，每一层把父部位拆成互不重叠、合并后全覆盖的子部位。"
    "对称重复部件在**同一层**合并为一项（写「手臂」不写「左臂/右臂」）。"
    "名称宜简（2~6 字）；深层可拆手指/装饰，但勿过度细碎。"
    "禁止在名称中使用左/右/前/后/上/下等方位词。只输出 JSON。"
)


SEMANTIC_TREE_USER_TEMPLATE = """12 视角整模型拼图（淡色、无高亮）。

任务：生成**层级语义树**——从根「整体」开始，每一层把父节点拆成若干子部位。
后续 pipeline 会从根逐层做几何切分；你只需给出**合理的层级结构**。

示例（类人，仅示意层级，勿照搬名称）：
  整体 → 上半身 / 下半身
  上半身 → 头 / 躯干 / 手臂
  手臂 → 大臂 / 小臂 / 手
  手 → 手掌 / 手腕 / 手指
  下半身 → 大腿 / 小腿 / 脚

命名与颗粒度
-----------------------------------
- 每项 2~6 字，不用括号、不用「含/及/+」合并多项。
- **禁止**用左/右/前/后/上/下等方位词。
- 第 1 层（根的子节点）宜粗：2~4 项（如上半身/下半身，或头/躯干/四肢）。
-  deeper 层逐步细化；叶子可以是手指、鞋面等细节。
- 子节点合并后应**完整覆盖**父节点几何，彼此不重叠。
- 树深度建议 2~{max_depth} 层（含根）；不要为凑深度而拆无意义项。
{hint_block}

约束
-----------------------------------
1. 根节点 name 固定为「整体」；children 非空。
2. 每个非叶节点的 children 至少 2 项、至多 8 项。
3. 叶节点 children = []。
4. 同一语义名不要在树的不同分支重复出现。
5. 不要出现 background / 其它 / 未分类 等兜底项。

只返回 JSON：
{{
  "object_kind": "≤12 字，模型类型（如「类人角色」「四足动物」「椅子」）",
  "tree": {{
    "name": "整体",
    "children": [
      {{
        "name": "子部位1",
        "children": [
          {{"name": "更细项", "children": []}}
        ]
      }}
    ]
  }},
  "reason": "≤30 字，说明层级划分依据"
}}
"""


def build_semantic_tree_prompt(
    *,
    hint: str = "",
    max_depth: int = 5,
) -> ClassifyPrompt:
    """构建「生成层级语义树」提示。"""
    hint_block = (
        f"- 用户补充：{hint.strip()}"
        if hint and hint.strip()
        else "- 无额外说明：按上图形态自选合适层级模板。"
    )
    return ClassifyPrompt(
        system=SEMANTIC_TREE_SYSTEM_PROMPT,
        user=SEMANTIC_TREE_USER_TEMPLATE.format(
            hint_block=hint_block,
            max_depth=max(2, int(max_depth)),
        ),
    )
