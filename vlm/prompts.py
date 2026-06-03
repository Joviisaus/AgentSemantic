"""
VLM 提示模板：
- 三分类（完全符合 / 局部符合 / 完全不符合）
- 局部符合时估计种子数（用于谱聚类分裂）

约定 VLM 必须返回严格的 JSON，便于稳定解析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Final, Sequence

LABEL_FULL: Final = "full"
LABEL_PARTIAL: Final = "partial"
LABEL_NONE: Final = "none"

VALID_LABELS: Final = (LABEL_FULL, LABEL_PARTIAL, LABEL_NONE)

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
    "在 12 视角拼图里，**只**报告颜色为**鲜艳红/橙**的高亮区域**实际**覆盖到的解剖部位。"
    "不要解释、不要猜测、不要联想任何语义；"
    "**淡灰色 / 米色 / 白色 / 任何非鲜艳红橙色的部位都不能列入**。"
    "**真实情况下，单个分块的红色通常只覆盖 1~3 个相邻部位**——"
    "如果你列出了超过 3 个部位，你**很可能在编造**，请重新只保留你**百分百在某张图里直接看到红色**的那些。"
    "严格按 JSON 格式回答，禁止 Markdown，禁止额外文字。"
)

OBSERVE_USER_PROMPT = """这是同一个角色的 12 个视角拼接渲染。
图中每个视角里只有**一个**分块被涂成**鲜艳红/橙色**，其他几何全部是淡灰色。

请逐视角看一遍，把红色**实际**覆盖到的角色部位**精简地**列出来。

观察纪律（违反即视为错误）
-----------------------------------
1. **真实分块的红色通常只在 1~3 个相邻部位**。例如「头+发+颈」、「腰+腹」、「右大腿+右小腿」。
   如果你列出了 4 个以上的部位，先停下来——**几乎一定**有部位你其实没看到红色，
   只是被自己想象进去了。**回去删掉那些不确定的**。
2. **只列你能在某张图里直接指出"这块红色在这里"的部位**。
   - 淡灰、白色、米色都**不算**红色。
   - 部位看起来是灰色但你"觉得它应该被涵盖" → **不能**列入。
   - 红色"路过"某个部位的边缘但没明显覆盖 → 不算。
3. 不知道用户在做什么任务，**不要**预先脑补"它应该是头/腿/腰带"，
   也不要为了让答案"显得严谨"而多写几个部位。
4. 颗粒度参考（任选）：头顶发型 / 整张脸 / 头部整体 / 颈部 / 胸前 / 胸前竖条带 /
   上腹 / 后背 / 腰部一圈 / 左/右上臂 / 左/右前臂 / 左/右手 / 左/右大腿 /
   左/右小腿 / 左/右脚 / 鞋面。
5. 如果整张图**完全没有**鲜艳红色，`covered_parts` 写 `[]`，并在 notes 里说明。

输出格式（极简）
-----------------------------------
{
  "covered_parts": ["1~3 项；多于 3 项一定是错的，自己先删"],
  "dominant_region": "用一句话说明红色集中在角色哪里：例如 '集中在头顶 + 颈部'、'分布于双腿前侧'、'只在腰部一圈窄环'",
  "notes": "可选：补充任何观察细节"
}
"""


# 第 2 阶段「纯逻辑」——只用文字，不带图，决定 full / partial / none
REASON_SYSTEM_PROMPT = (
    "你是一个语义分类器。你不会看到任何图片。"
    "用户会告诉你：(1) 视觉观察员之前列出的 covered_parts（红色覆盖部位）；"
    "(2) 用户要找的语义。你只需要按规则做集合比较并输出 JSON。"
)

REASON_USER_TEMPLATE = """前面的视觉观察员已经把"红色高亮覆盖的部位"列出来：
covered_parts = {covered_parts_json}

用户要找的语义：「{semantic}」

请按顺序推理，并把结果填到 JSON。

**核心规则：宽进 + 严判**
- 分类阶段（label）**宽**：只要 covered_parts 中**含有或可能包含**语义所在区域，就允许判 partial，
  让 pipeline 继续切，把语义切干净。
- 占比 / 强制承诺阶段（dominant_side / committed_if_forced）**严**：
  把握不到主导是语义就承诺 "none"，避免把"以脸为主、夹一点头发"的块在收尾时被错认为「头发」。

-----------------------------------
1. **想清楚「{semantic}」在角色身上对应哪些部位**（通常 1~3 项），填 `semantic_parts`。
   - 优先写**纯净的语义部位**：例如语义=「头发」→ "头顶发型 / 头顶尖发 / 后脑发束"；
     语义=「腰带」→ "腰部一圈条带"。
   - **包容性补全**：如果 covered_parts 里只有**包含语义的笼统部位**而没有明确分出语义，
     依然要把语义部位写进 semantic_parts（视作 covered_parts 隐含包括它）。例如：
     语义=「头发」，covered_parts = ["头部整体"] → semantic_parts = ["头顶发型（包在头部整体里）"]，
     不要因为没单独写出"头发"就把 semantic_parts 留空判 none。
   - 但是：covered_parts 跟语义所在区域**完全无关**时（如语义=「头发」，covered_parts=["小腿","左脚"]），
     semantic_parts 必须留空。

2. **`extra_parts`**：covered_parts 减去 semantic_parts 之后剩下的部位。
   - 把诸如 "面部 / 整张脸 / 颈部 / 胸前 / 上臂 / 大腿..." 这类**近邻但不属于语义**的部位老实写进去。
   - 笼统部位（如"头部整体"）拆开后，**头发以外的那部分**也要算 extra（写成"头部除头发外区域"等）。

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
1. 不要"扫一眼正面图就下结论"。你必须**逐张图**检查红色覆盖区域。
2. 如果不同视角下红色出现在身体的不同部位（比如正面图红在头顶 + 胸口，背面图红在腰部），
   这些**全部**都算高亮分块的一部分，**不能只挑一处**作答。
3. 高亮往往会跨越多个解剖部位（例如同时覆盖「头发 + 面部 + 上身」）——
   这种情况几乎一定是 partial。
4. 你天生有"确认偏差"倾向：看到红色就想说"是的就是用户问的语义"。**对抗这种倾向**：
   先把红色覆盖到的部位**全部列出来**，再和「{semantic}」对照。

判定标准
-----------------------------------
- "full"：红色覆盖的每一处部位都属于「{semantic}」。允许只覆盖语义的一小段，
  但**不能多出任何一寸**不属于语义的几何。
  例：语义=「面部」，红色仅覆盖「眼睛」 → full。
- "partial"：红色与「{semantic}」有交集，但**同时还覆盖了明显不属于语义的部位**
  （不论是相邻部位还是远端零件——脸 / 颈 / 胸 / 背 / 腰 / 腹 / 上臂 / 前臂 / 手 / 大腿 / 小腿 / 脚 等）。
  例：语义=「头发」，红色覆盖「头发 + 面部 + 胸前条带」 → partial。
  例：语义=「腰带」，红色覆盖「腰带 + 鞋子 + 腕带」 → partial。
- "none"：红色与「{semantic}」完全没有交集。
  例：语义=「面部」，红色覆盖「腿部」 → none。

输出格式（必须包含全部字段，按顺序填写）
-----------------------------------
{{
  "covered_parts": [
    "用中文列出红色覆盖到的所有部位，每个部位单独一项；",
    "颗粒度示例：'头顶尖发'、'整张脸'、'颈部'、'胸前竖条带'、'左上臂'、'腰部一圈'、'双脚鞋面'。",
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

强烈推荐的 seed_count 选法
-----------------------------------
- 默认 seed_count = max(2, len(covered_parts) + 1)。
  例：covered_parts=[头顶尖发, 整张脸, 颈部] → 3 个不同部位，建议 seed_count=4，留一个余量给边界过渡。
- 如果 extra_parts 里出现「面部 / 整张脸 / 颈部 / 头部整体 / 上臂 / 大腿」等
  **大块的躯干/头部解剖体**，几何上常和语义连在一起，**至少给 4**，便于谱聚类找到细的切线。
- 当前块面数越大（>3000）越倾向于多切几刀，**不要怯于给 4~5**。
- 仅当 covered_parts 只有 1 项（红色就一处连续小区域）才允许 seed_count=2。
- 上限 {max_seeds}。

之前太多 case 你都给了 2，结果一直切不开 → 这次请**主动多切一刀**。

只能返回如下 JSON：
{{
  "seed_count": <int, 通常 3 或 4，复杂时 5；最少 2，最多 {max_seeds}>,
  "reason": "中文简述：根据 covered_parts/extra_parts 列了几个部位，为什么需要这个 seed_count。"
}}
"""


@dataclass
class ClassifyPrompt:
    system: str
    user: str


def build_classify_prompt(semantic: str) -> ClassifyPrompt:
    return ClassifyPrompt(
        system=CLASSIFY_SYSTEM_PROMPT,
        user=CLASSIFY_USER_TEMPLATE.format(semantic=semantic.strip()),
    )


def build_observe_prompt() -> ClassifyPrompt:
    """阶段 1：纯观察，不带任何语义信息。"""
    return ClassifyPrompt(
        system=OBSERVE_SYSTEM_PROMPT,
        user=OBSERVE_USER_PROMPT,
    )


def build_reason_prompt(semantic: str, covered_parts: list[str]) -> ClassifyPrompt:
    """阶段 2：拿阶段 1 的 covered_parts 与语义做纯文本逻辑判定。"""
    import json as _json
    return ClassifyPrompt(
        system=REASON_SYSTEM_PROMPT,
        user=REASON_USER_TEMPLATE.format(
            semantic=semantic.strip(),
            covered_parts_json=_json.dumps(covered_parts, ensure_ascii=False),
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
- 「{semantic}」=「头发」时，full 组的所有候选必须**只**覆盖头顶尖发 / 后脑发 / 头发本体，
  **不能包括**：整张脸、面部、额头、颈部、下巴、耳朵、头部除发型外的部分。
- 「{semantic}」=「腰带」时，full 组的所有候选必须**只**覆盖腰部一圈条带，
  **不能包括**：腹部、后背、臀部、胸前。
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
   - 所有 yes 候选 → full 组（地理上分散也无所谓）
   - 所有 no  候选 → none 组
   - 所有 mixed 候选 → 一个或多个 partial 组（按空间相邻拆，便于后续分裂）
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
# 思路：先枚举**短名、粗粒度、全覆盖**部位清单（初稿，后续可细化；不区分左右）。
# 再把清单作为封闭候选集，逐块 multi-label：每个分块必须从候选中至少选 1 项，
#   - 1 个 → 该块直接归到该部位
#   - ≥2 个 → 分裂成 len(matched) 个种子继续切（套用 CC → segmat → FPS 级联）
# 直到每个叶分块只命中 1 个语义为止（不允许 background / 未归类面）。
# ---------------------------------------------------------------------------


ENUMERATE_SYSTEM_PROMPT = (
    "你是 3D 模型粗粒度语义分区员。"
    "看 12 视角整模型拼图（无高亮），输出**短名、全覆盖、互不重叠**的部位清单；"
    "左右对称肢合并为一项（写「手臂」不写「左臂/右臂」）。"
    "名称宜简（2~6 字），不拆手指/花纹级细节。只输出 JSON。"
)


ENUMERATE_USER_TEMPLATE = """12 视角整模型拼图（淡色、无高亮）。

任务：粗粒度**全覆盖**分区——各项叠起来包住整个主体，彼此不重叠、不重复。
这是初稿，后续会细化；名称**短而准**（如「头」「躯干」「手臂」），勿写长描述。

命名与颗粒度
-----------------------------------
- 每项 2~6 字，不用括号、不用「含/及/+」合并多项。
- **禁止**用「左/右」区分对称肢：写「手臂」「前臂」「大腿」「小腿」「前肢」「后肢」等统称。
- 人形常见 6~8 项：头、颈、躯干、上臂、前臂、大腿、小腿（手/脚可并入前臂/小腿）。
- 四足：头、颈、躯干、尾、前肢、后肢。
- 简单道具/机械：2~4 项（椅面、椅背、椅腿 …）。
- 不拆手指/脚趾/发丝；不要把整条腿与躯干合成一项。
{hint_block}

约束
-----------------------------------
1. 全覆盖：主体几何都要能归入某一项；难归类则并入最近大块。
2. 互不重叠：不要同时列「躯干」与「上身」这类同义重复。
3. 数量 2~12 项；**不得**出现 background / 其它 / 未分类 等兜底项。
4. **强约束**：最终每项至少 1 个 face；模型上不存在或无法切出的部位不要列入（收尾阶段会 delete 或 split 补救）。

只返回 JSON：
{{
  "object_kind": "≤12 字，模型类型（如「类人角色」「四足动物」「椅子」）",
  "semantic_parts": ["短名1", "短名2", "..."],
  "reason": "≤20 字，可选，说明分区依据"
}}
"""


MULTI_LABEL_SYSTEM_PROMPT = (
    "你是 3D 分块多标签判别员。"
    "12 视角拼图里仅一块为鲜艳红/橙高亮，其余淡灰。"
    "只能从封闭候选集中勾选红色**实际覆盖**的部位；**至少选 1 项**，禁止 background。"
    "只输出 JSON。"
)


MULTI_LABEL_USER_TEMPLATE = """从候选集中标出红色高亮块覆盖了哪些部位。

候选（封闭集，只能从中选，须**原样拼写**）：
{candidate_table}

规则
-----------------------------------
1. 只认鲜艳红/橙；灰/白/米色不算红。
2. **matched_parts 至少 1 项**；看不清时选占比最大或几何最相关的候选，禁止留空。
3. 通常 1~3 个相邻候选；≥4 项先删减不确定项。
4. 候选未列出的几何 → 归入**空间最近**的候选，勿自造新词。

只返回 JSON：
{{
  "matched_parts": ["候选原文", "..."],
  "dominant": "matched 中占比最大的一项（必填，且须在 matched_parts 内）"
}}

说明：共 {n_candidates} 个候选；matched_parts 每项须与候选完全一致。
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
) -> ClassifyPrompt:
    """构建「多标签 — 命中候选集中的哪些部位」提示。"""
    parts = [str(x).strip() for x in candidate_parts if str(x).strip()]
    if not parts:
        raise ValueError("candidate_parts 不能为空")
    candidate_table = "\n".join(f"- {i+1}. {p}" for i, p in enumerate(parts))
    return ClassifyPrompt(
        system=MULTI_LABEL_SYSTEM_PROMPT,
        user=MULTI_LABEL_USER_TEMPLATE.format(
            candidate_table=candidate_table,
            n_candidates=len(parts),
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
        ),
    )
