"""
VLM 提示模板：
- 三分类（完全符合 / 局部符合 / 完全不符合）
- 局部符合时估计种子数（用于谱聚类分裂）

约定 VLM 必须返回严格的 JSON，便于稳定解析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Final, Literal, Sequence

MediaType = Literal["mesh", "pointcloud"]

LABEL_FULL: Final = "full"
LABEL_PARTIAL: Final = "partial"
LABEL_NONE: Final = "none"

VALID_LABELS: Final = (LABEL_FULL, LABEL_PARTIAL, LABEL_NONE)

# ---------------------------------------------------------------------------
# 输入形态说明 + 判断方法总览（注入各阶段 prompt，让 VLM 知道「图是什么」「怎么判」）
# ---------------------------------------------------------------------------

RENDER_GUIDE: Final[Dict[str, str]] = {
    "mesh": (
        "【图像是什么】\n"
        "同一块 mesh 的 12 个视角拼图。每个视角里：\n"
        "  - **鲜艳红/橙色** = 当前待判定的**这一个分块**（三角面着色）；\n"
        "  - **淡灰色** = 模型其余部分，仅作空间定位，**不属于**当前分块。\n"
        "红色可能因视角遮挡只在一两个视角里明显——**必须逐视角检查**。"
    ),
    "pointcloud": (
        "【图像是什么】\n"
        "同一块点云的 12 个视角拼图。每个视角里：\n"
        "  - **鲜艳红/橙色圆点/小球** = 当前待判定的**这一个分块**的采样点；\n"
        "  - **淡灰色点** = 整模型其余点云，仅作定位，**不属于**当前分块。\n"
        "点云比 mesh 稀疏，红色可能呈点簇/点带；**逐视角找齐所有红点**再下结论。\n"
        "⚠️ 点云特有风险：小型道具/武器可能只是臂弯处一小簇红点，**勿只写笼统「手臂」而漏掉手上物件**。"
    ),
}

JUDGMENT_OVERVIEW_TEMPLATE: Final = """【VLM 如何判断 full / partial / none】
本系统用**两阶段**降低「看到用户语义就乱认」的偏差：

  阶段 1 · **纯观察（带图，不出现语义）**
    → 只回答：红色高亮**实际覆盖到哪些部位/部件** → 输出 covered_parts

  阶段 2 · **纯逻辑（不带图，只看文字）**
    → 拿 covered_parts 与用户要找的「{semantic}」做**集合比较** → 输出 label

三类 label 对后续 pipeline 的含义：
  - **full**    ：红色覆盖范围**全部落在**「{semantic}」内（含其子部件，如头部的耳朵）→ 不再切分
  - **partial** ：红色越界到语义**外**的独立大部件（如头部块夹了胸部）→ 触发几何再切
  - **none**    ：红色与「{semantic}」**完全无关** → 整块丢弃

**禁止**跳过观察直接猜 label；**禁止**在观察阶段偷看/联想用户语义。
"""

FORCE_TERMINATE_NOTE: Final = """
【强制收尾说明】
此分块已**无法继续几何切分**（点数过少或已达最大切分深度）。
你仍须按规则给出 label；同时 `committed_if_forced` 表示：
「若 pipeline 必须在此刻收工，这块最终算不算语义？」——**只能填 full 或 none**。
判 full 要非常确定红色主体就是「{semantic}」；有任何夹带非语义部位 → none。
"""

# 阶段 2 注入：当用户语义是手持/佩戴类道具时
PROP_SEMANTIC_NOTE_TEMPLATE: Final = """
【本任务语义类型：独立道具 / 手持物「{semantic}」】
观察员**可能只写了肢体名**（手臂、躯干）而漏写嵌在其中的道具——这很常见，尤其在点云。
只要 covered 或 dominant_region 暗示道具可能在持握/肩扛区域，**禁止**直接判 none；
应判 **partial**，让几何切分把道具从肢体上剥出来。
"""

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
    "你是一个严格、克制的 3D 视觉观察员。"
    "你的**唯一任务**是：在 12 视角拼图里，如实列出**鲜艳红/橙色高亮**实际覆盖到的部位/部件。"
    "用户尚未告诉你「要找什么语义」——**禁止**猜测、联想、解释任务目标。"
    "严格按 JSON 格式回答，禁止 Markdown，禁止额外文字。"
)

OBSERVE_ATTACHED_PROPS_BLOCK: Final = """
**肢体上的附属物 / 手持物（极易漏检，必须单独看）**
-----------------------------------
红色若覆盖手臂、手掌、躯干侧面，**不要只写笼统肢体名就结束**：
Step A. 沿肢体走向找：掌心、指间、前臂末端、肩扛/腋下位置有没有**另一簇**更细长的红点？
Step B. 若看见与肢体形态不同的突出几何（枪管、棍体、瓶身、刀刃、包袋、头盔…），**必须单独列一项**，
     用形状命名（如「手持长条形物件」「肩上扛着的棍状物」），**禁止**只写「右臂」了事。
Step C. 若肢体与物件点云粘成一片、实在分不清边界，写复合项：
     「右臂及手持物件」「躯干连肩扛物体」——**不要**因此省略物件。
"""

OBSERVE_USER_TEMPLATE = """{render_guide}
{attached_props_block}

{judgment_overview}

==================================
阶段 1 任务：纯观察（本阶段**不要**出现任何用户语义）
==================================

请**逐视角**检查 12 张图，把红色高亮**实际**覆盖到的部位/部件精简列出。

**怎么观察（必须按顺序做）**
-----------------------------------
Step 1. 从左到右、从上到下扫完 12 个视角；每个视角单独记下「这里有没有鲜艳红/橙」。
Step 2. 把所有视角里出现红色的位置**合并**成一份部位清单（同一部位多视角只列一次）。
Step 3. 对每个列出的部位，问自己：「我能在**某一张**图里指着红色说就在这里吗？」
        - 不能指着说 → **删掉**该项。
Step 4. 写 `dominant_region`：红色主要集中在模型哪里（一句话）。

观察纪律（违反即视为错误）
-----------------------------------
1. **真实分块的红色通常只在 1~3 个相邻部位**。列出 4 项以上时，先删掉不确定的。
2. **只列鲜艳红/橙色**；淡灰、白、米色**不算**红色。
3. 红色只擦过边缘、没有明显覆盖 → **不算**覆盖该部位。
4. 颗粒度参考（按模型类型选用，勿生搬硬套）：
   - 人形/动物：头/脸/颈/胸/背/腰/上臂/前臂/手/大腿/小腿/脚 …
   - 道具/机械：枪管/握把/扳机/弹匣/轮/翼/底座 …（用你**在图上看到的**形状命名）
5. 若**所有视角都没有**鲜艳红色 → `covered_parts` = []，在 notes 里说明。

输出格式（极简）
-----------------------------------
{{
  "covered_parts": ["1~3 项；多于 3 项先自查删项"],
  "dominant_region": "例如 '集中在枪管中段'、'分布于双腿前侧'、'只在腰部窄环'",
  "notes": "可选：补充观察细节"
}}
"""


# 第 2 阶段「纯逻辑」——只用文字，不带图，决定 full / partial / none
REASON_SYSTEM_PROMPT = (
    "你是一个语义分类器。你不会看到任何图片。"
    "用户会给你：(1) 阶段 1 观察员列出的 covered_parts（红色实际覆盖部位）；"
    "(2) 用户要找的语义关键词。"
    "你必须**严格按规则做集合比较**并输出 JSON，禁止凭语感跳步。"
)

REASON_USER_TEMPLATE = """{judgment_overview}
{force_terminate_note}
{prop_semantic_note}
==================================
阶段 2 任务：纯逻辑推理（**无图**，只看文字）
==================================

阶段 1 观察结果：
  covered_parts = {covered_parts_json}
  dominant_region = {dominant_region_json}

用户要找的语义：「{semantic}」

**推理步骤（必须按顺序执行，并把中间结果写入 JSON 字段）**
-----------------------------------
Step 1 → 填 `semantic_parts`：
  想清楚「{semantic}」在这个模型上对应哪些部位（通常 1~3 项）。
  - 人形例：语义=「头发」→ ["头顶发型"]；语义=「腰带」→ ["腰部一圈条带"]
  - 道具例：语义=「水枪」→ ["枪身/枪管整体"] 或更细的 ["握把","枪管"]（按 covered_parts 粒度对齐）
  - **包容性补全**：covered=["头部整体"]、语义=「头发」→ semantic_parts 仍应写出头发（视作头部整体里包含头发）
  - covered 与语义区域**完全无关**时（如语义=「头发」，covered=["小腿"]）→ semantic_parts = []

Step 2 → 填 `extra_parts`（**只计真正越界的部位，子部件不算 extra**）：
  extra_parts = covered_parts 里**落在语义范围之外**的部位。
  
  ⚠️ **语义包含 / 子部件规则（极重要，违反会导致无意义再切分）**：
  - 用户语义往往是**粗粒度**总称；covered 里更细的子部位若常识上**属于**该语义，**必须并入语义，不能写进 extra**。
  - 语义=「头部」：耳朵、面部、额头、颅顶、后脑、下巴、鼻子、眼睛、嘴巴、脸颊、头发、发型
    **都是头部的一部分** → extra 必须为 []（即使观察员分开写了「头部」「耳朵」）
  - 语义=「面部」：眼睛、鼻子、嘴巴、额头、脸颊、下巴 **属于面部** → 不算 extra
  - 语义=「头发」：头顶尖发、后脑发束属于头发；**面部、耳朵、颈部不属于头发** → 可算 extra
  - 语义=「水枪」：枪管、握把、扳机、弹匣 **都属于水枪** → 彼此不算 extra
  - 语义=「腰带」：腰部条带是语义；腹部、后背、腿部 **越界** → 算 extra
  
  **只有** covered 里出现了语义**范围外**的独立大部件，才填 extra 并判 partial。
  例：语义=「头部」，covered=["头部","耳朵"] → extra=[] → **full**（不是 partial）
  例：语义=「头部」，covered=["头部","颈部","胸前"] → extra=["颈部","胸前"] → partial
  例：语义=「头发」，covered=["头发","面部"] → extra=["面部"] → partial

  笼统部位要拆开时：covered=["头部整体"]、语义=「头发」→ extra 写「头部除头发外区域」

Step 3 → 填 `missing_overlap`：
  covered 与语义区域完全无交集 → true；否则 false
  
  ⚠️ **手持 / 佩戴 / 独立道具类语义**（枪、剑、棍、瓶、包、帽、工具…）：
  - 观察员只写了「躯干」「手臂」等肢体、**没写出道具名**，不代表一定无关！
  - 道具常**嵌在**手臂/手掌/肩扛区域的红点里；只要 covered 含 **手/臂/掌/指/腕** 或
    dominant_region 暗示「伸出、持握、肩扛、末端有物」→ **missing_overlap 必须为 false**，
    semantic_parts 写「{semantic}（可能嵌在肢体覆盖区内，待切分）」→ label 倾向 **partial**
  - 只有 covered **全是**与道具无关的远端肢体（如语义=「水枪」但红色只在「左脚/尾巴」）才可 true
  例：语义=「水枪」，covered=["躯干","右臂"] → **不可判 none**；missing_overlap=false，partial
  例：语义=「水枪」，covered=["手持长条物件","右臂"] → missing_overlap=false，partial 或 full
  例：语义=「水枪」，covered=["左小腿"] → missing_overlap=true，none

Step 4 → 估 `semantic_area_ratio`（0~1，红色里语义占多大比例）：
  几乎全是语义 → ≥0.85；一半一半 → ≈0.5；绝大部分是 extra → ≤0.2
  **诚实估**：脸大头发小，比例就要小

Step 5 → 填 `dominant_side`（由 ratio 推导）：
  ratio ≥ 0.7 → "semantic"；ratio ≤ 0.3 → "extra"；中间 → "balanced"

Step 6 → 填 `label`（决定要不要继续切）：
  ┌─ covered 为空 ────────────────────────────→ "none"
  ├─ missing_overlap=true（确与语义完全无关）──→ "none"
  ├─ extra_parts 非空（语义外大部件或肢体混道具）→ "partial"
  └─ extra 为空 且 covered 与语义有交集 ──────→ "full"

  **不要**因为观察员把同一语义拆成多个子部位名（头+耳、枪管+握把）就判 partial——
  子部位已全部落在语义范围内时，应直接 **full**，避免无意义再切分。

Step 7 → 填 `committed_if_forced`（**严**，强制收尾时最终承诺，仅 full/none）：
  dominant_side="semantic" 且 ratio ≥ 0.7 → "full"
  其它任何情况（extra / balanced / 不确定）→ "none"
  **宁漏勿滥**：extra 主导的块绝不能承诺 full

**核心原则：子部件归并 + 越界才切**
- 语义范围内的子部位（耳朵之于头部、握把之于水枪）**不是** extra，应判 **full**
- 只有红色明显越界到语义外的大部件时，才判 **partial** 触发几何再切
- committed_if_forced（收尾用）仍偏严：语义占比 ≥0.7 且 extra 为空 → full

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
  "reason": "一句中文：红色覆盖了什么、语义占比多少、为何给这个 label/committed_if_forced"
}}
"""

CLASSIFY_USER_TEMPLATE = """{render_guide}

{judgment_overview}

用户给定的语义区域：「{semantic}」

==================================
任务：判断**红色高亮分块**相对于「{semantic}」是 full / partial / none。
（单阶段直判版；正式 pipeline 优先用两阶段 observe → reason）
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
- "full"：红色覆盖的每一处部位都属于「{semantic}」——**包含该语义的子部件**
  （如语义=「头部」时，耳朵、面部、头发都算头部，不算越界）。
  例：语义=「头部」，红色覆盖「头部 + 耳朵」 → **full**。
  例：语义=「面部」，红色仅覆盖「眼睛」 → full。
- "partial"：红色与「{semantic}」有交集，但**同时还覆盖了语义范围外的独立大部件**
  （颈 / 胸 / 背 / 腰 / 腹 / 上臂 / 大腿 / 脚 …；或语义=「头发」时夹带面部）。
  例：语义=「头发」，红色覆盖「头发 + 面部」 → partial。
  例：语义=「头部」，红色覆盖「头部 + 胸前」 → partial。
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
    "covered 里**落在语义范围外**的部位（子部件如归并后为空则写 []）",
    "语义=「头部」时耳朵/面部不算 extra；语义=「头发」时面部才算 extra"
  ],
  "missing_overlap": "true / false 字符串：若 covered_parts 与 semantic_parts 完全无交集为 true，否则 false",
  "label": "full | partial | none",
  "confidence": 0.0~1.0,
  "reason": "用一句中文总结：具体红色覆盖到了什么，为什么得到这个 label。"
}}

label 决策表（**严格按此推导，不要凭感觉**）
-----------------------------------
- extra_parts 为空 且 missing_overlap=false → "full"
- extra_parts 非空（确为语义外大部件）且 missing_overlap=false → "partial"
- missing_overlap=true → "none"

子部件已全部落在语义内时直接 full，**不要**为「头+耳」这类组合再判 partial。
"""


SEED_SYSTEM_PROMPT = (
    "你是一个 3D 语义切分助手。"
    "上一轮已对红色分块做了 full/partial/none 判定；当前块是 partial，需要估计几何再切成几个子块。"
    "你会收到 12 视角拼图 + 上一轮的结构化观察结果（covered/semantic/extra parts）。"
    "请严格按 JSON 格式回答。"
)

SEED_USER_TEMPLATE = """{render_guide}

用户给定的语义区域：「{semantic}」

上一轮分类的结构化结果（**先验事实**，优先采信，不要凭图片推翻）：
- covered_parts（红色覆盖的所有部位）= {covered_parts_json}
- semantic_parts（其中属于「{semantic}」的部位）= {semantic_parts_json}
- extra_parts（红色多出来、不属于语义的部位）= {extra_parts_json}
- 当前块规模 ≈ {block_size} {size_unit}

==================================
任务：估计 seed_count（切成几个子块）
==================================

**目标**：切完后，**至少有一个子块只覆盖 semantic_parts**，其它子块落在 extra_parts 上。

**怎么估 seed_count（按顺序想）**
-----------------------------------
Step 1. 数 covered_parts 里有几个**不同部位** → 基准 = max(2, 部位数 + 1)
Step 2. 若 extra_parts 含大块躯干/头部/整段肢体 → 至少给 4（几何上难一刀切开）
Step 3. 块越大（>{large_threshold} {size_unit}）越倾向多切，复杂时给 4~5
Step 4. 仅当 covered_parts 只有 1 个小区域 → 才允许 seed_count=2
Step 5. 上限 {max_seeds}

例：covered=[头顶尖发, 整张脸, 颈部] → 3 部位 → 建议 4（留边界余量）
例：covered=[枪管, 握把]、semantic=[枪管] → 建议 3~4，把握把剥出去

只能返回如下 JSON：
{{
  "seed_count": <int, 通常 3 或 4，复杂时 5；最少 2，最多 {max_seeds}>,
  "reason": "中文：列了几个部位、为何需要这个 seed_count"
}}
"""


@dataclass
class ClassifyPrompt:
    system: str
    user: str


def _media_defaults(media_type: str) -> Dict[str, Any]:
    if media_type == "pointcloud":
        return {
            "size_unit": "点",
            "large_threshold": 3000,
        }
    return {
        "size_unit": "面",
        "large_threshold": 3000,
    }


def build_classify_prompt(semantic: str, *, media_type: MediaType = "mesh") -> ClassifyPrompt:
    guide = RENDER_GUIDE.get(media_type, RENDER_GUIDE["mesh"])
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
    """用户语义是否像手持/佩戴/独立道具（而非人体部位）。"""
    s = semantic.strip()
    if not s:
        return False
    prop_hints = (
        "枪", "剑", "刀", "棍", "杖", "瓶", "杯", "壶", "包", "袋", "帽", "盔", "盾",
        "弓", "箭", "锤", "斧", "铲", "锄", "筒", "炮", "机", "械", "具", "器", "棒",
        "铲", "鞭", "链", "铃", "笛", "琴", "伞", "扇", "镜", "灯", "钥", "币", "牌",
        "箱", "盒", "罐", "碗", "盘", "勺", "叉", "筷", "笔", "刷", "锯", "钳", "锤",
    )
    body_hints = (
        "头", "脸", "面", "发", "颈", "胸", "背", "腰", "腹", "臀", "腿", "脚", "趾",
        "手", "指", "臂", "肘", "膝", "眼", "鼻", "嘴", "耳", "牙", "爪", "尾", "翼",
    )
    if any(h in s for h in prop_hints):
        # 「机枪手」这类含人体词但核心是枪的，仍当道具
        if s.endswith("手") and len(s) <= 4:
            return False
        return True
    # 纯造词/英文名等：无人体部位特征时保守不注入
    return not any(h in s for h in body_hints)


def build_observe_prompt(*, media_type: MediaType = "mesh", semantic: str = "") -> ClassifyPrompt:
    """阶段 1：纯观察，不带任何语义信息。

    观察阶段**不注入真实 semantic**，避免确认偏差；总览里用语义占位符。
    """
    sem_placeholder = "（阶段 2 才会告知，本阶段禁止猜测）"
    guide = RENDER_GUIDE.get(media_type, RENDER_GUIDE["mesh"])
    overview = JUDGMENT_OVERVIEW_TEMPLATE.format(semantic=sem_placeholder)
    # 点云 + 所有形态：强调肢体上附属物，降低「只写手臂漏掉枪」的概率
    attached = OBSERVE_ATTACHED_PROPS_BLOCK
    return ClassifyPrompt(
        system=OBSERVE_SYSTEM_PROMPT,
        user=OBSERVE_USER_TEMPLATE.format(
            render_guide=guide,
            attached_props_block=attached,
            judgment_overview=overview,
        ),
    )


def build_reason_prompt(
    semantic: str,
    covered_parts: list[str],
    *,
    media_type: MediaType = "mesh",
    force_terminate: bool = False,
    dominant_region: str = "",
) -> ClassifyPrompt:
    """阶段 2：拿阶段 1 的 covered_parts 与语义做纯文本逻辑判定。"""
    import json as _json

    sem = semantic.strip()
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
    dom = (dominant_region or "").strip() or "（未提供）"
    return ClassifyPrompt(
        system=REASON_SYSTEM_PROMPT,
        user=REASON_USER_TEMPLATE.format(
            judgment_overview=JUDGMENT_OVERVIEW_TEMPLATE.format(semantic=sem),
            force_terminate_note=force_note,
            prop_semantic_note=prop_note,
            semantic=sem,
            covered_parts_json=_json.dumps(covered_parts, ensure_ascii=False),
            dominant_region_json=_json.dumps(dom, ensure_ascii=False),
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


# ---------------------------------------------------------------------------
# 同部位不连通块细分命名：一个「腿」断成 4 块 → 让 VLM 决定要不要拆成
# 左前腿/右前腿/左后腿/右后腿（参考各块在模型上的方位）。
# ---------------------------------------------------------------------------

COMPONENT_NAME_SYSTEM_PROMPT = (
    "你是 3D 模型部位命名助手。"
    "某个语义部位在几何上断成了多个互不相连的块；"
    "12 视角拼图里每个块用**不同颜色 + 数字标签 (1..N)** 标出，其余几何为淡灰仅作参考。"
    "请结合各块在模型上的**左右 / 前后 / 上下方位**，决定是否给每个块一个更具体的名字"
    "（如 左前腿 / 右前腿 / 左后腿 / 右后腿、左翼 / 右翼）；"
    "若它们本就是同一部位的碎片、或无法可靠区分方位，则沿用原名。"
    "只输出 JSON，禁止 Markdown，禁止额外文字。"
)

COMPONENT_NAME_USER_TEMPLATE = """原语义部位：「{part}」。它在几何上断成了 {n} 个互不相连的块。
拼图里这些块按数字标签 1..{n} 标出（每块一个独立颜色 + 重心数字标签）。

各块方位参考（模型局部坐标系，已按数字标签编号）：
{component_table}

坐标约定：+X=右 / -X=左，+Y=上 / -Y=下，+Z=前 / -Z=后。

任务
-----------------------------------
1. 逐个数字块看清它在主体上的位置（结合 12 视角 + 上表坐标）。
2. 给每个编号一个 `label`：
   - 若这些块其实是**对称 / 重复的同类部件**（四条腿、一对翅膀、左右轮…），
     用**方位前缀**区分：左/右、前/后、上/下。例：「左前腿」「右后腿」。
   - 若它们应当是**同一个部位的碎片**、或方位无法可靠区分 → label 直接写原名「{part}」。
3. label 要短（≤6 字）；不要给毫无依据的方位（拿不准就沿用原名）。
   同一原部位下，**真正不同的块**才用不同名字。

只返回 JSON：
{{
  "split": true | false,
  "components": [
    {{"id": 1, "label": "左前腿", "reason": "位于 -X、+Z（左前）"}},
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
) -> ClassifyPrompt:
    """生成 seed_count 询问提示。如果传入了 covered/extra 等上下文，
    VLM 就不用再凭图片臆测，能更稳地给出更高的 seed_count。
    """
    import json as _json

    defaults = _media_defaults(media_type)
    if block_points is not None:
        block_size = int(block_points)
    else:
        block_size = int(block_faces)
    guide = RENDER_GUIDE.get(media_type, RENDER_GUIDE["mesh"])

    return ClassifyPrompt(
        system=SEED_SYSTEM_PROMPT,
        user=SEED_USER_TEMPLATE.format(
            render_guide=guide,
            semantic=semantic.strip(),
            max_seeds=max(2, int(max_seeds)),
            covered_parts_json=_json.dumps(
                covered_parts or [], ensure_ascii=False
            ),
            semantic_parts_json=_json.dumps(
                semantic_parts or [], ensure_ascii=False
            ),
            extra_parts_json=_json.dumps(extra_parts or [], ensure_ascii=False),
            block_size=block_size,
            size_unit=defaults["size_unit"],
            large_threshold=defaults["large_threshold"],
        ),
    )
