"""main.py —— 调用 ratio_graph 里的数据结构。

流程:
    A = 1/1, B = 0/1
    第 1 轮:A + B          -> C = 1/2
    第 2 轮:C + A, C + B   -> D = 3/4, E = 1/4
    ...每一轮只用上一轮新产生的节点去连前面的节点
    终止条件:出现了 ratio = 1/3 的节点,就打印它的完整合成链。

最后:把结果节点(命中的 / 最接近的)的祖先合成图,也就是那棵"乱伦家族树",
连同连线一起画成 PNG —— 见下面的 draw_pedigree()。

注意(重要):按现在的规则,ratio 永远只会是"二进分数"(分母是 2 的幂),
而 1/3 的分母含因子 3,所以这个终止条件永远不会被触发,见文件末尾说明。
"""

from fractions import Fraction
from typing import Optional

from ratio_graph import RatioGraph, column_name, frac_str

# ---- 参数 ----
# round 7 大概是3.37万亿人 能达到33.5% 苏格兰人
MAX_ROUNDS = 7           # 去重后每轮只留独特分数,但 12 轮几乎就无法继续运行了
TARGET = Fraction(1, 3)  # 终止条件(按原要求,不改)
SHOW_FIRST_N = 5         # 展示前几个节点的合成链
DRAW_TREE = True         # 最后把结果节点的家族树画成 PNG
SHOW_TREE = False        # True = 额外弹窗显示(要有图形界面)


class _Slot:
    """画图时的排布项:要么是个真实节点,要么是跨代连线在中间各代占位的"哑节点"。

    哑节点不画出来,只占一个位置,好让跨代的线从方框之间的空隙里绕过去。
    """

    __slots__ = ("x", "y", "d", "real", "node", "up", "down", "seq")

    def __init__(self, d: int, real: bool, node=None, seq: int = 0) -> None:
        self.x = 0.0
        self.y = 0.0
        self.d = d               # 第几代
        self.real = real         # True = 真实节点,False = 哑节点
        self.node = node
        self.up: list = []       # 上面一层接过来的项
        self.down: list = []     # 下面一层接出去的项
        self.seq = seq


def draw_pedigree(
    node, out_path: Optional[str] = None, show: bool = False
) -> Optional[str]:
    """把 node 的祖先合成图("乱伦家族树")画成图片,返回文件路径。

    画的是 node 以及它的全部祖先构成的 DAG:

    * 初始点(A、B)在第 0 代,越往下代数越大,结果节点在最下面一层;
    * 连线就是父子关系,箭头从父指向子(子 = 两个父节点取算术平均);
    * 跨代的连线会在中间各代插一个"哑节点"占位,线从方框之间的空隙绕过去,
      不会压在别的节点上;
    * 方框里的 #n 和字母都只按**这棵树里的先后**连续编(1..N、A..Z),不跳号;
      程序里的全局名有跳号(那是全图的创建顺序),所以在标题下面注了一句
      "图内 X 在程序里的全局名是 Y";
    * 图上两种橙色标出"乱伦"在哪:被连了多次的【复用祖先】(橙色方框),
      以及【近亲连线】(橙色粗线)—— 那种连线两端的辈分是重叠的:
      父节点同时又是另一个父节点的祖先,即父辈跟自己的后代配上了。

    只画祖先,不画全图:去重后第 7 轮全图也就 129 个独特分数。
    没装 matplotlib 时直接跳过,不影响主流程。
    """
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")           # 不弹窗,只存文件
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        from matplotlib.path import Path as MplPath
    except ImportError:
        print("(没装 matplotlib,跳过画图:pip install matplotlib)")
        return None

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    # ---- 1) 收集 node 的全部祖先(含自身)----
    seen, stack = set(), [node]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(cur.parents)
    # 父节点的 index 一定小于子节点,所以按 index 排就是"父在子前"的拓扑序
    order = sorted(seen, key=lambda n: n.index)
    # 图内标签:只按"在这棵树里出现的先后"编号,所以 1..N 和 A..Z 都是连续的,不跳号。
    # 程序里的全局名(有跳号)另外注在标题下面。
    ordinal = {n: i + 1 for i, n in enumerate(order)}
    local = {n: column_name(i) for i, n in enumerate(order)}

    # ---- 2) 算代数(初始点 = 第 0 代)、祖先集合,和"父 -> 子"的反向邻接表 ----
    depth: dict = {}
    anc: dict = {}                                   # 每个节点的全部祖先
    kids: dict = {}
    for n in order:
        depth[n] = 0 if n.is_founder else 1 + max(depth[p] for p in n.parents)
        anc[n] = set()
        for p in n.parents:
            anc[n].add(p)
            anc[n] |= anc[p]                         # order 已按 index 排,父的先算好
            kids.setdefault(p, []).append(n)
    max_depth = max(depth.values())

    # ---- 3) 建"排布项":真实节点各一项;跨代的连线在中间各代插一个哑节点占位,
    #         线就从方框之间的空隙里穿过去,不会压在别的节点上 ----
    slots: list = []
    layers: dict = {}

    def add_slot(d: int, real: bool, n=None) -> _Slot:
        s = _Slot(d, real, n, seq=len(slots))
        slots.append(s)
        layers.setdefault(d, []).append(s)
        return s

    slot_of: dict = {}
    for n in order:
        slot_of[n] = add_slot(depth[n], True, n)
    for row in layers.values():
        row.sort(key=lambda s: s.node.index)         # 同层按产生顺序,输出稳定

    chains: dict = {}                                # 每条边 -> 折线经过的一串项
    for n in order:
        for p in n.parents:
            chain = [slot_of[p]]
            for d in range(depth[p] + 1, depth[n]):  # 中间各代:插哑节点
                chain.append(add_slot(d, False))
            chain.append(slot_of[n])
            for a, b in zip(chain, chain[1:]):       # 串成上下相邻的一串
                a.down.append(b)
                b.up.append(a)
            chains[(p, n)] = chain

    # ---- 4) 排坐标 ----
    # 4a) 先定"同层里谁左谁右":按相邻层的名次做重心排序,交叉会少很多。
    #     顺序一旦定下就不再动,坐标只在这个顺序里挪。
    GAP_REAL, GAP_DUMMY = 2.2, 0.5                   # 框间距 / 两条跨代线之间的间距

    def gap(s: _Slot) -> float:
        return GAP_REAL if s.real else GAP_DUMMY

    row_list = {d: sorted(layers.get(d, ()), key=lambda s: s.seq)
                for d in range(max_depth + 1)}
    rank = {s: i for d, row in row_list.items() for i, s in enumerate(row)}

    def bary(s: _Slot, side: str) -> float:
        """这一项在相邻层里的平均名次;没有邻居就留在原地。"""
        near = getattr(s, side)
        return sum(rank[t] for t in near) / len(near) if near else rank[s]

    for _ in range(8):
        for d in range(1, max_depth + 1):            # 下行:看上面一层的名次
            row_list[d].sort(key=lambda s: (bary(s, "up"), rank[s]))
            for i, s in enumerate(row_list[d]):
                rank[s] = i
        for d in range(max_depth - 1, -1, -1):       # 上行:看下面一层的名次
            row_list[d].sort(key=lambda s: (bary(s, "down"), rank[s]))
            for i, s in enumerate(row_list[d]):
                rank[s] = i

    # 4a-2) 收尾:同层里相邻两项试着一换,交叉数能少就换(Sugiyama 的 transpose 步)
    segs = [pair for chain in chains.values() for pair in zip(chain, chain[1:])]

    def crossings(d: int) -> int:
        """第 d 层到第 d+1 层之间,两条线交叉的对数。"""
        up = [s for s in segs if s[0].d == d]
        cnt = 0
        for i in range(len(up)):
            for j in range(i + 1, len(up)):
                (a1, b1), (a2, b2) = up[i], up[j]
                if (rank[a1] - rank[a2]) * (rank[b1] - rank[b2]) < 0:
                    cnt += 1
        return cnt

    def re_rank(row: list) -> None:
        for i, s in enumerate(row):
            rank[s] = i

    for _ in range(6):
        improved = False
        for d in range(max_depth + 1):
            row = row_list[d]
            for i in range(len(row) - 1):
                before = crossings(d - 1) + crossings(d)
                row[i], row[i + 1] = row[i + 1], row[i]
                re_rank(row)
                after = crossings(d - 1) + crossings(d)
                if after < before:
                    improved = True
                else:                                # 换了没变好,换回去
                    row[i], row[i + 1] = row[i + 1], row[i]
                    re_rank(row)
        if not improved:
            break

    # 4b) 再排具体坐标:重心法 + 同层保中心的最小间距(顺序不再变)
    for d, row in row_list.items():
        for i, s in enumerate(row):
            s.x = float(i) * GAP_REAL
    for _ in range(10):
        for d in range(max_depth - 1, -1, -1):       # 上行:父 = 下面几项的重心
            for s in row_list[d]:
                if s.down:
                    s.x = sum(t.x for t in s.down) / len(s.down)
        for d in range(1, max_depth + 1):            # 下行:子 = 上面几项的重心
            for s in row_list[d]:
                if s.up:
                    s.x = sum(t.x for t in s.up) / len(s.up)
        for d in range(max_depth + 1):               # 同层拉开,再整体挪回重心
            row = row_list[d]
            want = [s.x for s in row]
            for i in range(1, len(row)):
                need = (gap(row[i - 1]) + gap(row[i])) / 2
                want[i] = max(want[i], want[i - 1] + need)
            if row:
                shift = (sum(s.x for s in row) - sum(want)) / len(row)
                for s, v in zip(row, want):
                    s.x = v + shift

    HW, HH = 0.70, 0.45                              # 方框半宽 / 半高(数据单位)
    Y_STEP = 1.5                                     # 每一代之间的高度差
    xs = [s.x for s in slots]
    cx = (min(xs) + max(xs)) / 2 if xs else 0.0
    for s in slots:
        s.x -= cx                                    # 整体居中
        s.y = -s.d * Y_STEP

    # ---- 5) 画 ----
    # 坐标轴铺满整张图 + figsize 按数据范围算,x/y 每单位对应的英寸数相同,
    # 这样不用 set_aspect,方框不会被拉扁,legend/标题放图内也不会撑变形。
    def border_point(s: _Slot, toward: tuple) -> tuple:
        """从方框中心朝 toward 方向射出,求与方框边界的交点:线从这里出发 / 停在这里。"""
        dx, dy = toward[0] - s.x, toward[1] - s.y
        t = 1.0
        if dx:
            t = min(t, HW / abs(dx))
        if dy:
            t = min(t, HH / abs(dy))
        return (s.x + dx * t, s.y + dy * t)

    def rounded_path(pts: list, k: float = 0.3) -> MplPath:
        """折线的每个转角用二次贝塞尔磨圆,免得是硬邦邦的直角。"""
        verts, codes = [pts[0]], [MplPath.MOVETO]
        for i in range(1, len(pts) - 1):
            (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
            d0 = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            d1 = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if not d0 or not d1:
                continue
            kk = min(k, d0 / 2, d1 / 2)
            verts += [(x1 - (x1 - x0) / d0 * kk, y1 - (y1 - y0) / d0 * kk),
                      (x1, y1),
                      (x1 + (x2 - x1) / d1 * kk, y1 + (y2 - y1) / d1 * kk)]
            codes += [MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3]
        verts.append(pts[-1])
        codes.append(MplPath.LINETO)
        return MplPath(verts, codes)

    C_FOUNDER, C_REUSE, C_MID, C_TARGET = "#ffd166", "#ef8354", "#bfd7ea", "#22333b"
    C_INCEST = "#d97706"
    reuse = {n for n in order if len(kids.get(n, ())) > 1}

    xmin, xmax = min(xs), max(xs)
    ymin = min(s.y for s in slots)
    x0, x1 = xmin - HW - 1.45, xmax + HW + 0.7       # 左边留给"第 N 代"
    y0, y1 = ymin - HH - 1.6, HH + 2.3               # 下面留图例、上面留标题
    SCALE = 0.8                                      # 1 个数据单位 = 0.8 英寸

    fig = plt.figure(figsize=((x1 - x0) * SCALE, (y1 - y0) * SCALE))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()

    for (p, child), chain in chains.items():         # 连线:父 -> 子,沿哑节点绕行
        pts = [(s.x, s.y) for s in chain]
        pts[0] = border_point(chain[0], pts[1])      # 从父方框边缘出发
        pts[-1] = border_point(chain[-1], pts[-2])   # 停在子方框边缘
        # 近亲连线:这个父节点同时也是"另一个父节点"的祖先,
        # 也就是父辈跟自己的后代配上了 —— 父母辈分重叠,就是乱伦的由来
        other = [q for q in child.parents if q is not p]
        incest = bool(other) and p in anc[other[0]]
        ax.add_patch(
            FancyArrowPatch(
                path=rounded_path(pts),
                arrowstyle="-|>", mutation_scale=12,
                lw=2.0 if incest else 1.1,
                color=C_INCEST if incest else "#8a8f98",
                zorder=1,
            )
        )

    for n in order:                                 # 节点方框
        s = slot_of[n]
        fill = (
            C_TARGET if n is node
            else C_FOUNDER if n.is_founder
            else C_REUSE if n in reuse
            else C_MID
        )
        ax.add_patch(
            FancyBboxPatch(
                (s.x - HW, s.y - HH), 2 * HW, 2 * HH,
                boxstyle="round,pad=0,rounding_size=0.14",
                facecolor=fill,
                edgecolor="#e5484d" if n is node else "#33415c",
                lw=2.4 if n is node else 1.2,
                zorder=2,
            )
        )
        ax.text(
            s.x, s.y + 0.16, f"#{ordinal[n]} {local[n]}",
            ha="center", va="center", zorder=3, fontsize=9.5, fontweight="bold",
            color="white" if n is node else "#1f2937",
        )
        ax.text(
            s.x, s.y - 0.19, frac_str(n.ratio),
            ha="center", va="center", zorder=3, fontsize=8,
            color="#e5e7eb" if n is node else "#4b5563",
        )

    for d in range(max_depth + 1):                  # 左边标代数
        ax.text(
            xmin - HW - 0.35, -d * Y_STEP, f"第 {d} 代",
            ha="right", va="center", fontsize=8.5, color="#6b7280",
        )

    ax.text(
        0.5, 0.985,
        f"{node.name} = {frac_str(node.ratio)} 的祖先合成图"
        f"({len(order)} 个节点,{sum(len(n.parents) for n in order)} 条连线)",
        transform=ax.transAxes, ha="center", va="top", fontsize=12,
    )
    notes = [
        f"方框里的 #n 和字母都只按这棵树里的先后连续编(1~{len(order)} / "
        f"A~{local[order[-1]]}),不跳号",
        "连线:父 → 子(子的 ratio = 两父取平均);橙色 = 近亲连线"
        "(父节点同时是另一父节点的祖先)",
    ]
    renamed = [(local[n], n.name) for n in order if local[n] != n.name]
    if renamed:
        notes.insert(1, "注意:图内 " + "、".join(a for a, _ in renamed)
                     + " 在程序里的全局名分别是 "
                     + "、".join(b for _, b in renamed))
    for i, line in enumerate(notes):
        ax.text(
            0.5, 0.946 - i * 0.030, line,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color="#6b7280",
        )

    ax.legend(
        handles=[
            Line2D([], [], marker="s", ls="", ms=9, mfc=C_FOUNDER, mec="#33415c", label="初始点"),
            Line2D([], [], marker="s", ls="", ms=9, mfc=C_REUSE, mec="#33415c", label="复用祖先(被连了多次)"),
            Line2D([], [], marker="s", ls="", ms=9, mfc=C_MID, mec="#33415c", label="中间节点"),
            Line2D([], [], marker="s", ls="", ms=9, mfc=C_TARGET, mec="#e5484d", label="结果节点"),
            Line2D([], [], ls="-", lw=2.0, color=C_INCEST, label="近亲连线"),
            Line2D([], [], ls="-", lw=1.1, color="#8a8f98", label="普通连线"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.004), bbox_transform=ax.transAxes,
        ncol=3, frameon=False, fontsize=8.5,
    )

    out_path = out_path or f"pedigree_{node.name}.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    graph = RatioGraph()

    # 两个初始点:A = 1/1,B = 0/1
    graph.add_founder(Fraction(1, 1))   # 自动命名 A
    graph.add_founder(Fraction(0, 1))   # 自动命名 B

    print("初始点: " + ", ".join(f"{n.name}={frac_str(n.ratio)}" for n in graph))
    print()

    hit = None
    for _ in range(MAX_ROUNDS):
        hit = graph.update(target=TARGET)
        print(
            f"第 {graph.round} 轮: 新增 {len(graph.frontier):>6} 个独特分数,"
            f"累计 {len(graph):>6} 个;等同于模拟了 {graph.simulated:,} 人"
        )
        if hit is not None:
            break

    print()
    print(f"前 {SHOW_FIRST_N} 个节点:")
    for node in graph.nodes[:SHOW_FIRST_N]:
        if node.is_founder:
            print(f"  {node.name} = {frac_str(node.ratio)}")
        else:
            print(f"  {node.pedigree()}   = {frac_str(node.ratio)}")

    print()
    if hit is not None:
        print(f"命中 ratio = {frac_str(TARGET)}: {hit.pedigree()}")
    else:
        print(f"跑了 {graph.round} 轮,没有出现 ratio = {frac_str(TARGET)} 的节点。")
        closest = min(
            graph.nodes,
            key=lambda n: (abs(n.ratio - TARGET), n.index),  # 并列时取先产生的
        )
        gap = abs(closest.ratio - TARGET)
        print(
            f"最接近 {frac_str(TARGET)} 的是 {closest.name} = "
            f"{frac_str(closest.ratio)}(差 {frac_str(gap)})"
        )
        print(f"  {closest.pedigree()}")

    # 把结果节点(命中的 / 最接近的)的家族树和连线画出来
    result = hit if hit is not None else closest
    if DRAW_TREE:
        path = draw_pedigree(result, show=SHOW_TREE)
        if path is not None:
            print(f"({result.name} 的家族树已存到 {path})")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 关于终止条件为什么不会触发
#
# A = 1/1,B = 0/1,两个都是整数;新节点的 ratio 恒为 (x + y) / 2。
# 整数是分母为 2^0 的二进分数,而两个分母为 2^n 的分数取平均后,
# 分母仍然是 2 的幂(约分后也是)。所以整个图里所有 ratio 的分母都是 2 的幂。
# 1/3 约分后分母是 3,含因子 3 —— 永远不可能出现。
#
# 也就是说,只要 ratio 的定义是"两父节点取算术平均",1/3 就不是可达目标。
# 可选的处理方式,等你确认后再改(现在一行都没动):
#   1) 把目标换成一个可达的二进分数,比如 3/8、1/2;
#   2) 改 ratio 的合成公式(例如加权、按亲缘系数算),让它能产生分母含 3 的值;
#   3) 只看"最接近 1/3 的节点"或限定轮数,把 1/3 当成一个方向而不是硬目标。
# ---------------------------------------------------------------------------
