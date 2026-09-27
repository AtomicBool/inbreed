"""main.py —— 调用 ratio_graph 里的数据结构。

流程：
    A = 1/1, B = 0/1
    第 1 轮：A + B          -> C = 1/2
    第 2 轮：C + A, C + B   -> D = 3/4, E = 1/4
    ...每一轮只用上一轮新产生的节点去连前面的节点
    终止条件：出现了 ratio = 1/3 的节点，就打印它的完整合成链。

注意（重要）：按现在的规则，ratio 永远只会是"二进分数"（分母是 2 的幂），
而 1/3 的分母含因子 3，所以这个终止条件永远不会被触发，见文件末尾说明。
"""

from fractions import Fraction

from ratio_graph import RatioGraph, frac_str

# ---- 参数 ----
MAX_ROUNDS = 5         # 图增长很快，先跑 5 轮（约 2280 个节点）
TARGET = Fraction(1, 3)  # 终止条件（按原要求，不改）
SHOW_FIRST_N = 5        # 展示前几个节点的合成链


def main() -> None:
    graph = RatioGraph()

    # 两个初始点：A = 1/1，B = 0/1
    graph.add_founder(Fraction(1, 1))   # 自动命名 A
    graph.add_founder(Fraction(0, 1))   # 自动命名 B

    print("初始点： " + ", ".join(f"{n.name}={frac_str(n.ratio)}" for n in graph))
    print()

    hit = None
    for _ in range(MAX_ROUNDS):
        hit = graph.update(target=TARGET)
        print(
            f"第 {graph.round} 轮： 新增 {len(graph.frontier):>6} 个节点，"
            f"累计 {len(graph):>6} 个"
        )
        if hit is not None:
            break

    print()
    print(f"前 {SHOW_FIRST_N} 个节点：")
    for node in graph.nodes[:SHOW_FIRST_N]:
        if node.is_founder:
            print(f"  {node.name} = {frac_str(node.ratio)}")
        else:
            print(f"  {node.pedigree()}   = {frac_str(node.ratio)}")

    print()
    if hit is not None:
        print(f"命中 ratio = {frac_str(TARGET)}： {hit.pedigree()}")
    else:
        print(f"跑了 {graph.round} 轮，没有出现 ratio = {frac_str(TARGET)} 的节点。")
        closest = min(
            graph.nodes,
            key=lambda n: (abs(n.ratio - TARGET), n.index),  # 并列时取先产生的
        )
        gap = abs(closest.ratio - TARGET)
        print(
            f"最接近 {frac_str(TARGET)} 的是 {closest.name} = "
            f"{frac_str(closest.ratio)}（差 {frac_str(gap)}）"
        )
        print(f"  {closest.pedigree()}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 关于终止条件为什么不会触发
#
# A = 1/1，B = 0/1，两个都是整数；新节点的 ratio 恒为 (x + y) / 2。
# 整数是分母为 2^0 的二进分数，而两个分母为 2^n 的分数取平均后，
# 分母仍然是 2 的幂（约分后也是）。所以整个图里所有 ratio 的分母都是 2 的幂。
# 1/3 约分后分母是 3，含因子 3 —— 永远不可能出现。
#
# 也就是说，只要 ratio 的定义是"两父节点取算术平均"，1/3 就不是可达目标。
# 可选的处理方式，等你确认后再改（现在一行都没动）：
#   1) 把目标换成一个可达的二进分数，比如 3/8、1/2；
#   2) 改 ratio 的合成公式（例如加权、按亲缘系数算），让它能产生分母含 3 的值；
#   3) 只看"最接近 1/3 的节点"或限定轮数，把 1/3 当成一个方向而不是硬目标。
# ---------------------------------------------------------------------------
