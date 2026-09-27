"""ratio_graph.py —— 比例合成图的数据结构(有向图)。

结构
----
* 每个 Node 存一个 ratio,用 fractions.Fraction 精确存储(不丢精度,不会退化成浮点)。
* 连接是有向的:子节点保存指向两个父节点的边(parents),父节点保存反向边(children)。
* 每轮 update() 只处理"上一轮新产生的节点"(frontier),让它们去连接之前已有的节点
  (settled),每对生成一个新节点:

        child.ratio = (a.ratio + b.ratio) / 2

  这样任意两节点最终都能通过某个后代连起来,而每轮的工作量只跟新节点数量有关
  (也就是你说的:不用重复考虑已经连过的组合)。

去重
----
  同一个 ratio 只保留首次出现的那一个节点,重复的丢弃(见 RatioGraph.by_ratio)。
  新分数只由父节点的 ratio 值决定,跟具体是哪个节点实例无关,所以丢重复分数
  不影响后续能合成出的分数集合,却能把节点数从"随轮数爆炸"压到"每轮 2 的幂"量级
  (第 r 轮累计 2^r + 1 个独特分数)。

节点命名
--------
按 A, B, ..., Z, AA, AB, ...(Excel 列名规则)自动递增。

配对顺序(pair_mode)
---------------------
规则:**每个新节点连接排在它前面的所有节点**——包括同一轮里比它先生出来的兄弟节点。
A-B 之所以配对,就是这个规则本身的一个实例(B 排在 A 后面),D-E 同理。

默认 "sequential"(按上面这条规则):

      第 1 轮:A + B           -> C
      第 2 轮:C x {A, B}      -> D, E          (D 先,E 后)
      第 3 轮:D x {A, B, C}   -> 3 个
               E x {A, B, C, D} -> 4 个          (E 连上了同轮的 D)
              共 7 个

另一种读法 "batch"(一轮一视同仁:本轮新节点只连【本轮开始前】已有的节点,
同轮兄弟之间不连)保留作对照,第 3 轮会得到 6 个而不是 7 个。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple, Union

Number = Union[int, str, Fraction]


def column_name(index: int) -> str:
    """0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, 27 -> AB, ..."""
    if index < 0:
        raise ValueError("index 必须 >= 0")
    name = ""
    index += 1
    while index > 0:
        index, rem = divmod(index - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def frac_str(value: Fraction) -> str:
    """始终按 n/d 显示,整数也写成 1/1、0/1,不缩写成 1、0。"""
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


class Node:
    """图里的一个节点:名字 + 精确的 ratio + 指向父节点的有向边。"""

    __slots__ = ("index", "name", "ratio", "parents", "children")

    def __init__(
        self,
        index: int,
        name: str,
        ratio: Fraction,
        parents: Sequence["Node"] = (),
    ) -> None:
        self.index = index            # 创建序号,同时决定默认名字
        self.name = name
        self.ratio = ratio
        self.parents: Tuple["Node", ...] = tuple(parents)  # 0 个(初始点)或 2 个
        self.children: List["Node"] = []                   # 反向边

    # ---------- 查询 ----------

    @property
    def is_founder(self) -> bool:
        """初始节点(没有父节点)。"""
        return not self.parents

    def expression(self) -> str:
        """完整的合成链,只看左边。

        A -> 'A'      C -> '(A+B)'      D -> '((A+B)+A)'
        """
        if not self.parents:
            return self.name
        return "(" + "+".join(p.expression() for p in self.parents) + ")"

    def pedigree(self) -> str:
        """合成链 -> 结果,例如 '(A+B) -> C'。"""
        return f"{self.expression()} -> {self.name}"

    def __repr__(self) -> str:
        if self.is_founder:
            return f"<Node {self.name}={self.ratio}>"
        return f"<Node {self.name}={self.ratio} {self.pedigree()}>"


class RatioGraph:
    """按轮次生长的比例合成图。

    只暴露三件事:add_founder() 建初始点、update() 推一轮、Node 自己负责表达合成链。
    循环怎么写、跑几轮、目标是几,全部交给调用方(main.py)。
    """

    def __init__(self, pair_mode: str = "sequential") -> None:
        if pair_mode not in ("batch", "sequential"):
            raise ValueError("pair_mode 只能是 'batch' 或 'sequential'")
        self.pair_mode = pair_mode
        self.nodes: List[Node] = []
        self.by_ratio: Dict[Fraction, Node] = {}  # ratio -> 首个该分数的节点(去重用)
        self.frontier: List[Node] = []   # 上一轮新产生的节点
        self.settled: List[Node] = []    # 本轮开始前就已存在的节点
        self.round = 0
        self.simulated = 0               # 去重前本应生成的节点数(等同模拟了多少人)
        self._orig_frontier = 0          # 去重前每轮 frontier 的规模(只记数量)
        self._orig_settled = 0           # 去重前每轮 settled 的规模

    # ---------- 构建 ----------

    def add_founder(self, ratio: Number, name: Optional[str] = None) -> Node:
        """加一个初始节点(没有父节点)。名字不传就按 A、B、C... 自动取。"""
        node = Node(
            index=len(self.nodes),
            name=name if name is not None else column_name(len(self.nodes)),
            ratio=Fraction(ratio),
        )
        self.nodes.append(node)
        self.by_ratio[Fraction(ratio)] = node
        self.simulated += 1
        return node

    def _link(self, a: Node, b: Node) -> Optional[Node]:
        """连接 a、b:ratio 取两父节点的算术平均。

        若这个 ratio 之前已经出现过,就丢弃(返回 None),不另建节点——未来能
        合成出哪些新分数只看父节点的 ratio 值,跟具体是哪个节点无关,所以重复
        分数对可达性没有任何贡献,只白白占内存。
        """
        ratio = (a.ratio + b.ratio) / 2
        if ratio in self.by_ratio:
            return None
        node = Node(
            index=len(self.nodes),
            name=column_name(len(self.nodes)),
            ratio=ratio,
            parents=(a, b),
        )
        self.nodes.append(node)
        self.by_ratio[ratio] = node
        a.children.append(node)
        b.children.append(node)
        return node

    # ---------- 一轮更新 ----------

    def update(self, target: Optional[Fraction] = None) -> Optional[Node]:
        """推进一步。

        第 1 轮:把两个初始节点连起来(A + B)。
        之后每轮:只把上一轮新产生的节点,按先后顺序连向排在它前面的所有节点
        (同轮里先生出来的那些也算),所以 D-E 之间也会有一条连接。

        去重:新节点的 ratio 若之前已经出现过,就丢弃(不另建节点)。因为新分数
        只看父节点的 ratio 值,跟具体是哪个节点实例无关,所以丢重复分数不影响
        后续能合成出的分数集合,却能把节点数压到"每轮 2 的幂"这个量级。

        参数 target:本轮生成的新节点里,如果有 ratio 恰好等于 target 的,
        就地返回它(该轮其余节点照常生成,不提前中断)。
        返回值:命中的节点,或 None。同一轮有多个命中时返回最先产生的那个。
        """
        self.round += 1

        # 并行推演"去重前"的规模:只记数量、不真建节点,用来报"等同模拟了多少人"
        if self.round == 1:
            orig_created = 1                       # A + B
        else:
            orig_created = self._orig_frontier * self._orig_settled
            if self.pair_mode == "sequential":
                orig_created += self._orig_frontier * (self._orig_frontier - 1) // 2
        self.simulated += orig_created

        if self.round == 1:
            if len(self.nodes) < 2:
                raise ValueError("第一轮需要至少两个初始节点")
            created = []
            node = self._link(self.nodes[0], self.nodes[1])
            if node is not None:
                created.append(node)
            self.settled = list(self.nodes[:2])
        elif self.pair_mode == "batch":
            created = []
            for new in self.frontier:
                for old in self.settled:
                    node = self._link(old, new)
                    if node is not None:
                        created.append(node)
            self.settled = self.settled + self.frontier
        else:  # sequential
            older = list(self.settled)
            created = []
            for new in self.frontier:
                for old in older:
                    node = self._link(old, new)
                    if node is not None:
                        created.append(node)
                older.append(new)          # 同一轮里先生出来的也算"之前的"
            self.settled = self.settled + self.frontier

        self.frontier = created

        if self.round == 1:
            self._orig_settled = 2
            self._orig_frontier = 1
        else:
            self._orig_settled = self._orig_settled + self._orig_frontier
            self._orig_frontier = orig_created

        if target is not None:
            for node in created:
                if node.ratio == target:
                    return node
        return None

    # ---------- 顺手一点的容器行为 ----------

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)

    def get(self, name: str) -> Node:
        """按名字取节点。"""
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    def __repr__(self) -> str:
        return (
            f"<RatioGraph round={self.round} nodes={len(self.nodes)} "
            f"frontier={len(self.frontier)} mode={self.pair_mode}>"
        )
