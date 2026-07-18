"""Natural-language reporting through the OpenAI Responses API."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import textwrap
from typing import Any, Mapping


DEFAULT_MODEL = "gpt-5.6-sol"


@dataclass(frozen=True)
class ReportResult:
    """Report text plus auditable API/cache metadata."""

    text: str
    provider: str
    model: str
    response_id: str | None = None
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    fallback_reason: str | None = None


_ANALYSIS_RUBRIC = textwrap.dedent(
    """
    You are the analysis component of CascadeRank Agent. Write a rigorous graph
    topology report in Simplified Chinese. The report is descriptive, not a
    claim of causal discovery. Treat all node scores as properties of the graph
    representation supplied by the caller. Do not infer social roles, scientific
    importance, document quality, or real-world influence from topology alone.

    Fixed analytical protocol
    -------------------------
    The graph is represented as an undirected, unweighted, simple graph after
    duplicate edges and self-loops are removed. Every external node identifier
    is mapped to one internal integer index, while output tables retain the
    external identifier. A disconnected input remains disconnected. Largest
    connected component size is divided by the original number of nodes, not by
    the number of surviving nodes. Consequently a curve can start below one when
    the original graph is disconnected. Node-removal fractions use floor(f*N),
    and rankings are fixed before the attack; they are not recomputed after each
    deletion. These conventions must be stated or respected when interpreting
    the curves.

    The five baseline scores have distinct meanings. Degree is local adjacency
    count normalized by graph size. Betweenness estimates how often a node lies
    on shortest paths and is the most direct baseline for brokerage, although it
    is model-dependent and can be approximated on large graphs. Closeness
    summarizes inverse path distance under the implementation's disconnected-
    graph convention. Eigenvector centrality rewards connections to already
    central neighbors and can be sensitive to component structure. PageRank is a
    damped random-walk stationary score and should not be described as an attack
    optimum.

    The GNN is a two-layer graph attention regressor. It receives node features
    plus compact structural descriptors. It is trained transductively for no
    more than fifty epochs against a topology-derived proxy target. Therefore a
    high GNN score means that the fitted model reproduces the supplied proxy on
    this graph. It does not establish that the model discovered an independent
    biological, bibliometric, or causal notion of importance. If the GNN attack
    is more destructive than random deletion, the supported conclusion is only
    that its fixed ranking concentrates nodes whose removal fragments this graph
    more rapidly under the stated protocol.

    Structural-hole interpretation
    ------------------------------
    Use the Chinese term “结构洞” explicitly. Call a node a structural-hole
    candidate only when the supplied evidence supports brokerage: for example,
    high betweenness together with low local clustering, multiple disconnected
    neighbor groups after the focal node is omitted, articulation-point status,
    or measurable fragmentation after deletion. Degree alone is insufficient.
    A hub whose neighbors are mutually connected is not automatically a
    structural-hole node. Conversely, a moderate-degree connector can be a
    plausible broker if it joins otherwise weakly connected neighborhoods.

    Distinguish three levels of language. “Observed” is reserved for values in
    the payload: degree, clustering, betweenness, neighbor-group count,
    articulation status, ranks, and component-size changes. “Consistent with”
    is appropriate for the structural-hole interpretation. “Causes” is not
    justified, because the attack is a deterministic intervention on a graph
    abstraction rather than an identified real-world causal experiment.

    Attack-curve interpretation
    ---------------------------
    Compare GNN, PageRank, and deterministic random strategies at the
    pre-specified five-percent and ten-percent checkpoints. Report absolute
    largest-component fractions and their gaps where available. Do not claim a
    visually obvious advantage if the numerical diagnostic is false. A single
    random permutation is a weak baseline: note that a publication-quality
    analysis would use many random seeds with uncertainty bands. Also state that
    rankings are evaluated in-sample on the same graph used to derive the GNN
    proxy; cross-dataset generalization requires held-out graphs or a frozen
    model evaluated without retraining.

    Required report structure
    -------------------------
    1. “结论摘要”: two to four precise conclusions tied to numbers.
    2. “关键节点的拓扑证据”: discuss at least three high-ranked GNN nodes when
       three are provided; distinguish local connectivity from brokerage.
    3. “结构洞节点”: name at least one supplied node identifier and explain why
       its measured neighborhood pattern is consistent with a structural hole.
    4. “渗流攻击比较”: compare all three strategies and explicitly cover the
       five-percent to ten-percent removal interval.
    5. “方法边界”: explain proxy supervision, transductive evaluation, random-
       baseline uncertainty, approximation if indicated, and representation
       dependence. Do not add generic ethics language.

    Evidence discipline
    -------------------
    Never invent an edge, community, node label, score, rank, confidence
    interval, p-value, model accuracy, or runtime. Never say that a node is an
    articulation point unless the payload says so. Never describe a citation
    network direction after it has been symmetrized. When evidence is missing,
    say “当前输出未提供该证据”. Use node identifiers exactly as rendered in the
    payload. Numeric comparisons should preserve the scale supplied by the
    caller and generally use three decimal places.

    Editorial standard
    ------------------
    Prefer short evidence-led paragraphs over promotional prose. Do not praise
    the model. Do not equate attention weights with explanations. Do not claim
    novelty. Do not imply that a faster attack curve validates the GNN target,
    because target construction and evaluation share the same topology. State
    the strongest alternative explanation: the GAT may be learning a smooth
    nonlinear ensemble of centrality-derived descriptors. Separate what the
    current run demonstrates from what would be required for a publishable
    scientific claim.

    Reproducibility checklist to apply silently
    --------------------------------------------
    Check that the node count and edge count are internally plausible; that all
    largest-component fractions lie between zero and one; that GNN-versus-random
    gaps have the correct sign; that structural-hole wording matches the
    supplied local evidence; that any centrality approximation is disclosed;
    and that the final report contains no unsupported causal or generalization
    language. If the payload contains a failed early-attack diagnostic, report
    the failure directly instead of rewriting it as success.

    The stable protocol above is intentionally placed before run-specific
    values. It is reused verbatim for the same graph snapshot so that the OpenAI
    Responses API can cache this exact prefix. Only the payload after the cache
    breakpoint changes between repeated training or attack runs.
    """
).strip()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_stable_graph_prefix(graph_summary: Mapping[str, Any]) -> str:
    """Build the exact graph/protocol prefix placed before the cache marker."""

    graph_json = json.dumps(
        dict(graph_summary),
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
        separators=(",", ":"),
    )
    prefix = (
        f"Fixed graph snapshot (canonical JSON):\n{graph_json}\n\n"
        f"{_ANALYSIS_RUBRIC}"
    )
    if len(prefix.encode("utf-8")) < 7000:
        raise ValueError("stable report prefix is too short for reliable caching")
    return prefix


def _cache_key(stable_prefix: str) -> str:
    digest = sha256(stable_prefix.encode("utf-8")).hexdigest()[:24]
    return f"cascaderank-{digest}"


def _usage_value(response: Any, name: str) -> int:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    value = getattr(details, name, 0)
    return int(value or 0)


def _candidate_text(payload: Mapping[str, Any]) -> str:
    candidate = payload.get("structural_hole_candidate") or {}
    node_id = candidate.get("node_id", "未识别")
    degree = candidate.get("degree", "未提供")
    betweenness = candidate.get("betweenness", "未提供")
    clustering = candidate.get("clustering", "未提供")
    groups = candidate.get("neighbor_groups_without_node", "未提供")
    articulation = candidate.get("is_articulation_point", "未提供")
    return (
        f"节点 {node_id} 是本次输出中的结构洞候选：degree={degree}，"
        f"betweenness={betweenness}，clustering={clustering}，移除该节点后"
        f"其邻居形成 {groups} 个互不连通组，articulation={articulation}。"
        "这一判断表示其局部经纪位置与结构洞一致，不构成现实因果结论。"
    )


def _offline_report(
    graph_summary: Mapping[str, Any],
    payload: Mapping[str, Any],
    reason: str,
) -> str:
    diagnostics = payload.get("early_attack_diagnostics") or {}
    top_nodes = list(payload.get("top_nodes") or [])[:5]
    node_lines = []
    for node in top_nodes:
        node_lines.append(
            "- 节点 {node_id}：GNN rank={gnn_rank}，degree={degree}，"
            "betweenness={betweenness:.6f}，clustering={clustering:.6f}。".format(
                node_id=node.get("node_id", "?"),
                gnn_rank=node.get("gnn_rank", "?"),
                degree=node.get("degree", "?"),
                betweenness=float(node.get("betweenness", 0.0)),
                clustering=float(node.get("clustering", 0.0)),
            )
        )
    if not node_lines:
        node_lines.append("- 当前输出未提供关键节点证据。")

    report = f"""本地回退报告（未调用 OpenAI API）

回退原因：{reason}

结论摘要

数据集 {graph_summary.get('dataset', '?')} 含
{graph_summary.get('nodes', '?')} 个节点和 {graph_summary.get('edges', '?')} 条无向边。
GNN 相对随机策略在 5% 和 10% 删除比例处的最大连通分量差值分别为
{float(diagnostics.get('gap_at_5pct', 0.0)):.3f} 和
{float(diagnostics.get('gap_at_10pct', 0.0)):.3f}。预设的“更快破坏”门禁结果为
{bool(diagnostics.get('gnn_faster_than_random', False))}。

关键节点的拓扑证据

{chr(10).join(node_lines)}

结构洞节点

{_candidate_text(payload)}

渗流攻击比较

曲线使用固定排名依次删除节点，纵轴以原始节点总数归一化。GNN、PageRank
和单次确定性随机排序均被比较；单次随机排序没有不确定性区间，不能替代多随机种子评估。

方法边界

两层 GAT 在同一张图上拟合拓扑代理标签，属于传导式、图内评估。结果不能证明模型
独立发现了关键性，也不能证明跨数据集泛化。最强替代解释是 GAT 学到了中心性与局部
结构描述符的平滑非线性组合。若要形成可发表的科学结论，需要冻结训练流程、使用独立图
测试，并为随机攻击和训练随机性报告重复试验及不确定性。
"""
    return textwrap.dedent(report).strip() + "\n"


def _ensure_structural_hole_section(
    text: str,
    payload: Mapping[str, Any],
) -> str:
    if "结构洞" in text:
        return text
    return f"{text.rstrip()}\n\n结构洞核验\n\n{_candidate_text(payload)}\n"


def generate_explanation(
    graph_summary: Mapping[str, Any],
    analysis_payload: Mapping[str, Any],
    output_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    mode: str = "auto",
    client: Any | None = None,
) -> ReportResult:
    """Generate and persist the report.

    ``auto`` uses OpenAI when a client or ``OPENAI_API_KEY`` is available and
    otherwise writes an explicitly labelled local report. ``openai`` fails
    closed when the API cannot be used. ``offline`` never makes a network call.
    """

    if mode not in {"auto", "openai", "offline"}:
        raise ValueError("mode must be one of: auto, openai, offline")
    if not model.startswith("gpt-5.6"):
        raise ValueError("explicit prompt caching requires a GPT-5.6 model")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stable_prefix = build_stable_graph_prefix(graph_summary)
    dynamic_payload = json.dumps(
        dict(analysis_payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    dynamic_prompt = (
        "Run-specific evidence follows. Apply the fixed protocol and write the "
        "required Simplified Chinese report. Use the exact phrase 结构洞 and "
        "name the supplied structural_hole_candidate.\n\n"
        f"{dynamic_payload}"
    )

    use_api = mode == "openai" or (
        mode == "auto" and (client is not None or bool(os.getenv("OPENAI_API_KEY")))
    )
    if not use_api:
        reason = "OPENAI_API_KEY 未设置" if mode == "auto" else "offline 模式"
        text = _offline_report(graph_summary, analysis_payload, reason)
        output_path.write_text(text, encoding="utf-8")
        return ReportResult(
            text=text,
            provider="local-fallback",
            model=model,
            fallback_reason=reason,
        )
    try:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        response = client.responses.create(
            model=model,
            reasoning={"effort": "low"},
            max_output_tokens=1800,
            prompt_cache_key=_cache_key(stable_prefix),
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": stable_prefix,
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        },
                        {"type": "input_text", "text": dynamic_prompt},
                    ],
                }
            ],
        )
        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned an empty report")
        text = _ensure_structural_hole_section(text, analysis_payload).rstrip() + "\n"
        output_path.write_text(text, encoding="utf-8")
        return ReportResult(
            text=text,
            provider="openai",
            model=str(getattr(response, "model", model) or model),
            response_id=getattr(response, "id", None),
            cached_tokens=_usage_value(response, "cached_tokens"),
            cache_write_tokens=_usage_value(response, "cache_write_tokens"),
        )
    except Exception as exc:
        if mode == "openai":
            raise RuntimeError(f"OpenAI report generation failed: {exc}") from exc
        reason = f"OpenAI 调用失败：{type(exc).__name__}"
        text = _offline_report(graph_summary, analysis_payload, reason)
        output_path.write_text(text, encoding="utf-8")
        return ReportResult(
            text=text,
            provider="local-fallback",
            model=model,
            fallback_reason=reason,
        )
