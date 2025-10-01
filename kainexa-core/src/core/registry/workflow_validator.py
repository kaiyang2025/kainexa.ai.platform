from __future__ import annotations
from typing import Any, Dict, List, Set


class WorkflowValidator:
    @staticmethod
    def validate_node_configs(dsl: Dict[str, Any]) -> bool:
        """
        요구 사항(최소):
        - dsl["nodes"]는 리스트
        - 각 노드는 id, type 보유
        - id는 유일
        - config가 있으면 dict
        유효하면 True, 아니면 ValueError
        """
        nodes = dsl.get("nodes", [])
        if not isinstance(nodes, list):
            raise ValueError("nodes must be a list")

        seen: Set[str] = set()
        for n in nodes:
            if not isinstance(n, dict):
                raise ValueError("node must be dict")
            nid = n.get("id")
            ntype = n.get("type")
            if not nid or not isinstance(nid, str):
                raise ValueError("node.id is required str")
            if nid in seen:
                raise ValueError(f"duplicated node id: {nid}")
            seen.add(nid)
            if not ntype or not isinstance(ntype, str):
                raise ValueError("node.type is required str")
            cfg = n.get("config", {})
            if cfg is not None and not isinstance(cfg, dict):
                raise ValueError("node.config must be dict if provided")
        return True

    @staticmethod
    def validate_edge_conditions(dsl: Dict[str, Any]) -> bool:
        """
        요구 사항(최소):
        - dsl["edges"]는 리스트
        - 각 엣지는 source/target이 있고, 둘 다 존재하는 노드여야 함
        - condition이 있으면 str 또는 dict 허용
        유효하면 True, 아니면 ValueError
        """
        edges = dsl.get("edges", [])
        if not isinstance(edges, list):
            raise ValueError("edges must be a list")

        node_ids = {n.get("id") for n in dsl.get("nodes", []) if isinstance(n, dict)}
        for e in edges:
            if not isinstance(e, dict):
                raise ValueError("edge must be dict")
            src = e.get("source")
            tgt = e.get("target")
            if not src or not tgt:
                raise ValueError("edge.source and edge.target are required")
            if src not in node_ids or tgt not in node_ids:
                raise ValueError("edge endpoints must reference existing nodes")
            cond = e.get("condition", None)
            if cond is not None and not isinstance(cond, (str, dict)):
                raise ValueError("edge.condition must be str or dict if provided")
        return True


__all__ = ["WorkflowValidator"]
