# optimization_utils.py
from typing import Dict
from config import Config

def create_compact_item(code: str, name: str, group_type: str = "first") -> Dict:
    """Create compact JSON item to minimize tokens"""
    if Config.abbreviate_keys:
        if group_type == "first":
            return {"c": code, "n": name}  # c=code, n=name
        else:
            return {"c": code, "n": name}
    else:
        if group_type == "first":
            return {"code": code, "name": name}
        else:
            return {"code": code, "name": name}

def expand_compact_result(item: Dict, group_type: str = "first") -> Dict:
    """Expand compact result back to full format"""
    if Config.abbreviate_keys:
        if group_type == "mapping":
            return {
                "First Group Code": item.get("fc", item.get("firstCode", "")),
                "First Group Name": item.get("fn", item.get("firstName", "")),
                "Second Group Code": item.get("sc", item.get("secondCode", None)),
                "Second Group Name": item.get("sn", item.get("secondName", None)),
                "similarity score": item.get("s", item.get("score", 0)),
                "reason for similarity score": item.get("r", item.get("reason", ""))
            }
    return item