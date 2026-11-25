"""Archive active ETF basket files and compare two exact trading dates."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SSE_PAGE = "https://www.sse.com.cn/disclosure/fund/etflist/"
SZSE_PAGE = "https://www.szse.cn/disclosure/fund/etf/index.html"
SSE_URL = "https://query.sse.com.cn/etfDownload/downloadETF2Bulletin.do?fundCode={code}"
SZSE_URL = "https://reportdocs.static.szse.cn/files/text/ETFDown/pcf_{code}_{date}.xml"
FIRST_PCF = "20261123"
MAX_XML_BYTES = 3_000_000
VALIDATION_ERRORS = (ET.ParseError, InvalidOperation, ValueError)
FUNDS = {
    "581000": "摩根大盘成长",
    "581010": "汇添富均衡智选",
    "581020": "华安均衡",
    "581030": "华夏质量",
    "581050": "易方达品质均衡",
    "581060": "永赢景气均衡",
    "581070": "平安红利",
    "581080": "招商价值",
    "581090": "华泰柏瑞价值精选",
    "158900": "南方均衡",
    "158901": "富国价值优选",
    "158902": "大成红利智选",
    "158903": "鹏华价值",
    "158905": "工银红利",
    "158906": "华宝均衡",
    "158907": "国泰鑫汇均衡",
    "158908": "天弘均衡量化",
    "158909": "建信均衡优选",
}
A_SHARE_CODES = {"101": r"6\d{5}", "102": r"[03]\d{5}", "106": r"(?:[48]\d{5}|920\d{3})"}


@dataclass(frozen=True)
class Security:
    name: str
    quantity: Fraction
    cash_flag: str


@dataclass
class Basket:
    unit: int
    security: dict[tuple[str, str], Security]
    metadata: dict


def _iso(value: str) -> str:
    if not re.fullmatch(r"[0-9]{8}", value):
        raise ValueError("交易日格式须为 YYYYMMDD")
    return calendar_date.fromisoformat(value).isoformat()


def _parse(raw: bytes, code: str, date: str, previous_date: str | None) -> Basket:
    exchange = "SH" if code.startswith("58") else "SZ"
    document = ET.fromstring(raw)
    ns = "" if exchange == "SH" else "{http://ts.szse.cn/Fund}"
    tag, code_tag = ("SSEPortfolioCompositionFile", "FundInstrumentID") if exchange == "SH" else ("PCFFile", "SecurityID")
    if document.tag != ns + tag or document.findtext(ns + code_tag) != code:
        raise ValueError("XML 模板或基金代码不符")
    if document.findtext(ns + "TradingDay") != date:
        raise ValueError("清单交易日不符")
    reference_date = document.findtext(ns + "PreTradingDay", "")
    reference_iso = _iso(reference_date)
    if reference_date >= date or (previous_date is not None and reference_date != previous_date):
        raise ValueError("清单对应上一交易日不符")
    unit = Decimal(document.findtext(ns + "CreationRedemptionUnit", ""))
    if not unit.is_finite() or unit <= 0 or unit != unit.to_integral_value():
        raise ValueError("最小申赎单位无效")

    if exchange == "SH":
        records = document.findall("./ComponentList/Component")
        expected = int(document.findtext("RecordNumber", "0"))
        fields = ("UnderlyingSecurityID", "InstrumentID", "InstrumentName", "Quantity", "SubstitutionFlag")
    else:
        records = document.findall(f"./{ns}Components/{ns}Component")
        expected = int(document.findtext(ns + "TotalRecordNum", "0"))
        fields = ("UnderlyingSecurityIDSource", "UnderlyingSecurityID", "UnderlyingSymbol", "ComponentShare", "SubstituteFlag")
    if expected <= 0 or len(records) != expected:
        raise ValueError("成分券记录数不符")

    security = {}
    for row in records:
        venue, stock_code, name, raw_quantity, flag = (row.findtext(ns + field, "").strip() for field in fields)
        if not re.fullmatch(A_SHARE_CODES.get(venue, r"(?!)"), stock_code):
            continue
        key = (venue, stock_code)
        quantity = Decimal(raw_quantity)
        if key in security or not name or not quantity.is_finite() or quantity < 0:
            raise ValueError(f"A 股明细无效 {stock_code}")
        security[key] = Security(name, Fraction(quantity) * 1_000_000 / int(unit), flag)
    if not security:
        raise ValueError("A 股明细为空")
    return Basket(int(unit), security, {
        "pcf_date": _iso(date), "holdings_reference_date": reference_iso,
        "fund_code": code, "fund_name": FUNDS[code], "exchange": exchange,
        "creation_unit_etf_shares": int(unit), "pcf_record_count": expected,
        "a_share_record_count": len(security), "sha256": hashlib.sha256(raw).hexdigest(),
    })


def _write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(contents)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_report(path: Path) -> dict:
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _archive_rows(report: dict) -> list[dict]:
    archive = report.get("archive")
    return [item for item in archive if isinstance(item, dict)] if isinstance(archive, list) else []


def _matching_comparison(report: dict, previous_date: str | None = None) -> dict:
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        return {}
    selected = [item for item in comparisons if isinstance(item, dict)
                and item.get("current_pcf_date") == report.get("report_date")
                and (previous_date is None or item.get("previous_pcf_date") == previous_date)]
    return max(selected, key=lambda item: item.get("previous_pcf_date", ""), default={})


def _complete_report(report: dict, report_stem: Path, date: str, previous_date: str, root: Path) -> bool:
    if report.get("report_date") != date or not report_stem.with_suffix(".md").is_file():
        return False
    comparison = _matching_comparison(report, previous_date)
    if comparison.get("comparable_funds") != len(FUNDS) or not isinstance(comparison.get("changes"), list):
        return False
    source_hashes = {(item.get("pcf_date"), item.get("fund_code")): item.get("sha256") for item in _archive_rows(report)}
    for day in [date, previous_date]:
        for code in FUNDS:
            claimed_hash = source_hashes.get((day, code))
            if not claimed_hash:
                return False
            try:
                archived = (root / "data/active_etf_pcf" / day / f"{code}.xml").read_bytes()
            except OSError:
                return False
            if hashlib.sha256(archived).hexdigest() != claimed_hash:
                return False
    return True


def _load(code: str, date: str, previous_date: str | None, root: Path, provenance: dict) -> tuple[Basket | None, str | None]:
    if date < FIRST_PCF:
        return None, "首批基金尚未上市"
    relative = Path("data/active_etf_pcf") / _iso(date) / f"{code}.xml"
    path = root / relative
    issue = "未找到归档"
    try:
        archived = _parse(path.read_bytes(), code, date, previous_date)
    except (OSError, *VALIDATION_ERRORS) as exc:
        if not isinstance(exc, FileNotFoundError):
            issue = f"归档校验失败（{exc}）"
    else:
        origin = provenance.get((archived.metadata["pcf_date"], code, archived.metadata["sha256"]), {})
        archived.metadata.update({
            "path": relative.as_posix(), "source_type": origin.get("source_type", "archive"),
            "source_url": origin.get("source_url"), "archive_entry": origin.get("archive_entry"),
        })
        return archived, None

    shanghai = code.startswith("58")
    if shanghai and previous_date is None:
        return None, f"{issue}，上交所公开下载提供最新清单"
    url = (SSE_URL if shanghai else SZSE_URL).format(code=code, date=date)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": SSE_PAGE if shanghai else SZSE_PAGE})
    try:
        with urlopen(request, timeout=12) as response:
            raw = response.read(MAX_XML_BYTES + 1)
        if len(raw) > MAX_XML_BYTES:
            raise ValueError("清单体积超限")
        basket = _parse(raw, code, date, previous_date)
        _write_bytes(path, raw)
    except (OSError, *VALIDATION_ERRORS) as exc:
        return None, f"{issue}；交易所清单不可用（{exc}）"
    basket.metadata.update({"path": relative.as_posix(), "source_type": "exchange", "source_url": url, "archive_entry": None})
    return basket, None


def _comparison(current: dict[str, Basket], previous: dict[str, Basket], date: str, previous_date: str) -> dict:
    changes = []
    by_fund = []
    for code in FUNDS:
        if code not in current or code not in previous:
            continue
        before, after = previous[code], current[code]
        tally = {"increase": 0, "decrease": 0}
        for key in sorted(before.security.keys() | after.security.keys()):
            old, new = before.security.get(key), after.security.get(key)
            old_quantity = old.quantity if old else Fraction()
            new_quantity = new.quantity if new else Fraction()
            change = new_quantity - old_quantity
            if not change:
                continue
            direction = "increase" if change > 0 else "decrease"
            tally[direction] += 1
            row = {
                "fund_code": code, "fund_name": FUNDS[code], "security_market": key[0],
                "security_code": key[1], "security_name": (new or old).name,
                "previous_per_million": round(float(old_quantity), 4), "current_per_million": round(float(new_quantity), 4),
                "delta_per_million": round(float(change), 4), "direction": direction,
                "previous_listed": old is not None, "current_listed": new is not None,
                "previous_cash_substitution_flag": old.cash_flag if old else None,
                "current_cash_substitution_flag": new.cash_flag if new else None,
            }
            changes.append(row)
        by_fund.append({
            "fund_code": code, "fund_name": FUNDS[code], "previous_creation_unit": before.unit,
            "current_creation_unit": after.unit, "changes": sum(tally.values()),
            "increases": tally["increase"], "decreases": tally["decrease"],
        })
    same_direction, mixed = _group_changes(changes)
    return {
        "previous_pcf_date": _iso(previous_date), "current_pcf_date": _iso(date),
        "comparable_funds": len(by_fund), "funds_with_quantity_changes": sum(bool(row["changes"]) for row in by_fund),
        "fund_security_changes": len(changes), "by_fund": by_fund, "changes": changes,
        "same_direction": same_direction, "mixed_direction": mixed,
    }


def _group_changes(changes: list[dict]) -> tuple[list[dict], list[dict]]:
    by_stock = defaultdict(list)
    for row in changes:
        key = row.get("security_key") or (row.get("security_market"), row["security_code"])
        by_stock[key].append(row)
    same_direction = []
    mixed = []
    for rows in by_stock.values():
        increases = sum(row["direction"] == "increase" for row in rows)
        decreases = len(rows) - increases
        group = {
            "security_code": rows[0]["security_code"], "security_name": rows[0]["security_name"],
            "funds": len(rows), "increases": increases, "decreases": decreases, "changes": rows,
        }
        if increases and decreases:
            mixed.append(group)
        elif len(rows) >= 2:
            group["direction"] = "increase" if increases else "decrease"
            same_direction.append(group)
    return tuple(sorted(groups, key=lambda item: (-item["funds"], item["security_code"]))
                 for groups in [same_direction, mixed])


def report_section(report: dict) -> str:
    """Render the concise chapter for the daily review and the standalone report."""
    comparison = _matching_comparison(report)
    if not comparison:
        return "\n## 主动 ETF 申赎清单跟踪\n\n没有可核验的相邻交易日清单配对。\n"
    coverage = report.get("coverage")
    if coverage is None:
        archive = _archive_rows(report)
        coverage = {name: len({row.get("fund_code") for row in archive if row.get("pcf_date") == comparison[key]})
                    for name, key in [("current", "current_pcf_date"), ("previous", "previous_pcf_date")]}
        coverage["expected"] = len(FUNDS)
        coverage["unavailable"] = {}
    common = comparison.get("same_direction")
    if common is None:
        common, _ = _group_changes(comparison["changes"])
    paired, expected = comparison["comparable_funds"], coverage["expected"]
    lines = [
        "\n## 主动 ETF 申赎清单跟踪\n",
        f"清单 {report['report_date']} 参考 {comparison['previous_pcf_date']} 收盘组合。"
        f"当日 {coverage['current']}/{expected} 只，上一交易日 {coverage['previous']}/{expected} 只，可比较 {paired}/{expected} 只。",
    ]
    if not paired:
        lines.append("没有可核验的相邻交易日清单配对，无法判断篮子变动。")
    else:
        if paired < expected:
            lines.append("以下结果仅覆盖已配对基金。")
        lines.append(f"{comparison['funds_with_quantity_changes']} 只基金出现数量变化，共 {len(comparison['changes'])} 条基金与证券变化。")
        for direction, label in [("increase", "同向增加"), ("decrease", "同向减少")]:
            groups = [group for group in common if group["direction"] == direction][:5]
            if groups:
                hits = "、".join(f"{group['security_name']}（{group['security_code']}，{group['funds']} 只）" for group in groups)
                lines.append(f"- **{label}** {hits}。")
        if not common:
            lines.append("已配对样本中未出现多只基金同向且没有反向变化的证券。")
        lines.append("口径为每百万份 ETF 对应的 A 股篮子数量，排除申赎现金。数量变化可能受申赎、现金替代、公司行为或取整影响。")
    if coverage["unavailable"]:
        for key, label in [("current", "当日"), ("previous", "上一交易日")]:
            missing = coverage["unavailable"].get(key, {})
            if missing:
                lines.append(f"- {label}缺失或无法核验 {'、'.join(missing)}。")
    return "\n\n".join(lines) + "\n"


def write_daily_pcf_report(date: str, previous_trade_date: str, root: Path = ROOT) -> dict:
    """Write date-validated archive additions and both daily report formats."""
    date_iso, previous_iso = _iso(date), _iso(previous_trade_date)
    if previous_iso >= date_iso:
        raise ValueError("上一交易日必须早于清单日期")
    root = Path(root)
    report_stem = root / "reports" / f"{date}_pcf"
    previous_report = root / "reports" / f"{previous_trade_date}_pcf.json"
    existing = _read_report(report_stem.with_suffix(".json"))
    if _complete_report(existing, report_stem, date_iso, previous_iso, root):
        return existing
    provenance = {
        (item.get("pcf_date"), item.get("fund_code"), item.get("sha256")): item
        for prior in [_read_report(previous_report), existing]
        for item in _archive_rows(prior)
    }
    current, previous = {}, {}
    unavailable = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = {
            (label, code): pool.submit(_load, code, day, reference, root, provenance)
            for label, day, reference in [("current", date, previous_trade_date), ("previous", previous_trade_date, None)]
            for code in FUNDS
        }
        for (label, code), future in pending.items():
            basket, error = future.result()
            if basket is not None:
                (current if label == "current" else previous)[code] = basket
            else:
                unavailable.setdefault(label, {})[code] = error
    report = {
        "report_date": date_iso, "sample": "first_18_mainland_active_stock_etfs",
        "normalization": "a_share_pcf_security_quantity_per_1000000_etf_shares",
        "coverage": {"expected": len(FUNDS), "current": len(current), "previous": len(previous), "unavailable": unavailable},
        "archive": [basket.metadata for entries in [previous, current] for basket in entries.values()],
        "comparisons": [_comparison(current, previous, date, previous_trade_date)],
    }
    md = [f"# 主动 ETF 申赎清单跟踪｜{date_iso}", report_section(report)]
    rows = report["comparisons"][0]["by_fund"]
    if rows:
        md.extend(["### 基金变化覆盖\n", "| 基金 | 代码 | 变化证券数 | 增加 | 减少 |", "|---|---|---:|---:|---:|"])
        for row in rows:
            md.append(f"| {row['fund_name']} | {row['fund_code']} | {row['changes']} | {row['increases']} | {row['decreases']} |")
    md.append(f"\n[逐项差异与来源](./{date}_pcf.json) · [原始 XML](../data/active_etf_pcf/) · [上交所]({SSE_PAGE}) · [深交所]({SZSE_PAGE})\n")
    _write_bytes(report_stem.with_suffix(".json"), (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    _write_bytes(report_stem.with_suffix(".md"), "\n".join(md).encode("utf-8"))
    return report
