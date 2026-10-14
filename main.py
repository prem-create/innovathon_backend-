"""
PhonePe statement API (merging version).

Users can upload many statement CSVs (even overlapping periods). Every
transaction is stored once, keyed by (user_id, transaction_id), so re-uploading
the same period adds nothing and a new period only adds the missing rows.

Run:
    pip install fastapi uvicorn python-multipart
    uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs

Endpoints
---------
POST /api/upload                          one or many CSVs (field name: "files")
GET  /api/home                            homepage cards (received / spend / net / count)
GET  /api/transactions/all|yearly|monthly|weekly
GET  /api/uploads                         upload history
DELETE /api/data                          wipe the user's data (optional, for testing / reset)
"""

import csv
import io
import os
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Iterator, Literal, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DB_PATH = os.getenv("PHONEPE_DB", "phonepe.db")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB per file
MAX_FILES = 20

app = FastAPI(title="PhonePe Statement API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# Schemas
# ==========================================================================
class Transaction(BaseModel):
    date: date                      # 2026-10-13
    time: str                       # 18:07 (24h)
    details: str                    # who the transaction was with
    transaction_id: str
    transaction_type: Literal["CREDIT", "DEBIT"]
    amount: float


class Period(BaseModel):
    label: Optional[str] = None     # as printed in the file
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    first_transaction_date: Optional[date] = None
    last_transaction_date: Optional[date] = None


class Summary(BaseModel):
    count: int
    total_credit: float
    total_debit: float


class FileResult(BaseModel):
    filename: str
    status: Literal["ok", "error"]
    error: Optional[str] = None
    period: Optional[Period] = None
    rows_in_file: int = 0
    added: int = 0                  # new transactions stored
    duplicates: int = 0             # already present, skipped


class UploadResponse(BaseModel):
    files: list[FileResult]
    total_added: int
    total_duplicates: int
    coverage: "Coverage"
    summary: Summary                # of the whole merged dataset


class Coverage(BaseModel):
    from_date: Optional[date] = None
    to_date: Optional[date] = None


UploadResponse.model_rebuild()


class HomeResponse(BaseModel):
    has_data: bool
    money_received: float
    total_spend: float
    net_cash_flow: float            # received - spend
    no_of_transactions: int
    coverage: Coverage              # earliest -> latest transaction we hold
    uploads_count: int
    last_uploaded_at: Optional[datetime] = None


class UploadRecord(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime
    period_label: Optional[str]
    from_date: Optional[date]
    to_date: Optional[date]
    rows_in_file: int
    added: int
    duplicates: int


class AllResponse(BaseModel):
    coverage: Coverage
    summary: Summary
    transactions: list[Transaction]


class Group(BaseModel):
    key: str                        # "2026", "2026-10", "2026-W42"
    start_date: date
    end_date: date
    summary: Summary
    transactions: list[Transaction]


class GroupedResponse(BaseModel):
    coverage: Coverage
    group_by: Literal["year", "month", "week"]
    groups: list[Group]


# ==========================================================================
# CSV parsing
# ==========================================================================
_DURATION_RE = re.compile(r"(\d{1,2}\s+\w{3,4},\s*\d{4})\s*-\s*(\d{1,2}\s+\w{3,4},\s*\d{4})")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _norm_month(raw: str) -> str:
    # PhonePe writes September as "Sept"; strptime's %b only accepts "Sep".
    return re.sub(r"\bSept\b", "Sep", _clean(raw))


def _parse_date(raw: str) -> date:
    return datetime.strptime(_norm_month(raw), "%b %d, %Y").date()


def _parse_duration(text: str) -> tuple[Optional[date], Optional[date]]:
    m = _DURATION_RE.search(text)
    if not m:
        return None, None
    try:
        f, t = (datetime.strptime(_norm_month(g), "%d %b, %Y").date() for g in m.groups())
        return f, t
    except ValueError:
        return None, None


def _parse_time(raw: str) -> str:
    """'06:07 pm' -> '18:07'."""
    raw = raw.strip()
    for fmt in ("%I:%M %p", "%I:%M:%S %p", "%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).strftime("%H:%M")
        except ValueError:
            continue
    return raw


def parse_phonepe_csv(content: bytes) -> tuple[Period, list[Transaction]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    rows = list(csv.reader(io.StringIO(text)))

    # Preamble: duration line + the real header row
    period = Period()
    header_idx = None
    for i, row in enumerate(rows):
        if not row:
            continue
        first = row[0].strip().lower()
        if first == "duration" and len(row) > 1:
            period.label = _clean(row[1])
            period.from_date, period.to_date = _parse_duration(row[1])
        if first == "date" and any("transaction id" in c.lower() for c in row):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            "Could not find the header row (Date, Time, Transaction Details, "
            "Transaction ID, ...). Is this a PhonePe statement CSV?"
        )

    header = [c.strip().lower() for c in rows[header_idx]]
    try:
        idx = {
            "date": header.index("date"),
            "time": header.index("time"),
            "details": header.index("transaction details"),
            "txn_id": header.index("transaction id"),
            "type": header.index("transaction type"),
        }
    except ValueError as e:
        raise ValueError(f"Missing expected column: {e}")
    amount_idx = header.index("amount") if "amount" in header else None

    # Data rows. Footer/disclaimer lines are skipped (no valid date + type).
    transactions: list[Transaction] = []
    for row in rows[header_idx + 1:]:
        if len(row) <= max(idx.values()):
            continue
        try:
            d = _parse_date(row[idx["date"]])
        except ValueError:
            continue
        ttype = row[idx["type"]].strip().upper()
        txn_id = row[idx["txn_id"]].strip()
        if ttype not in ("CREDIT", "DEBIT") or not txn_id:
            continue

        amount = 0.0
        if amount_idx is not None and amount_idx < len(row):
            try:
                amount = float(row[amount_idx].replace(",", "").strip() or 0)
            except ValueError:
                amount = 0.0

        transactions.append(
            Transaction(
                date=d,
                time=_parse_time(row[idx["time"]]),
                details=_clean(row[idx["details"]]),
                transaction_id=txn_id,
                transaction_type=ttype,
                amount=amount,
            )
        )

    transactions.sort(key=lambda t: (t.date, t.time), reverse=True)
    if transactions:
        period.first_transaction_date = transactions[-1].date
        period.last_transaction_date = transactions[0].date
    return period, transactions


# ==========================================================================
# Storage (SQLite). The PRIMARY KEY does the de-duplication.
# ==========================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    filename      TEXT NOT NULL,
    uploaded_at   TEXT NOT NULL,
    period_label  TEXT,
    from_date     TEXT,
    to_date       TEXT,
    rows_in_file  INTEGER NOT NULL,
    added         INTEGER NOT NULL,
    duplicates    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    user_id          TEXT NOT NULL,
    transaction_id   TEXT NOT NULL,
    date             TEXT NOT NULL,      -- YYYY-MM-DD
    time             TEXT NOT NULL,      -- HH:MM
    details          TEXT NOT NULL,
    transaction_type TEXT NOT NULL,
    amount           REAL NOT NULL,
    upload_id        INTEGER REFERENCES uploads(id),
    PRIMARY KEY (user_id, transaction_id)
);
CREATE INDEX IF NOT EXISTS idx_txn_user_date ON transactions (user_id, date);
"""


def get_db() -> Iterator[sqlite3.Connection]:
    # One connection per request, so sharing it between the threadpool and the
    # event loop (async upload handler) is safe.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
    finally:
        conn.close()


def get_user_id(x_user_id: str = Header("default", alias="X-User-Id")) -> str:
    """
    Data is kept per user. Right now the user comes from an X-User-Id header
    (defaults to "default"). Replace this with your real auth (JWT/session).
    """
    return x_user_id.strip() or "default"


def _iso(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def _d(s: Optional[str]) -> Optional[date]:
    return date.fromisoformat(s) if s else None


def _store(conn: sqlite3.Connection, user_id: str, filename: str,
           period: Period, txns: list[Transaction]) -> FileResult:
    """Insert only the transactions we don't already have."""
    cur = conn.execute(
        "INSERT INTO uploads (user_id, filename, uploaded_at, period_label, from_date, to_date,"
        " rows_in_file, added, duplicates) VALUES (?,?,?,?,?,?,?,0,0)",
        (user_id, filename, datetime.now().isoformat(timespec="seconds"),
         period.label, _iso(period.from_date), _iso(period.to_date), len(txns)),
    )
    upload_id = cur.lastrowid

    added = 0
    for t in txns:
        r = conn.execute(
            "INSERT OR IGNORE INTO transactions "
            "(user_id, transaction_id, date, time, details, transaction_type, amount, upload_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (user_id, t.transaction_id, t.date.isoformat(), t.time, t.details,
             t.transaction_type, t.amount, upload_id),
        )
        added += r.rowcount  # 1 if inserted, 0 if it already existed

    duplicates = len(txns) - added
    conn.execute("UPDATE uploads SET added=?, duplicates=? WHERE id=?", (added, duplicates, upload_id))
    return FileResult(filename=filename, status="ok", period=period,
                      rows_in_file=len(txns), added=added, duplicates=duplicates)


def _query(conn: sqlite3.Connection, user_id: str, *, ttype: Optional[str] = None,
           date_from: Optional[date] = None, date_to: Optional[date] = None) -> list[Transaction]:
    sql = "SELECT * FROM transactions WHERE user_id=?"
    args: list = [user_id]
    if ttype:
        sql += " AND transaction_type=?"; args.append(ttype)
    if date_from:
        sql += " AND date>=?"; args.append(date_from.isoformat())
    if date_to:
        sql += " AND date<=?"; args.append(date_to.isoformat())
    sql += " ORDER BY date DESC, time DESC"
    return [
        Transaction(date=r["date"], time=r["time"], details=r["details"],
                    transaction_id=r["transaction_id"], transaction_type=r["transaction_type"],
                    amount=r["amount"])
        for r in conn.execute(sql, args)
    ]


def _coverage(conn: sqlite3.Connection, user_id: str) -> Coverage:
    r = conn.execute("SELECT MIN(date) lo, MAX(date) hi FROM transactions WHERE user_id=?",
                     (user_id,)).fetchone()
    return Coverage(from_date=_d(r["lo"]), to_date=_d(r["hi"]))


def _summary(txns: list[Transaction]) -> Summary:
    return Summary(
        count=len(txns),
        total_credit=round(sum(t.amount for t in txns if t.transaction_type == "CREDIT"), 2),
        total_debit=round(sum(t.amount for t in txns if t.transaction_type == "DEBIT"), 2),
    )


def _group(txns: list[Transaction], by: str) -> list[Group]:
    buckets: dict[str, list[Transaction]] = defaultdict(list)
    bounds: dict[str, tuple[date, date]] = {}
    for t in txns:
        d = t.date
        if by == "year":
            key = str(d.year)
            bounds[key] = (date(d.year, 1, 1), date(d.year, 12, 31))
        elif by == "month":
            key = f"{d.year}-{d.month:02d}"
            nxt = date(d.year + (d.month == 12), (d.month % 12) + 1, 1)
            bounds[key] = (date(d.year, d.month, 1), nxt - timedelta(days=1))
        else:  # ISO week, Mon-Sun
            iso = d.isocalendar()
            key = f"{iso.year}-W{iso.week:02d}"
            monday = d - timedelta(days=d.weekday())
            bounds[key] = (monday, monday + timedelta(days=6))
        buckets[key].append(t)
    return [
        Group(key=k, start_date=bounds[k][0], end_date=bounds[k][1],
              summary=_summary(buckets[k]), transactions=buckets[k])
        for k in sorted(buckets, reverse=True)
    ]


# ==========================================================================
# Endpoints
# ==========================================================================
@app.post("/api/upload", response_model=UploadResponse)
async def upload_statements(
    files: list[UploadFile] = File(..., description="One or more PhonePe CSV statements"),
    conn: sqlite3.Connection = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """
    Upload one or many CSVs. New transactions are merged into the user's data;
    transactions that already exist (same transaction_id) are skipped.
    A bad file is reported in its own result and doesn't fail the others.
    """
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Max {MAX_FILES} files per request.")

    # Read + parse first (async), then write to the DB in one transaction.
    parsed: list[tuple[str, Optional[Period], list[Transaction], Optional[str]]] = []
    for f in files:
        name = f.filename or "unnamed.csv"
        if not name.lower().endswith(".csv"):
            parsed.append((name, None, [], "Only .csv files are accepted."))
            continue
        content = await f.read()
        if not content:
            parsed.append((name, None, [], "File is empty."))
        elif len(content) > MAX_FILE_SIZE:
            parsed.append((name, None, [], "File too large (max 10 MB)."))
        else:
            try:
                period, txns = parse_phonepe_csv(content)
                if not txns:
                    parsed.append((name, period, [], "No transactions found in the file."))
                else:
                    parsed.append((name, period, txns, None))
            except ValueError as e:
                parsed.append((name, None, [], str(e)))

    results: list[FileResult] = []
    with conn:  # commit on success, roll back if anything raises
        for name, period, txns, err in parsed:
            if err:
                results.append(FileResult(filename=name, status="error", error=err, period=period))
            else:
                results.append(_store(conn, user_id, name, period, txns))

    if all(r.status == "error" for r in results):
        raise HTTPException(422, detail=[r.model_dump(mode="json") for r in results])

    return UploadResponse(
        files=results,
        total_added=sum(r.added for r in results),
        total_duplicates=sum(r.duplicates for r in results),
        coverage=_coverage(conn, user_id),
        summary=_summary(_query(conn, user_id)),
    )


@app.get("/api/home", response_model=HomeResponse)
def home(
    date_from: Optional[date] = Query(None, description="Optional, YYYY-MM-DD"),
    date_to: Optional[date] = Query(None, description="Optional, YYYY-MM-DD"),
    conn: sqlite3.Connection = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """
    Homepage cards. `has_data=false` means nothing has been uploaded yet, so
    the frontend should show the upload screen instead.
    """
    r = conn.execute(
        "SELECT COUNT(*) n,"
        " COALESCE(SUM(CASE WHEN transaction_type='CREDIT' THEN amount END),0) received,"
        " COALESCE(SUM(CASE WHEN transaction_type='DEBIT'  THEN amount END),0) spent"
        " FROM transactions WHERE user_id=?"
        " AND (? IS NULL OR date>=?) AND (? IS NULL OR date<=?)",
        (user_id, _iso(date_from), _iso(date_from), _iso(date_to), _iso(date_to)),
    ).fetchone()

    u = conn.execute("SELECT COUNT(*) n, MAX(uploaded_at) last FROM uploads WHERE user_id=?",
                     (user_id,)).fetchone()
    received, spent = round(r["received"], 2), round(r["spent"], 2)
    return HomeResponse(
        has_data=r["n"] > 0,
        money_received=received,
        total_spend=spent,
        net_cash_flow=round(received - spent, 2),
        no_of_transactions=r["n"],
        coverage=_coverage(conn, user_id),
        uploads_count=u["n"],
        last_uploaded_at=u["last"],
    )


@app.get("/api/transactions/all", response_model=AllResponse)
def get_all(
    type: Optional[Literal["CREDIT", "DEBIT"]] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    conn: sqlite3.Connection = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    """Every stored transaction (merged across all uploads)."""
    txns = _query(conn, user_id, ttype=type, date_from=date_from, date_to=date_to)
    return AllResponse(coverage=_coverage(conn, user_id), summary=_summary(txns), transactions=txns)


@app.get("/api/transactions/yearly", response_model=GroupedResponse)
def get_yearly(
    year: Optional[int] = Query(None, description="e.g. 2026"),
    type: Optional[Literal["CREDIT", "DEBIT"]] = None,
    conn: sqlite3.Connection = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    txns = _query(conn, user_id, ttype=type)
    if year:
        txns = [t for t in txns if t.date.year == year]
    return GroupedResponse(coverage=_coverage(conn, user_id), group_by="year", groups=_group(txns, "year"))


@app.get("/api/transactions/monthly", response_model=GroupedResponse)
def get_monthly(
    year: Optional[int] = Query(None),
    month: Optional[int] = Query(None, ge=1, le=12),
    type: Optional[Literal["CREDIT", "DEBIT"]] = None,
    conn: sqlite3.Connection = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    txns = _query(conn, user_id, ttype=type)
    if year:
        txns = [t for t in txns if t.date.year == year]
    if month:
        txns = [t for t in txns if t.date.month == month]
    return GroupedResponse(coverage=_coverage(conn, user_id), group_by="month", groups=_group(txns, "month"))


@app.get("/api/transactions/weekly", response_model=GroupedResponse)
def get_weekly(
    year: Optional[int] = Query(None, description="ISO year"),
    week: Optional[int] = Query(None, ge=1, le=53, description="ISO week (Mon-Sun)"),
    type: Optional[Literal["CREDIT", "DEBIT"]] = None,
    conn: sqlite3.Connection = Depends(get_db),
    user_id: str = Depends(get_user_id),
):
    txns = _query(conn, user_id, ttype=type)
    if year:
        txns = [t for t in txns if t.date.isocalendar().year == year]
    if week:
        txns = [t for t in txns if t.date.isocalendar().week == week]
    return GroupedResponse(coverage=_coverage(conn, user_id), group_by="week", groups=_group(txns, "week"))


@app.get("/api/uploads", response_model=list[UploadRecord])
def upload_history(conn: sqlite3.Connection = Depends(get_db), user_id: str = Depends(get_user_id)):
    """Which files were uploaded, and how many rows each actually contributed."""
    rows = conn.execute("SELECT * FROM uploads WHERE user_id=? ORDER BY id DESC", (user_id,))
    return [UploadRecord(**dict(r)) for r in rows]


@app.delete("/api/data")
def delete_all(conn: sqlite3.Connection = Depends(get_db), user_id: str = Depends(get_user_id)):
    """Remove everything stored for this user."""
    with conn:
        conn.execute("DELETE FROM transactions WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM uploads WHERE user_id=?", (user_id,))
    return {"status": "deleted"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================================================================
# Swagger UI fix: show file pickers for UploadFile fields
# --------------------------------------------------------------------------
# Newer FastAPI/Pydantic emit file fields as
#   {"type": "string", "contentMediaType": "application/octet-stream"}
# which Swagger UI renders as plain text boxes ("Add string item").
# Rewriting that to {"type": "string", "format": "binary"} gives the usual
# "Choose files" button. Only the generated docs change, not the API itself.
# ==========================================================================
def _fix_file_fields(node):
    if isinstance(node, dict):
        if node.get("type") == "string" and node.get("contentMediaType") == "application/octet-stream":
            node.pop("contentMediaType")
            node["format"] = "binary"
        for v in node.values():
            _fix_file_fields(v)
    elif isinstance(node, list):
        for v in node:
            _fix_file_fields(v)


_original_openapi = app.openapi


def _custom_openapi():
    if app.openapi_schema is None:
        schema = _original_openapi()
        _fix_file_fields(schema)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi