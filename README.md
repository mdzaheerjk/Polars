<div align="center">

# 🐻‍❄️ Polars for AI/ML Engineers

### From your first DataFrame to production-grade feature pipelines

![Polars](https://img.shields.io/badge/Polars-1.x-CD792C?style=for-the-badge&logo=polars&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Rust](https://img.shields.io/badge/Powered_by-Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
![Arrow](https://img.shields.io/badge/Apache-Arrow-D22128?style=for-the-badge&logo=apache&logoColor=white)
![Level](https://img.shields.io/badge/Level-Beginner_→_Advanced-success?style=for-the-badge)
![AI/ML](https://img.shields.io/badge/Focus-AI%2FML_Engineering-blueviolet?style=for-the-badge)

</div>

---

## 📑 Table of Contents

| # | Section | Level |
|---|---------|-------|
| 1 | [Why Polars?](#1--why-polars) | 🟢 |
| 2 | [Installation & Setup](#2--installation--setup) | 🟢 |
| 3 | [Core Data Structures & Dtypes](#3--core-data-structures--dtypes) | 🟢 |
| 4 | [Reading & Writing Data](#4--reading--writing-data) | 🟢 |
| 5 | [Inspecting Data](#5--inspecting-data) | 🟢 |
| 6 | [Expressions: The Heart of Polars](#6--expressions-the-heart-of-polars) | 🟢 |
| 7 | [Select, Filter, With_Columns, Sort](#7--select-filter-with_columns-sort) | 🟢 |
| 8 | [Column Selectors](#8--column-selectors) | 🟡 |
| 9 | [Null, NaN & Missing Data](#9--null-nan--missing-data) | 🟡 |
| 10 | [Casting & Type Handling](#10--casting--type-handling) | 🟡 |
| 11 | [String Operations](#11--string-operations) | 🟡 |
| 12 | [Date & Time Operations](#12--date--time-operations) | 🟡 |
| 13 | [Conditional Logic: when/then/otherwise](#13--conditional-logic) | 🟡 |
| 14 | [GroupBy & Aggregations](#14--groupby--aggregations) | 🟡 |
| 15 | [Window Functions with `over`](#15--window-functions-with-over) | 🟡 |
| 16 | [Joins, Concatenation, Reshaping](#16--joins-concatenation-reshaping) | 🟡 |
| 17 | [List, Array & Struct Types](#17--list-array--struct-types) | 🔴 |
| 18 | [Rolling, Dynamic & Time-Series](#18--rolling-dynamic--time-series) | 🔴 |
| 19 | [Lazy API & Query Optimization](#19--lazy-api--query-optimization) | 🔴 |
| 20 | [Streaming & Larger-Than-Memory Data](#20--streaming--larger-than-memory-data) | 🔴 |
| 21 | [Polars for ML: Feature Engineering](#21--polars-for-ml-feature-engineering) | 🔴 |
| 22 | [Polars + scikit-learn / PyTorch / TensorFlow / HF](#22--polars--ml-frameworks) | 🔴 |
| 23 | [NLP & Text Preprocessing Pipelines](#23--nlp--text-preprocessing-pipelines) | 🔴 |
| 24 | [Custom Functions, UDFs & Plugins](#24--custom-functions-udfs--plugins) | 🔴 |
| 25 | [SQL Interface](#25--sql-interface) | 🟡 |
| 26 | [Performance Tuning Playbook](#26--performance-tuning-playbook) | 🔴 |
| 27 | [Pandas → Polars Cheat Sheet](#27--pandas--polars-cheat-sheet) | 🟢 |
| 28 | [Common Pitfalls](#28--common-pitfalls) | 🟡 |
| 29 | [Practice Projects](#29--practice-projects) | 🟡 |
| 30 | [Interview Questions & Answers](#30--interview-questions--answers) | 🔴 |
| 31 | [Resources](#31--resources) | 🟢 |

---

## 1. 🟢 Why Polars?

**Polars** is a blazingly fast DataFrame library written in **Rust**, built on the **Apache Arrow** columnar memory format, with Python, Rust, Node.js and R bindings.

### Core design ideas

| Idea | What it means for you |
|------|-----------------------|
| **Columnar memory (Arrow)** | Operations on a column are cache-friendly and vectorized (SIMD). Zero-copy interop with other Arrow tools. |
| **Multi-threaded by default** | Uses all CPU cores automatically. No `multiprocessing`, no Dask for most workloads. |
| **Expression system** | You describe *what* to compute; Polars decides *how* and parallelizes independent expressions. |
| **Lazy evaluation + optimizer** | Build a query plan, Polars rewrites it (predicate/projection pushdown, CSE) before running. |
| **Streaming engine** | Process datasets larger than RAM in batches. |
| **No index** | Row order is the only identity. Fewer footguns than pandas' index alignment. |
| **Strict typing** | Every column has one dtype. Nulls are first-class (validity bitmap), not `NaN` hacks. |

### Polars vs Pandas vs Spark

| Feature | Pandas | Polars | PySpark |
|---------|--------|--------|---------|
| Language core | C / NumPy | Rust / Arrow | JVM |
| Parallelism | Single-thread (mostly) | Multi-thread auto | Distributed cluster |
| Lazy execution | ❌ | ✅ | ✅ |
| Larger-than-RAM | ❌ | ✅ (streaming) | ✅ |
| Index | Yes (complex) | No | No |
| Null handling | NaN / None / NA mix | Unified `null` | null |
| Typical speedup vs pandas | 1× | **5–30×** | n/a (cluster) |
| Best for | Small/medium, ecosystem | Single-machine, up to ~100s of GB | Multi-node TB+ |

> **Rule of thumb for AI/ML:** Use Polars for data loading, cleaning, feature engineering and dataset prep; hand off to NumPy/PyTorch/sklearn at the model boundary.

### Mental model

```
 Your code (expressions)
        │
        ▼
 Logical plan  ──►  Optimizer  ──►  Physical plan  ──►  Parallel execution (Rust)
 (lazy only)      (pushdowns,       (operators)          (all cores, SIMD)
                   CSE, etc.)
```

---

## 2. 🟢 Installation & Setup

```bash
pip install polars                 # core
pip install "polars[all]"          # all optional deps
pip install polars-lts-cpu         # for older CPUs without AVX2
```

Common optional extras:

```bash
pip install "polars[pyarrow,numpy,pandas,fsspec,sqlalchemy,connectorx,deltalake,excel]"
```

```python
import polars as pl
import polars.selectors as cs   # column selectors (section 8)

print(pl.__version__)
```

### Display configuration

```python
pl.Config.set_tbl_rows(20)          # rows shown
pl.Config.set_tbl_cols(15)          # columns shown
pl.Config.set_fmt_str_lengths(80)   # string truncation
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")  # markdown-style tables
pl.Config.set_tbl_hide_dataframe_shape(True)

# Temporarily scope config
with pl.Config(tbl_rows=50):
    print(df)
```

---

## 3. 🟢 Core Data Structures & Dtypes

### Series — one column

```python
s = pl.Series("scores", [0.91, 0.85, 0.77, None])
s.dtype       # Float64
s.null_count()  # 1
s.mean()      # ignores nulls
```

### DataFrame — a collection of Series

```python
df = pl.DataFrame({
    "id":     [1, 2, 3, 4],
    "name":   ["Asha", "Ravi", "Meera", "Karan"],
    "score":  [91.5, 85.0, None, 77.2],
    "passed": [True, True, False, True],
})
print(df)
```

```
shape: (4, 4)
┌─────┬───────┬───────┬────────┐
│ id  ┆ name  ┆ score ┆ passed │
│ --- ┆ ---   ┆ ---   ┆ ---    │
│ i64 ┆ str   ┆ f64   ┆ bool   │
╞═════╪═══════╪═══════╪════════╡
│ 1   ┆ Asha  ┆ 91.5  ┆ true   │
│ ... ┆ ...   ┆ ...   ┆ ...    │
└─────┴───────┴───────┴────────┘
```

### Other ways to create a DataFrame

```python
pl.DataFrame([[1, "a"], [2, "b"]], schema=["n", "c"], orient="row")
pl.DataFrame({"x": [1, 2]}, schema={"x": pl.Int32})
pl.from_pandas(pdf)
pl.from_numpy(arr, schema=["f1", "f2", "f3"])
pl.from_dicts([{"a": 1}, {"a": 2}])
pl.from_arrow(arrow_table)
```

### LazyFrame — a deferred query

```python
lf = df.lazy()                 # from eager
lf = pl.scan_csv("data.csv")   # directly from file (never loads everything)
df = lf.collect()              # execute
```

### Dtype reference

<details>
<summary><b>📦 Full dtype table (click to expand)</b></summary>

| Category | Dtypes | Notes |
|----------|--------|-------|
| Signed int | `Int8, Int16, Int32, Int64, Int128` | Default int = `Int64` |
| Unsigned int | `UInt8, UInt16, UInt32, UInt64` | Great for ids / counts |
| Float | `Float32, Float64` | `Float32` is standard for ML; halves memory |
| Decimal | `Decimal(precision, scale)` | Exact money math |
| Boolean | `Boolean` | |
| String | `String` (alias `Utf8`) | UTF-8, Arrow-backed |
| Binary | `Binary` | |
| Categorical | `Categorical`, `Enum` | `Enum` = fixed known categories, faster |
| Date/Time | `Date, Datetime(unit, tz), Time, Duration` | `unit` = `"ns" "us" "ms"` |
| Nested | `List(inner)`, `Array(inner, width)`, `Struct({...})` | See section 17 |
| Special | `Object`, `Null`, `Unknown` | `Object` is slow, avoid |

</details>

### Schema, shape, columns

```python
df.schema        # Schema({'id': Int64, ...})
df.dtypes        # [Int64, String, Float64, Boolean]
df.columns       # ['id', 'name', 'score', 'passed']
df.shape         # (4, 4)
df.height, df.width
df.estimated_size("mb")
```

### Memory-saving tip for ML

```python
df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))     # halve float memory
df = df.with_columns(pl.col("label").cast(pl.UInt8))
df = df.with_columns(pl.col("city").cast(pl.Categorical))
```

---

## 4. 🟢 Reading & Writing Data

### Eager readers

```python
df = pl.read_csv("data.csv")
df = pl.read_csv("data.csv", separator=";", has_header=True, null_values=["NA", ""],
                 schema_overrides={"zip": pl.String}, try_parse_dates=True,
                 infer_schema_length=10_000, n_rows=1000)
df = pl.read_parquet("data.parquet")
df = pl.read_ndjson("logs.ndjson")
df = pl.read_json("data.json")
df = pl.read_excel("book.xlsx", sheet_name="Sheet1")   # needs fastexcel
df = pl.read_ipc("data.arrow")                         # Arrow IPC / Feather v2
df = pl.read_database("SELECT * FROM t", connection=conn)
df = pl.read_database_uri("SELECT * FROM t", "postgresql://user:pw@host/db")
```

### Lazy scanners (preferred for anything big)

```python
lf = pl.scan_csv("big.csv")
lf = pl.scan_parquet("data/*.parquet")                 # glob supported
lf = pl.scan_parquet("data/", hive_partitioning=True)  # partitioned datasets
lf = pl.scan_ndjson("logs.ndjson")
lf = pl.scan_ipc("data.arrow")
lf = pl.scan_delta("path/to/delta_table")
lf = pl.scan_parquet("s3://bucket/path/*.parquet", storage_options={"aws_region": "ap-south-1"})
```

### Writers

```python
df.write_csv("out.csv")
df.write_parquet("out.parquet", compression="zstd", compression_level=3)
df.write_ipc("out.arrow")
df.write_ndjson("out.ndjson")
df.write_json("out.json")
df.write_delta("delta_dir")
df.write_database("table", connection="sqlite:///my.db", if_table_exists="replace")

# Lazy sinks: write without ever materializing the full result
lf.sink_parquet("out.parquet")
lf.sink_csv("out.csv")
lf.sink_ipc("out.arrow")
```

### Which format should an ML engineer use?

| Format | Speed | Size | Schema preserved | Verdict |
|--------|-------|------|------------------|---------|
| CSV | 🐢 | Large | ❌ | Only for interchange |
| **Parquet** | 🚀 | Small | ✅ | **Default for datasets** |
| Arrow IPC / Feather | 🚀🚀 | Medium | ✅ | Fast scratch / memory-map |
| NDJSON | 🐢 | Large | ~ | Logs, LLM datasets |
| Delta | 🚀 | Small | ✅ | Versioned tables |

> **Always convert CSV → Parquet once**, then work from Parquet. It is the single highest-leverage habit.

```python
pl.scan_csv("raw.csv").sink_parquet("raw.parquet")
```

---

## 5. 🟢 Inspecting Data

```python
df.head(5)        # first 5
df.tail(5)
df.sample(n=5, seed=42)
df.glimpse()      # compact transposed view (great for wide data)
df.describe()     # count, null_count, mean, std, min, quartiles, max
df.schema
df.null_count()   # nulls per column
df.is_empty()
df.n_unique()     # distinct rows
df["name"]        # -> Series
df.get_column("name")
df.row(0)         # tuple
df.row(0, named=True)  # dict
df.rows()         # list of tuples (slow for big data)
df.to_dicts()
```

### Quick EDA recipe

```python
def quick_eda(df: pl.DataFrame):
    print("shape:", df.shape)
    print(df.glimpse(return_as_string=True))
    nulls = df.null_count().transpose(include_header=True, column_names=["nulls"])
    print(nulls.sort("nulls", descending=True).head(10))
    print(df.describe())
    cat_cols = [c for c, t in df.schema.items() if t in (pl.String, pl.Categorical)]
    for c in cat_cols[:5]:
        print(df[c].value_counts(sort=True).head(5))
```

---

## 6. 🟢 Expressions: The Heart of Polars

An **expression** is a *recipe* for computing a column. It does nothing until placed in a **context** (`select`, `with_columns`, `filter`, `group_by().agg`, ...).

```python
pl.col("score") * 2                       # expression (not data!)
pl.col("score").mean()                    # expression
(pl.col("score") - pl.col("score").mean()) / pl.col("score").std()   # composed
```

### Contexts

| Context | Purpose | Output rows |
|---------|---------|-------------|
| `select` | Pick / compute columns | Same or aggregated |
| `with_columns` | Add / replace columns, keep the rest | Same |
| `filter` | Keep rows matching a boolean expression | Fewer |
| `group_by().agg` | Aggregate per group | One per group |
| `sort` | Order rows | Same |

### Why expressions matter

```python
# Polars runs these independent expressions IN PARALLEL
df.select(
    pl.col("score").mean().alias("mean"),
    pl.col("score").std().alias("std"),
    pl.col("score").max().alias("max"),
    pl.col("name").n_unique().alias("unique_names"),
)
```

### Expression building blocks

```python
pl.col("a")                 # one column
pl.col("a", "b")            # several
pl.col("^feat_.*$")         # regex on names
pl.col(pl.Float64)          # all columns of a dtype
pl.col("*")  /  pl.all()    # every column
pl.exclude("id", "label")   # all except
pl.lit(5)                   # literal
pl.len()                    # row count (replaces pl.count())
pl.struct("a", "b")         # pack columns
pl.concat_str(["a", "b"], separator=" ")
pl.int_range(pl.len())      # 0..n-1 row index
pl.when(...).then(...).otherwise(...)
```

### Renaming outputs

```python
pl.col("score").mean().alias("avg_score")
pl.col("a", "b").name.suffix("_raw")
pl.col("a", "b").name.prefix("x_")
pl.all().name.to_uppercase()
pl.all().name.map(lambda n: n.lower().replace(" ", "_"))
```

---

## 7. 🟢 Select, Filter, With_Columns, Sort

### select

```python
df.select("id", "score")
df.select(pl.col("score") * 100)
df.select(pl.all().exclude("id"))
df.select(pl.col("score").alias("s"), (pl.col("score") > 80).alias("high"))
```

### with_columns (the workhorse)

```python
df = df.with_columns(
    (pl.col("score") / 100).alias("score_norm"),
    pl.col("name").str.to_uppercase().alias("name_up"),
    pl.lit("v1").alias("version"),
    pl.int_range(pl.len()).alias("row_id"),
)
```

> **Important:** expressions inside one `with_columns` run in **parallel** and **cannot see each other's new columns**. Chain another `.with_columns` if you need that.

```python
df.with_columns(a=pl.col("x") * 2).with_columns(b=pl.col("a") + 1)   # ✅
df.with_columns(a=pl.col("x") * 2, b=pl.col("a") + 1)                # ❌ 'a' not yet defined
```

### filter

```python
df.filter(pl.col("score") > 80)
df.filter((pl.col("score") > 80) & (pl.col("passed")))
df.filter(pl.col("name").is_in(["Asha", "Ravi"]))
df.filter(pl.col("score").is_between(70, 90))
df.filter(pl.col("score").is_not_null())
df.filter(~pl.col("name").str.contains("^A"))
df.filter(pl.col("score") > 80, pl.col("passed"))   # comma = AND
```

### sort, unique, drop, rename

```python
df.sort("score", descending=True, nulls_last=True)
df.sort(["passed", "score"], descending=[True, False])
df.top_k(3, by="score");  df.bottom_k(3, by="score")
df.unique(subset=["name"], keep="first", maintain_order=True)
df.drop("id");  df.drop(cs.string())
df.rename({"score": "marks"})
df.slice(10, 5);  df.limit(5);  df[10:15]
```

---

## 8. 🟡 Column Selectors

`polars.selectors` lets you select columns by **type, name pattern, or position** — hugely helpful for wide ML tables.

```python
import polars.selectors as cs

df.select(cs.numeric())
df.select(cs.string())
df.select(cs.float(), cs.integer())
df.select(cs.starts_with("feat_"))
df.select(cs.ends_with("_id"))
df.select(cs.contains("score"))
df.select(cs.matches(r"^f\d+$"))
df.select(cs.by_dtype(pl.Float32))
df.select(cs.first(), cs.last())
df.select(cs.temporal())
df.select(cs.all() - cs.numeric())          # set algebra: difference
df.select(cs.numeric() & ~cs.starts_with("id"))   # intersection & negation
df.select(cs.numeric() | cs.boolean())      # union
```

### Real use: scale every numeric feature except the label

```python
features = cs.numeric() - cs.by_name("label")
df = df.with_columns(
    (features - features.mean()) / features.std()
)
```

---

## 9. 🟡 Null, NaN & Missing Data

Polars distinguishes:

| Concept | Meaning | Check |
|---------|---------|-------|
| `null` | Missing value (any dtype) | `is_null()` |
| `NaN` | A *float* that is "not a number" (e.g., `0/0`) | `is_nan()` |

```python
df.select(
    pl.col("x").is_null().sum().alias("nulls"),
    pl.col("x").is_nan().sum().alias("nans"),
)
```

### Handling nulls

```python
df.drop_nulls()                          # any null in any column
df.drop_nulls(subset=["score"])
df.fill_null(0)
df.with_columns(pl.col("score").fill_null(strategy="mean"))
df.with_columns(pl.col("score").fill_null(strategy="forward"))   # ffill
df.with_columns(pl.col("score").fill_null(strategy="backward"))  # bfill
df.with_columns(pl.col("score").fill_null(pl.col("score").median()))
df.with_columns(pl.col("score").interpolate())                   # linear
df.with_columns(pl.col("x").fill_nan(None))                      # NaN -> null
df.with_columns(pl.coalesce("a", "b", pl.lit(0)))                # first non-null
```

### Group-wise imputation (very common in ML)

```python
df = df.with_columns(
    pl.col("income").fill_null(pl.col("income").median().over("city"))
)
```

### Missing-indicator features

```python
df = df.with_columns(
    pl.col("income").is_null().cast(pl.UInt8).alias("income_missing")
)
```

---

## 10. 🟡 Casting & Type Handling

```python
df.with_columns(
    pl.col("age").cast(pl.Int32),
    pl.col("price").cast(pl.Float32),
    pl.col("zip").cast(pl.String),
    pl.col("flag").cast(pl.Boolean),
    pl.col("amount").cast(pl.Int64, strict=False),   # bad values -> null instead of error
)
```

### Strings to numbers / dates

```python
pl.col("s").cast(pl.Float64, strict=False)
pl.col("d").str.to_date("%Y-%m-%d")
pl.col("ts").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC")
```

### Categorical vs Enum

```python
# Categorical: categories discovered from data
df.with_columns(pl.col("color").cast(pl.Categorical))

# Enum: fixed, known categories (faster, safe, ordered)
sizes = pl.Enum(["S", "M", "L", "XL"])
df.with_columns(pl.col("size").cast(sizes))
df.sort("size")   # sorts by enum order, not alphabetically
```

### Physical encoding for ML

```python
df["color"].cast(pl.Categorical).to_physical()   # integer codes (UInt32)
```

---

## 11. 🟡 String Operations

All under the `.str` namespace.

```python
s = pl.col("text")

s.str.len_chars();  s.str.len_bytes()
s.str.to_lowercase();  s.str.to_uppercase();  s.str.to_titlecase()
s.str.strip_chars();  s.str.strip_chars_start();  s.str.strip_chars_end()
s.str.replace("a", "b");  s.str.replace_all(r"\s+", " ")
s.str.contains("pattern");  s.str.contains("pat", literal=True)
s.str.starts_with("The");  s.str.ends_with("?")
s.str.extract(r"(\d+)", group_index=1)
s.str.extract_all(r"\d+")                 # -> List[String]
s.str.count_matches(r"\w+")
s.str.split(" ")                          # -> List[String]
s.str.split_exact(",", 2)                 # -> Struct
s.str.slice(0, 5);  s.str.head(5);  s.str.tail(5)
s.str.pad_start(8, "0");  s.str.zfill(8)
s.str.reverse()
s.str.json_decode();  s.str.json_path_match("$.a.b")
s.str.to_integer(base=16)
s.str.find("abc")
s.str.contains_any(["spam", "scam"])      # Aho-Corasick, very fast
s.str.replace_many(["a", "b"], ["1", "2"])
```

### Example: clean messy text

```python
clean = (
    pl.col("review")
    .str.to_lowercase()
    .str.replace_all(r"https?://\S+", "")        # URLs
    .str.replace_all(r"[^a-z0-9\s]", " ")        # punctuation
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
)
df = df.with_columns(clean.alias("review_clean"))
```

---

## 12. 🟡 Date & Time Operations

```python
from datetime import date, datetime, timedelta

df = pl.DataFrame({"ts": ["2026-01-15 10:30:00", "2026-03-02 22:05:10"]})
df = df.with_columns(pl.col("ts").str.to_datetime("%Y-%m-%d %H:%M:%S"))
```

### Extract components

```python
pl.col("ts").dt.year();    pl.col("ts").dt.month();   pl.col("ts").dt.day()
pl.col("ts").dt.hour();    pl.col("ts").dt.minute();  pl.col("ts").dt.second()
pl.col("ts").dt.weekday(); pl.col("ts").dt.week();    pl.col("ts").dt.quarter()
pl.col("ts").dt.ordinal_day()
pl.col("ts").dt.date();    pl.col("ts").dt.time()
pl.col("ts").dt.epoch("s")
```

### Arithmetic & truncation

```python
pl.col("ts") + pl.duration(days=7, hours=2)
(pl.col("end") - pl.col("start")).dt.total_seconds()
(pl.col("end") - pl.col("start")).dt.total_days()
pl.col("ts").dt.truncate("1h")           # bucket to hour
pl.col("ts").dt.round("15m")
pl.col("ts").dt.offset_by("1mo")
pl.col("ts").dt.convert_time_zone("Asia/Kolkata")
pl.col("ts").dt.replace_time_zone("UTC")
pl.col("ts").dt.strftime("%b %d, %Y")
```

### Ranges

```python
pl.date_range(date(2026, 1, 1), date(2026, 12, 31), "1d", eager=True)
pl.datetime_range(datetime(2026,1,1), datetime(2026,1,2), "30m", eager=True)
```

### Cyclical encoding for ML (hour-of-day, month)

```python
import math
df = df.with_columns(
    (2 * math.pi * pl.col("ts").dt.hour() / 24).sin().alias("hour_sin"),
    (2 * math.pi * pl.col("ts").dt.hour() / 24).cos().alias("hour_cos"),
    (2 * math.pi * (pl.col("ts").dt.month() - 1) / 12).sin().alias("month_sin"),
    (2 * math.pi * (pl.col("ts").dt.month() - 1) / 12).cos().alias("month_cos"),
    (pl.col("ts").dt.weekday() >= 6).cast(pl.UInt8).alias("is_weekend"),
)
```

---

## 13. 🟡 Conditional Logic

```python
df.with_columns(
    pl.when(pl.col("score") >= 90).then(pl.lit("A"))
      .when(pl.col("score") >= 75).then(pl.lit("B"))
      .when(pl.col("score") >= 60).then(pl.lit("C"))
      .otherwise(pl.lit("F"))
      .alias("grade")
)
```

### Conditional aggregation (no `apply` needed)

```python
df.group_by("team").agg(
    pl.col("score").filter(pl.col("passed")).mean().alias("mean_passed"),
    (pl.col("score") > 80).sum().alias("n_high"),
)
```

### Bucketing numeric values

```python
pl.col("age").cut([18, 30, 45, 60], labels=["<18","18-30","30-45","45-60","60+"])
pl.col("income").qcut(4, labels=["Q1","Q2","Q3","Q4"])     # quantile bins
pl.col("x").clip(lower_bound=0, upper_bound=1)             # clip / winsorize
```

### Mapping values (dictionary replace)

```python
pl.col("country").replace({"IN": "India", "US": "United States"})
pl.col("label").replace_strict({"cat": 0, "dog": 1}, default=-1, return_dtype=pl.Int8)
```

---

## 14. 🟡 GroupBy & Aggregations

```python
df = pl.DataFrame({
    "team":  ["A", "A", "B", "B", "B"],
    "role":  ["dev", "qa", "dev", "dev", "qa"],
    "score": [90, 70, 85, 60, 75],
})

df.group_by("team").agg(
    pl.len().alias("n"),
    pl.col("score").mean().alias("mean"),
    pl.col("score").std().alias("std"),
    pl.col("score").min().alias("min"),
    pl.col("score").max().alias("max"),
    pl.col("score").median().alias("median"),
    pl.col("score").quantile(0.9).alias("p90"),
    pl.col("role").n_unique().alias("roles"),
    pl.col("score").first().alias("first"),
    pl.col("score").last().alias("last"),
    pl.col("score").sort(descending=True).head(2).alias("top2"),   # list per group
    pl.col("role").mode().first().alias("most_common_role"),
)
```

### Options

```python
df.group_by("team", maintain_order=True)      # keep first-appearance order (slower)
df.group_by("team", "role")                   # multi-key
df.group_by(pl.col("ts").dt.year().alias("year"))   # group by expression
```

### Multiple columns at once

```python
df.group_by("team").agg(
    pl.col("a", "b", "c").mean().name.suffix("_mean"),
    cs.numeric().max().name.suffix("_max"),
)
```

### value_counts, unique, n_unique

```python
df["role"].value_counts(sort=True, normalize=True)   # frequencies
df["role"].unique();  df["role"].n_unique()
df.select(pl.col("role").unique_counts())
```

### Class distribution (classification sanity check)

```python
df.group_by("label").agg(pl.len().alias("n")).with_columns(
    (pl.col("n") / pl.col("n").sum()).alias("share")
).sort("label")
```

### Pivot & unpivot

```python
wide = df.pivot(on="role", index="team", values="score", aggregate_function="mean")
long = wide.unpivot(index="team", on=["dev", "qa"],
                    variable_name="role", value_name="score")
```

---

## 15. 🟡 Window Functions with `over`

`over` computes an aggregate **per group but keeps all rows** — no join-back needed. This is the single most useful feature for feature engineering.

```python
df.with_columns(
    pl.col("score").mean().over("team").alias("team_mean"),
    (pl.col("score") - pl.col("score").mean().over("team")).alias("diff_from_team"),
    pl.col("score").rank("dense", descending=True).over("team").alias("rank_in_team"),
    pl.col("score").cum_sum().over("team").alias("cum_score"),
    pl.col("score").shift(1).over("team").alias("prev_score"),
    pl.len().over("team").alias("team_size"),
    pl.col("score").max().over("team", "role").alias("max_by_team_role"),
)
```

### Within-group z-score

```python
df.with_columns(
    ((pl.col("score") - pl.col("score").mean().over("team"))
      / pl.col("score").std().over("team")).alias("z_in_team")
)
```

### Ordered windows

```python
df.with_columns(
    pl.col("score").sort_by("date").cum_sum().over("user_id").alias("running_total"),
    pl.col("score").diff().over("user_id", order_by="date").alias("delta"),
)
```

### Top-N per group

```python
df.filter(pl.col("score").rank("ordinal", descending=True).over("team") <= 2)
# or
df.sort("score", descending=True).group_by("team").head(2)
```

---

## 16. 🟡 Joins, Concatenation, Reshaping

### Joins

```python
left.join(right, on="id", how="inner")
left.join(right, on="id", how="left")
left.join(right, on="id", how="right")
left.join(right, on="id", how="full", coalesce=True)    # outer
left.join(right, on="id", how="semi")     # rows in left that HAVE a match (filter)
left.join(right, on="id", how="anti")     # rows in left with NO match
left.join(right, how="cross")             # cartesian product
left.join(right, left_on="uid", right_on="user_id", suffix="_r")
left.join(right, on=["a", "b"], how="left", validate="m:1")   # assert join cardinality
```

> **Tip:** `semi` / `anti` joins are the cleanest way to do train/test leakage checks and "is in" filtering on big tables.

### As-of join (point-in-time features — crucial to avoid leakage)

```python
trades.sort("time").join_asof(
    quotes.sort("time"),
    on="time",
    by="ticker",
    strategy="backward",        # latest quote at or before trade time
    tolerance="5s",
)
```

### Inequality / range joins

```python
left.join_where(right, pl.col("ts") >= pl.col("start"), pl.col("ts") < pl.col("end"))
```

### Concatenation

```python
pl.concat([df1, df2], how="vertical")            # stack rows (same schema)
pl.concat([df1, df2], how="vertical_relaxed")    # coerce compatible dtypes
pl.concat([df1, df2], how="diagonal")            # union of columns, fill null
pl.concat([df1, df2], how="horizontal")          # side by side
```

### Reshape

```python
df.transpose(include_header=True)
df.explode("tags")                       # list column -> rows
df.unnest("my_struct")                   # struct -> columns
df.to_dummies(columns=["color"])         # one-hot
df.partition_by("team", as_dict=True)    # dict of DataFrames
```

---

## 17. 🔴 List, Array & Struct Types

Nested types are first-class — perfect for embeddings, token lists, and JSON-like data.

### List (variable length)

```python
df = pl.DataFrame({"id": [1, 2], "tokens": [[101, 7, 9, 102], [101, 5, 102]]})

df.select(
    pl.col("tokens").list.len().alias("n_tokens"),
    pl.col("tokens").list.first(),
    pl.col("tokens").list.get(1, null_on_oob=True),
    pl.col("tokens").list.slice(0, 2),
    pl.col("tokens").list.sum(),
    pl.col("tokens").list.mean(),
    pl.col("tokens").list.sort(),
    pl.col("tokens").list.unique(),
    pl.col("tokens").list.contains(7),
    pl.col("tokens").list.join(","),                     # only for list[str]
    pl.col("tokens").list.eval(pl.element() * 2),        # per-element expression
    pl.col("tokens").list.to_struct(n_field_strategy="max_width"),
)
df.explode("tokens")
df.group_by("id").agg(pl.col("tokens"))                  # build lists
```

### Array (fixed width — ideal for embeddings)

```python
emb = pl.DataFrame(
    {"id": [1, 2], "vec": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
    schema={"id": pl.Int64, "vec": pl.Array(pl.Float32, 3)},
)
emb.select(
    pl.col("vec").arr.sum(), pl.col("vec").arr.mean(),
    pl.col("vec").arr.get(0), pl.col("vec").arr.max(),
)
np_matrix = emb["vec"].to_numpy()          # shape (n, 3), zero-copy-ish
```

### Cosine similarity with expressions

```python
q = [0.2, 0.1, 0.9]
def cos_sim(col: str, q: list[float]) -> pl.Expr:
    qn = sum(x * x for x in q) ** 0.5
    dot = pl.sum_horizontal(pl.col(col).arr.get(i) * q[i] for i in range(len(q)))
    norm = pl.sum_horizontal(pl.col(col).arr.get(i) ** 2 for i in range(len(q))).sqrt()
    return dot / (norm * qn)

emb.with_columns(cos_sim("vec", q).alias("sim")).sort("sim", descending=True)
```

> For real vector search at scale use FAISS / a vector DB; Polars is great for the **preprocessing** and **metadata filtering** around it.

### Struct (named fields)

```python
df = pl.DataFrame({"user": [{"name": "Asha", "age": 21}, {"name": "Ravi", "age": 25}]})
df.select(pl.col("user").struct.field("name"))
df.unnest("user")
df.select(pl.struct("a", "b").alias("ab"))
df.select(pl.col("x").value_counts())            # returns Struct{x, count}
```

### Horizontal (row-wise) operations

```python
df.with_columns(
    pl.sum_horizontal("a", "b", "c").alias("total"),
    pl.mean_horizontal(cs.numeric()).alias("row_mean"),
    pl.max_horizontal("a", "b"),
    pl.all_horizontal(pl.col("x") > 0, pl.col("y") > 0),
    pl.any_horizontal(cs.boolean()),
    pl.concat_list("a", "b", "c").alias("as_list"),
)
```

---

## 18. 🔴 Rolling, Dynamic & Time-Series

### Rolling windows (fixed number of rows)

```python
df.with_columns(
    pl.col("price").rolling_mean(window_size=7).alias("ma7"),
    pl.col("price").rolling_std(window_size=7).alias("std7"),
    pl.col("price").rolling_min(7), pl.col("price").rolling_max(7),
    pl.col("price").ewm_mean(span=10).alias("ema10"),
    pl.col("price").rolling_mean(7, min_samples=1, center=False),
)
```

### Time-based rolling (window by duration)

```python
df.sort("ts").with_columns(
    pl.col("amount").rolling_sum_by("ts", window_size="7d", closed="left").alias("sum_7d"),
    pl.col("amount").rolling_mean_by("ts", window_size="30d").alias("mean_30d"),
)
```

> Use `closed="left"` so the current row is **excluded** — prevents target leakage.

### Per-entity rolling features

```python
df.sort("ts").with_columns(
    pl.col("amount").rolling_mean_by("ts", window_size="7d", closed="left")
      .over("user_id").alias("user_mean_7d")
)
```

### Dynamic group-by (time buckets)

```python
df.sort("ts").group_by_dynamic("ts", every="1h", period="1h", closed="left").agg(
    pl.col("value").mean().alias("hourly_mean"),
    pl.len().alias("events"),
)
df.sort("ts").group_by_dynamic("ts", every="1d", group_by="sensor").agg(pl.col("v").sum())
```

### Rolling group-by (windows anchored at each row)

```python
df.sort("ts").rolling(index_column="ts", period="3d").agg(pl.col("v").sum())
```

### Lags, leads, diffs, pct_change

```python
df.with_columns(
    pl.col("y").shift(1).alias("lag1"),
    pl.col("y").shift(7).alias("lag7"),
    pl.col("y").shift(-1).alias("lead1"),
    pl.col("y").diff().alias("diff"),
    pl.col("y").pct_change().alias("ret"),
    pl.col("y").cum_max(), pl.col("y").cum_min(),
)
```

### Upsample / fill gaps

```python
df.sort("ts").upsample(time_column="ts", every="1h").with_columns(pl.col("v").interpolate())
```

### Time-based train/val/test split (never random for time-series!)

```python
cutoff_val  = pl.datetime(2026, 6, 1)
cutoff_test = pl.datetime(2026, 9, 1)
train = df.filter(pl.col("ts") <  cutoff_val)
val   = df.filter(pl.col("ts").is_between(cutoff_val, cutoff_test, closed="left"))
test  = df.filter(pl.col("ts") >= cutoff_test)
```

---

## 19. 🔴 Lazy API & Query Optimization

### Eager vs lazy

```python
# Eager: runs each step immediately, materializes intermediates
df = pl.read_csv("big.csv")
out = df.filter(pl.col("x") > 0).group_by("k").agg(pl.col("y").mean())

# Lazy: builds a plan, optimizes, runs once
out = (
    pl.scan_csv("big.csv")
      .filter(pl.col("x") > 0)
      .group_by("k")
      .agg(pl.col("y").mean())
      .collect()
)
```

### Optimizations Polars applies automatically

| Optimization | What happens |
|--------------|--------------|
| **Predicate pushdown** | Filters are applied while scanning the file, skipping rows / Parquet row groups |
| **Projection pushdown** | Only the columns you use are read from disk |
| **Slice pushdown** | `.head(n)` stops reading early |
| **Common subplan elimination** | Repeated sub-queries computed once |
| **Common subexpression elimination** | Repeated expressions computed once |
| **Expression simplification** | Constant folding, algebraic rewrites |
| **Join ordering / type coercion** | Cheaper join and dtype plans |

### Inspect the plan

```python
lf = pl.scan_parquet("data.parquet").filter(pl.col("x") > 0).select("x", "y")
print(lf.explain())                 # optimized plan
print(lf.explain(optimized=False))  # as written
lf.show_graph()                     # graphviz diagram
lf.profile()                        # (result, timing DataFrame per node)
lf.collect_schema()                 # schema WITHOUT running
```

Read plans bottom-up. You want to see `SELECTION` and `PROJECT x/y COLUMNS` pushed into the `Parquet SCAN`.

### Good lazy habits

```python
lf = pl.scan_parquet("data/*.parquet")

features = (
    lf.filter(pl.col("is_valid"))
      .with_columns(...)
      .group_by("user_id").agg(...)
)
labels = pl.scan_parquet("labels.parquet")

# Run several queries in parallel, sharing scans
a, b = pl.collect_all([features, labels])
```

### `pipe` for reusable, testable stages

```python
def add_time_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.col("ts").dt.hour().alias("hour"),
        pl.col("ts").dt.weekday().alias("dow"),
    )

def add_user_stats(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(pl.col("amt").mean().over("user").alias("user_avg_amt"))

result = lf.pipe(add_time_features).pipe(add_user_stats).collect()
```

---

## 20. 🔴 Streaming & Larger-Than-Memory Data

```python
# Streaming engine (batch-wise execution)
result = lf.collect(engine="streaming")       # newer Polars (1.2x+)
result = lf.collect(streaming=True)           # older Polars

# Write results without ever holding them in RAM
pl.scan_csv("huge.csv") \
  .filter(pl.col("country") == "IN") \
  .with_columns(pl.col("price").cast(pl.Float32)) \
  .sink_parquet("india.parquet")
```

### Strategies for datasets bigger than memory

1. **Convert to Parquet** with sensible row-group size, `zstd` compression.
2. **`scan_*` + filter + select early** so pushdowns kick in.
3. **`sink_parquet`** instead of `collect()` for big outputs.
4. **Partition on disk** (e.g., by date) and use `hive_partitioning=True`.
5. **Process per-shard** in a loop, writing intermediate Parquet files.
6. Cast to smaller dtypes (`Float32`, `UInt8`, `Categorical`) as early as possible.

### Chunked batch iteration (for training loops)

```python
reader = pl.read_csv_batched("huge.csv", batch_size=100_000)
while (batches := reader.next_batches(5)):
    for chunk in batches:
        ...   # process each DataFrame chunk

# Or slice a lazy frame
n = lf.select(pl.len()).collect().item()
for offset in range(0, n, 100_000):
    chunk = lf.slice(offset, 100_000).collect()
    ...
```

### Partitioned writes

```python
df.write_parquet("dataset/", partition_by=["year", "month"])
pl.scan_parquet("dataset/", hive_partitioning=True).filter(pl.col("year") == 2026)
```

---

## 21. 🔴 Polars for ML: Feature Engineering

### End-to-end tabular pipeline

```python
import polars as pl
import polars.selectors as cs

raw = pl.scan_parquet("customers.parquet")

TARGET = "churned"
NUM = ["age", "tenure", "monthly_spend"]
CAT = ["plan", "city"]

df = (
    raw
    .filter(pl.col(TARGET).is_not_null())
    .with_columns(
        # 1. impute
        pl.col(NUM).fill_null(pl.col(NUM).median()),
        pl.col(CAT).fill_null("unknown"),
        # 2. ratios & interactions
        (pl.col("monthly_spend") / (pl.col("tenure") + 1)).alias("spend_per_month"),
        (pl.col("age") * pl.col("tenure")).alias("age_x_tenure"),
        # 3. log transform for skew
        pl.col("monthly_spend").log1p().alias("log_spend"),
    )
    .collect()
)
```

### Train / test split

```python
SEED = 42

# Simple random split
shuffled = df.sample(fraction=1.0, shuffle=True, seed=SEED)
n_train = int(0.8 * df.height)
train, test = shuffled[:n_train], shuffled[n_train:]

# Reproducible, row-hash based split (stable even when data grows)
df = df.with_columns((pl.col("customer_id").hash(seed=SEED) % 100).alias("bucket"))
train = df.filter(pl.col("bucket") < 80)
test  = df.filter(pl.col("bucket") >= 80)

# Stratified split
test = df.group_by("label", maintain_order=True).map_groups(
    lambda g: g.sample(fraction=0.2, seed=SEED)
)
train = df.join(test.select("customer_id"), on="customer_id", how="anti")
```

### Scaling — fit on train, apply to test (avoid leakage!)

```python
stats = train.select(
    [pl.col(c).mean().alias(f"{c}_mean") for c in NUM] +
    [pl.col(c).std().alias(f"{c}_std")   for c in NUM]
).row(0, named=True)

def standardize(d: pl.DataFrame) -> pl.DataFrame:
    return d.with_columns(
        [((pl.col(c) - stats[f"{c}_mean"]) / stats[f"{c}_std"]).alias(c) for c in NUM]
    )

train_s, test_s = standardize(train), standardize(test)
```

```python
# Min-max
(pl.col("x") - pl.col("x").min()) / (pl.col("x").max() - pl.col("x").min())
# Robust scaling
(pl.col("x") - pl.col("x").median()) / (pl.col("x").quantile(0.75) - pl.col("x").quantile(0.25))
# Winsorize outliers
pl.col("x").clip(pl.col("x").quantile(0.01), pl.col("x").quantile(0.99))
```

### Encoding categoricals

```python
# One-hot
df.to_dummies(columns=["plan"], drop_first=True)

# Ordinal / label
df.with_columns(pl.col("plan").cast(pl.Categorical).to_physical().alias("plan_id"))

# Frequency encoding
df.with_columns(pl.len().over("city").alias("city_freq"))

# Target (mean) encoding — compute on TRAIN only, with smoothing
m = 20
prior = train["churned"].mean()
te = train.group_by("city").agg(
    pl.col("churned").sum().alias("s"), pl.len().alias("n")
).with_columns(
    ((pl.col("s") + m * prior) / (pl.col("n") + m)).alias("city_te")
).select("city", "city_te")
test = test.join(te, on="city", how="left").with_columns(pl.col("city_te").fill_null(prior))

# Hashing trick
pl.col("category").hash() % 1024
```

### Out-of-fold target encoding (K-fold, leakage-safe)

```python
K = 5
df = df.with_columns((pl.int_range(pl.len()).shuffle(seed=SEED) % K).alias("fold"))
oof = []
for k in range(K):
    tr, va = df.filter(pl.col("fold") != k), df.filter(pl.col("fold") == k)
    enc = tr.group_by("city").agg(pl.col("churned").mean().alias("city_te"))
    oof.append(va.join(enc, on="city", how="left"))
df = pl.concat(oof)
```

### Aggregate (entity-level) features

```python
user_features = events.group_by("user_id").agg(
    pl.len().alias("n_events"),
    pl.col("amount").sum().alias("total_amt"),
    pl.col("amount").mean().alias("avg_amt"),
    pl.col("amount").std().alias("std_amt"),
    pl.col("ts").max().alias("last_seen"),
    (pl.col("ts").max() - pl.col("ts").min()).dt.total_days().alias("active_days"),
    pl.col("category").n_unique().alias("n_categories"),
    pl.col("category").mode().first().alias("fav_category"),
    (pl.col("is_refund").mean()).alias("refund_rate"),
)
```

### Recency / Frequency / Monetary (RFM)

```python
now = pl.lit(datetime(2026, 11, 23))
rfm = tx.group_by("customer_id").agg(
    (now - pl.col("ts").max()).dt.total_days().alias("recency"),
    pl.len().alias("frequency"),
    pl.col("amount").sum().alias("monetary"),
).with_columns(
    pl.col("recency").qcut(5, labels=[str(i) for i in range(5, 0, -1)]).alias("R"),
    pl.col("frequency").qcut(5, labels=[str(i) for i in range(1, 6)], allow_duplicates=True).alias("F"),
    pl.col("monetary").qcut(5, labels=[str(i) for i in range(1, 6)]).alias("M"),
)
```

### Correlation, feature selection helpers

```python
df.select(pl.corr("a", "b"))
df.select(cs.numeric()).corr()                         # correlation matrix (DataFrame)
df.select(pl.corr("feature", "label", method="spearman"))

# Drop near-constant features
const = [c for c in df.columns if df[c].n_unique() <= 1]
df = df.drop(const)

# Missing-rate filter
miss = df.null_count().row(0, named=True)
drop_cols = [c for c, n in miss.items() if n / df.height > 0.6]
```

### Class imbalance helpers

```python
# Class weights
counts = df["label"].value_counts()
weights = counts.with_columns((df.height / (pl.len() * pl.col("count"))).alias("w"))

# Random under-sampling of the majority class
minority = df.filter(pl.col("label") == 1)
majority = df.filter(pl.col("label") == 0).sample(n=minority.height, seed=SEED)
balanced = pl.concat([minority, majority]).sample(fraction=1.0, shuffle=True, seed=SEED)
```

### Data validation (cheap, fast sanity checks)

```python
def validate(df: pl.DataFrame):
    assert df["id"].is_unique().all(), "duplicate ids"
    assert df["age"].is_between(0, 120).all(), "age out of range"
    assert df.select(pl.col("label").is_in([0, 1]).all()).item()
    assert df.null_count().select(pl.sum_horizontal(pl.all())).item() == 0
```

---

## 22. 🔴 Polars + ML Frameworks

### → NumPy

```python
X = df.select(FEATURES).to_numpy()                 # 2D ndarray
y = df["label"].to_numpy()                         # 1D ndarray
X32 = df.select(pl.col(FEATURES).cast(pl.Float32)).to_numpy()
```

### ↔ pandas / Arrow

```python
pdf = df.to_pandas()                  # needs pyarrow
df  = pl.from_pandas(pdf)
tbl = df.to_arrow()
df  = pl.from_arrow(tbl)
```

### scikit-learn (2 ways)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X = df.select(FEATURES)
y = df["label"]

# sklearn >= 1.4 accepts Polars directly
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=300, n_jobs=-1).fit(X_tr, y_tr)
print(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))

# Make transformers output Polars
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().set_output(transform="polars")
X_scaled = scaler.fit_transform(X_tr)       # returns pl.DataFrame
```

### Pipelines with ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pre = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
]).set_output(transform="polars")

pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier())])
pipe.fit(train.select(NUM + CAT), train["label"])
```

### XGBoost / LightGBM / CatBoost

```python
import xgboost as xgb, lightgbm as lgb

dtrain = xgb.DMatrix(train.select(FEATURES).to_numpy(), label=train["label"].to_numpy(),
                     feature_names=FEATURES)
bst = xgb.train({"objective": "binary:logistic", "eval_metric": "auc"}, dtrain, 300)

lgb_train = lgb.Dataset(train.select(FEATURES).to_pandas(), label=train["label"].to_numpy())
```

> XGBoost, LightGBM and CatBoost increasingly accept Polars / Arrow directly — check your installed version's docs before converting.

### PyTorch

```python
import torch
from torch.utils.data import Dataset, DataLoader

class PolarsDataset(Dataset):
    def __init__(self, df: pl.DataFrame, features: list[str], target: str):
        # Convert ONCE; indexing numpy is far faster than indexing a DataFrame
        self.X = torch.from_numpy(df.select(features).cast(pl.Float32).to_numpy())
        self.y = torch.from_numpy(df[target].cast(pl.Float32).to_numpy())

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

loader = DataLoader(PolarsDataset(train, FEATURES, "label"),
                    batch_size=256, shuffle=True, num_workers=2)
```

```python
# Direct tensor conversion (Polars >= 1.x)
X = df.select(FEATURES).to_torch()
```

> ⚠️ **Never** do `df[i]` / `df.row(i)` inside `__getitem__` — it is orders of magnitude slower than indexing a pre-built tensor.

### Hugging Face Datasets

```python
from datasets import Dataset as HFDataset

hf = HFDataset.from_polars(df)        # polars -> HF
df2 = hf.to_polars()                  # HF -> polars
```

### TensorFlow / Keras

```python
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices((
    df.select(FEATURES).to_numpy(), df["label"].to_numpy()
)).shuffle(10_000).batch(256).prefetch(tf.data.AUTOTUNE)
```

### Plotting

```python
df.plot.line(x="ts", y="value")          # via altair (hvplot-style API)
import matplotlib.pyplot as plt
plt.hist(df["score"].to_numpy(), bins=30)
import seaborn as sns
sns.histplot(data=df.to_pandas(), x="score")
```

### Experiment tracking: log dataset stats

```python
import json
stats = df.describe().to_dicts()
# mlflow.log_dict({"schema": {k: str(v) for k, v in df.schema.items()}, "describe": stats}, "data_profile.json")
```

---

## 23. 🔴 NLP & Text Preprocessing Pipelines

### Clean → tokenize → count

```python
corpus = pl.DataFrame({"text": ["I love Polars!", "Polars is FAST, really fast.", None]})

tokens = (
    corpus.drop_nulls()
    .with_columns(
        pl.col("text")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z\s]", "")
        .str.split(" ")
        .alias("tokens")
    )
)

STOP = ["is", "i", "the", "a", "an"]
tokens = tokens.with_columns(
    pl.col("tokens").list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != ""))).alias("tokens")
)
```

### Vocabulary & frequencies

```python
vocab = (
    tokens.select(pl.col("tokens").explode().alias("tok"))
    .group_by("tok").agg(pl.len().alias("freq"))
    .filter(pl.col("freq") >= 2)
    .sort("freq", descending=True)
    .with_row_index("idx")                   # token -> id
)

tok2id = dict(zip(vocab["tok"], vocab["idx"]))
tokens = tokens.with_columns(
    pl.col("tokens").list.eval(pl.element().replace_strict(tok2id, default=-1)).alias("ids")
)
```

### TF (term frequency) / TF-IDF

```python
N = tokens.height
tf = (tokens.with_row_index("doc").explode("tokens")
      .group_by("doc", "tokens").agg(pl.len().alias("tf")))
df_ = tf.group_by("tokens").agg(pl.col("doc").n_unique().alias("df"))
tfidf = tf.join(df_, on="tokens").with_columns(
    (pl.col("tf") * ((N / pl.col("df")).log())).alias("tfidf")
)
```

### Fixed-length padding for transformers

```python
MAXLEN = 128
df = (
    df.with_columns(pl.col("ids").list.slice(0, MAXLEN))            # truncate
      .with_columns(
          pl.col("ids").list.len().alias("len"),
          pl.concat_list(
              pl.col("ids"),
              pl.lit(0).repeat_by(MAXLEN - pl.col("ids").list.len()),   # pad with 0
          ).alias("ids_padded"),
      )
)
```

### LLM / instruction-dataset prep (JSONL)

```python
ds = pl.read_ndjson("raw_chat.jsonl")

ds = (
    ds.filter(pl.col("response").str.len_chars().is_between(20, 4000))
      .filter(~pl.col("prompt").str.contains_any(["BANNED1", "BANNED2"]))
      .unique(subset=["prompt"])                                   # exact dedup
      .with_columns(
          pl.format("### Instruction:\n{}\n\n### Response:\n{}", "prompt", "response").alias("text"),
          pl.col("text").str.len_chars().alias("n_chars"),
      )
)
ds.write_ndjson("train.jsonl")
```

### Tokenize with Hugging Face tokenizers (batched)

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

texts = df["text"].to_list()
enc = tok(texts, truncation=True, max_length=128, padding="max_length")
df = df.with_columns(pl.Series("input_ids", enc["input_ids"], dtype=pl.List(pl.Int32)))
```

### Simple near-duplicate filtering (shingle hash)

```python
df.with_columns(
    pl.col("text").str.to_lowercase().str.replace_all(r"\W+", " ").hash().alias("h")
).unique(subset="h")
```

---

## 24. 🔴 Custom Functions, UDFs & Plugins

### Performance ladder (fastest → slowest)

| Rank | Approach | Use when |
|------|----------|----------|
| 🥇 | Native expressions | Always try first |
| 🥈 | `map_batches` on a whole Series with NumPy | Need NumPy/SciPy vectorization |
| 🥉 | Expression plugins (Rust) | Hot path, custom kernel |
| 🐌 | `map_elements` (row-by-row Python) | Last resort |

### `map_batches` — vectorized UDF

```python
import numpy as np

df.with_columns(
    pl.col("x").map_batches(lambda s: pl.Series(np.tanh(s.to_numpy())), return_dtype=pl.Float64)
)
```

### `map_elements` — per-value Python (slow)

```python
df.with_columns(
    pl.col("text").map_elements(my_python_fn, return_dtype=pl.String, skip_nulls=True)
)
```

> Typical cost: 10–100× slower than native. If you reach for this, first ask: *can it be `when/then`, `replace`, `str.*`, `list.eval`, or `struct`?*

### Whole-frame functions

```python
df.pipe(my_function, arg=1)
df.group_by("g").map_groups(lambda g: g.with_columns(pl.col("x") - pl.col("x").mean()))
```

### Registering your own namespace

```python
@pl.api.register_expr_namespace("ml")
class MLNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def zscore(self) -> pl.Expr:
        return (self._expr - self._expr.mean()) / self._expr.std()

    def minmax(self) -> pl.Expr:
        return (self._expr - self._expr.min()) / (self._expr.max() - self._expr.min())

df.with_columns(pl.col("x").ml.zscore().alias("x_z"))

@pl.api.register_dataframe_namespace("feat")
class FeatNS:
    def __init__(self, df: pl.DataFrame):
        self._df = df
    def add_lags(self, col: str, lags=(1, 7)):
        return self._df.with_columns([pl.col(col).shift(l).alias(f"{col}_lag{l}") for l in lags])

df.feat.add_lags("y")
```

### Rust expression plugins (advanced)

Use [`pyo3-polars`](https://github.com/pola-rs/pyo3-polars) + `maturin` to compile custom Rust kernels and call them from Python with `register_plugin_function`. Reach for this only when profiling proves a Python/NumPy path is the bottleneck.

---

## 25. 🟡 SQL Interface

```python
ctx = pl.SQLContext(users=users_df, orders=orders_lf, eager=True)

result = ctx.execute("""
    SELECT u.country,
           COUNT(*)            AS n_orders,
           AVG(o.amount)       AS avg_amount,
           SUM(o.amount)       AS revenue
    FROM users u
    JOIN orders o ON u.id = o.user_id
    WHERE o.amount > 10
    GROUP BY u.country
    HAVING COUNT(*) > 5
    ORDER BY revenue DESC
    LIMIT 10
""")

# One-liner on a single frame
df.sql("SELECT label, COUNT(*) AS n FROM self GROUP BY label")

# Mix SQL expressions with the expression API
pl.sql_expr("score * 2 AS score2")
```

---

## 26. 🔴 Performance Tuning Playbook

### Checklist

- [ ] **Use Parquet**, not CSV, as your working format
- [ ] **Use `scan_*` + `.collect()`** (lazy) for any multi-step pipeline
- [ ] **Select only needed columns** and **filter early**
- [ ] **Use native expressions**; avoid `map_elements`
- [ ] **Avoid Python loops over rows** (`iter_rows`, `rows`, `to_dicts`)
- [ ] **Downcast dtypes** (`Float32`, `UInt8`, `Int32`, `Categorical`/`Enum`)
- [ ] **Prefer `over()`** to group-by + join-back
- [ ] **Sort once**, then use `set_sorted` where valid to unlock fast paths
- [ ] **Use `streaming`** for data larger than RAM
- [ ] **Profile** with `lf.profile()` and read `lf.explain()`
- [ ] **Avoid `pl.Object`** columns
- [ ] **Chain `.with_columns`** only when needed (one call = parallel)

### Do / Don't

```python
# ❌ slow: Python loop
out = [row["a"] * 2 for row in df.iter_rows(named=True)]

# ✅ fast: expression
out = df.select((pl.col("a") * 2))

# ❌ slow: group_by then join back
agg = df.group_by("g").agg(pl.col("x").mean().alias("m"))
df = df.join(agg, on="g")

# ✅ fast: window
df = df.with_columns(pl.col("x").mean().over("g").alias("m"))

# ❌ many sequential with_columns that are independent
df = df.with_columns(a=...).with_columns(b=...).with_columns(c=...)

# ✅ one call, runs in parallel
df = df.with_columns(a=..., b=..., c=...)
```

### Thread control

```python
import os
os.environ["POLARS_MAX_THREADS"] = "8"      # must be set BEFORE importing polars
import polars as pl
print(pl.thread_pool_size())
```

### Useful environment variables

| Variable | Effect |
|----------|--------|
| `POLARS_MAX_THREADS` | Limit thread pool size |
| `POLARS_VERBOSE=1` | Print engine diagnostics |
| `POLARS_FMT_MAX_ROWS` | Table print rows |
| `POLARS_STREAMING_CHUNK_SIZE` | Streaming batch size |

### Benchmark template

```python
import time
def timeit(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)

print("polars:", timeit(lambda: pl_pipeline().collect()))
print("pandas:", timeit(lambda: pd_pipeline()))
```

### GPU engine (optional)

```python
# NVIDIA GPUs via RAPIDS cuDF backend
result = lf.collect(engine="gpu")
```

---

## 27. 🟢 Pandas → Polars Cheat Sheet

| Task | Pandas | Polars |
|------|--------|--------|
| Read CSV | `pd.read_csv(f)` | `pl.read_csv(f)` / `pl.scan_csv(f)` |
| Select cols | `df[["a","b"]]` | `df.select("a","b")` |
| Filter | `df[df.a > 5]` | `df.filter(pl.col("a") > 5)` |
| New column | `df["c"] = df.a * 2` | `df.with_columns((pl.col("a")*2).alias("c"))` |
| Rename | `df.rename(columns={...})` | `df.rename({...})` |
| Drop | `df.drop(columns=["a"])` | `df.drop("a")` |
| Sort | `df.sort_values("a", ascending=False)` | `df.sort("a", descending=True)` |
| GroupBy agg | `df.groupby("g")["x"].mean()` | `df.group_by("g").agg(pl.col("x").mean())` |
| Group transform | `df.groupby("g")["x"].transform("mean")` | `pl.col("x").mean().over("g")` |
| Value counts | `df.a.value_counts()` | `df["a"].value_counts()` |
| Unique | `df.a.unique()` | `df["a"].unique()` |
| Drop dups | `df.drop_duplicates()` | `df.unique()` |
| Null check | `df.isna()` | `df.null_count()` / `pl.all().is_null()` |
| Fill NA | `df.fillna(0)` | `df.fill_null(0)` |
| Drop NA | `df.dropna()` | `df.drop_nulls()` |
| Apply | `df.a.apply(f)` | `pl.col("a").map_elements(f)` (avoid) |
| Cast | `df.a.astype(float)` | `pl.col("a").cast(pl.Float64)` |
| Merge | `pd.merge(a,b,on="k",how="left")` | `a.join(b, on="k", how="left")` |
| Concat rows | `pd.concat([a,b])` | `pl.concat([a,b])` |
| Pivot | `df.pivot_table(...)` | `df.pivot(on=, index=, values=, aggregate_function=)` |
| Melt | `df.melt(...)` | `df.unpivot(...)` |
| Where / mask | `np.where(c, a, b)` | `pl.when(c).then(a).otherwise(b)` |
| Cut | `pd.cut` | `pl.col().cut()` |
| Rolling | `df.a.rolling(7).mean()` | `pl.col("a").rolling_mean(7)` |
| Shift | `df.a.shift(1)` | `pl.col("a").shift(1)` |
| One-hot | `pd.get_dummies(df)` | `df.to_dummies()` |
| Describe | `df.describe()` | `df.describe()` |
| Row count | `len(df)` | `df.height` / `pl.len()` |
| Index reset | `reset_index()` | *(no index)* `with_row_index()` |
| Sample | `df.sample(frac=0.5)` | `df.sample(fraction=0.5)` |
| To list | `df.a.tolist()` | `df["a"].to_list()` |
| iloc | `df.iloc[0]` | `df.row(0)` / `df[0]` |
| String contains | `df.a.str.contains("x")` | `pl.col("a").str.contains("x")` |
| Datetime | `df.a.dt.year` | `pl.col("a").dt.year()` |

### Mindset shifts

| Pandas habit | Polars way |
|--------------|-----------|
| Mutate in place (`df["x"] = ...`) | Immutable: always return a new frame |
| Boolean mask indexing | `filter` with expressions |
| Index alignment | No index; align with `join` |
| `apply` everywhere | Compose expressions |
| `inplace=True` | Not supported (and not needed) |
| Object dtype strings | Native `String` dtype |
| `NaN` as missing | `null` as missing |

---

## 28. 🟡 Common Pitfalls

<details>
<summary><b>⚠️ Click to expand the 14 most common mistakes</b></summary>

1. **Using `df["col"] = ...`** → Polars frames are immutable. Use `with_columns`.
2. **Referencing a just-created column in the same `with_columns`** → chain another `with_columns`.
3. **Confusing `null` and `NaN`** → `fill_null` does not touch `NaN`; use `fill_nan`.
4. **Using `map_elements` for simple logic** → rewrite using expressions.
5. **Forgetting `.collect()`** on a `LazyFrame` and wondering why it prints a plan.
6. **Loading CSV with wrong inferred types** → set `schema_overrides` or raise `infer_schema_length` (IDs like `"007"` and zip codes!).
7. **Leaky scaling / encoding** → compute statistics on **train only**, apply to val/test.
8. **Non-deterministic order after `group_by`** → pass `maintain_order=True` or sort explicitly.
9. **Random split on time-series** → always split by time.
10. **Rolling features that include the current row** → use `closed="left"` or `shift(1)`.
11. **Inequality on strings vs dates** → parse to `Date`/`Datetime` first.
12. **Row-wise iteration in a PyTorch `Dataset`** → convert to tensors once.
13. **Using `pl.count()`** → deprecated; use `pl.len()`.
14. **Using `melt` / `groupby`** → renamed to `unpivot` / `group_by`; check the version you run.

</details>

### Version-drift note

Polars is young and evolves fast (1.x is API-stable, but deprecations still occur). Pin your version in `requirements.txt` and read release notes when upgrading.

```
polars==1.*   # pin an exact version in production, e.g. polars==1.34.0
```

---

## 29. 🟡 Practice Projects

| Level | Project | Skills Practiced |
|-------|---------|------------------|
| 🟢 | **Titanic EDA with Polars** | read_csv, describe, filter, group_by, null handling |
| 🟢 | **CSV → Parquet converter CLI** | scan_csv, sink_parquet, dtypes |
| 🟡 | **NYC Taxi 100M rows analytics** | lazy API, streaming, dynamic group_by |
| 🟡 | **Customer churn feature store** | window functions, RFM, target encoding |
| 🟡 | **Text cleaning pipeline for sentiment model** | str namespace, list.eval, vocab building |
| 🔴 | **Time-series forecasting features (M5 / Store sales)** | rolling_by, lags, as-of joins, time splits |
| 🔴 | **LLM instruction-dataset curator** | NDJSON, dedup, filtering, templating, stats |
| 🔴 | **Streaming feature pipeline → PyTorch DataLoader** | sink_parquet, partitioning, tensor datasets |
| 🔴 | **Polars vs Pandas vs DuckDB benchmark suite** | profiling, plots, write-up |
| 🔴 | **Custom Polars plugin in Rust** | pyo3-polars, maturin |

---

## 30. 🔴 Interview Questions & Answers

<details>
<summary><b>🟢 Beginner</b></summary>

**Q1. What is Polars and why is it fast?**
A Rust-based DataFrame library on Apache Arrow's columnar format. Fast because of vectorized/SIMD columnar ops, automatic multi-threading, a lazy query optimizer, and no Python overhead in the hot path.

**Q2. Eager vs lazy?**
Eager (`DataFrame`) executes each call immediately. Lazy (`LazyFrame`) records a plan, optimizes it (predicate/projection pushdown, CSE), and runs on `.collect()`.

**Q3. Difference between `select` and `with_columns`?**
`select` returns only the expressions you list; `with_columns` adds/replaces columns and keeps all others.

**Q4. How are missing values handled?**
Unified `null` via validity bitmaps for every dtype; `NaN` is a separate float value. Use `is_null`, `fill_null`, `drop_nulls`; `fill_nan` for NaN.

**Q5. Does Polars have an index?**
No. Row position is the only identity; use `with_row_index()` or joins for alignment.

</details>

<details>
<summary><b>🟡 Intermediate</b></summary>

**Q6. What is an expression?**
A lazy description of a computation on columns (`pl.col("a").mean()`), evaluated inside a context (`select`, `with_columns`, `filter`, `agg`). Independent expressions are run in parallel.

**Q7. How do you compute a group statistic but keep all rows?**
`pl.col("x").mean().over("group")`.

**Q8. `semi` vs `anti` join?**
`semi` keeps left rows that have a match in right; `anti` keeps left rows with no match. Neither adds right-side columns.

**Q9. Categorical vs Enum?**
`Categorical` infers categories from data; `Enum` has a fixed, ordered set declared upfront — faster, stricter, and sorts by declared order.

**Q10. Why is `map_elements` discouraged?**
It calls Python per element, bypassing vectorization, parallelism and the optimizer.

**Q11. How do you avoid leakage when scaling?**
Fit statistics (mean/std/quantiles) on the training split only, then apply those constants to val/test.

</details>

<details>
<summary><b>🔴 Advanced</b></summary>

**Q12. How do you process a 200 GB dataset on a 32 GB machine?**
Convert to partitioned Parquet; `scan_parquet` with early filter/select so pushdowns prune data; use the streaming engine; `sink_parquet` outputs; downcast dtypes.

**Q13. What optimizations does the lazy engine perform?**
Predicate pushdown, projection pushdown, slice pushdown, common subplan/subexpression elimination, expression simplification, type coercion, and join ordering. Inspect via `explain()`.

**Q14. Build leakage-free rolling features for a user-level model.**
`sort("ts")` then `rolling_mean_by("ts", window_size="7d", closed="left").over("user_id")` — the window excludes the current row, and `over` isolates each entity.

**Q15. Point-in-time correct joins?**
`join_asof(..., by=entity, strategy="backward", tolerance=...)` on sorted keys so each row only sees data available at its timestamp.

**Q16. How would you feed Polars data to PyTorch efficiently?**
Convert to a contiguous `float32` NumPy array/tensor once in `__init__`; index tensors in `__getitem__`; or pre-shard to Parquet and stream batches.

**Q17. When would you still choose pandas / DuckDB / Spark?**
Pandas for ecosystem compatibility and small data; DuckDB for SQL-first analytic workloads and out-of-core joins; Spark when data exceeds a single machine.

**Q18. How does Polars interoperate with Arrow?**
Its in-memory format *is* Arrow, so `to_arrow` / `from_arrow` are largely zero-copy, enabling cheap exchange with DuckDB, PyArrow, and Parquet.

**Q19. Explain `group_by_dynamic` vs `rolling`.**
`group_by_dynamic` creates fixed, non-overlapping (or strided) time buckets; `rolling` creates a window anchored at each row.

**Q20. How do you extend Polars?**
Namespaces via `register_expr_namespace`, `map_batches` with NumPy, or Rust plugins using `pyo3-polars`.

</details>

---

## 31. 🟢 Resources

| Resource | Link |
|----------|------|
| 📘 Official User Guide | https://docs.pola.rs/ |
| 📗 Python API Reference | https://docs.pola.rs/api/python/stable/reference/ |
| 💻 GitHub | https://github.com/pola-rs/polars |
| 🧩 Awesome Polars | https://github.com/ddotta/awesome-polars |
| 🔌 pyo3-polars (plugins) | https://github.com/pola-rs/pyo3-polars |
| 📝 Release notes | https://github.com/pola-rs/polars/releases |
| 💬 Discord | https://discord.gg/4UfP5cfBE7 |

---

<div align="center">

**⭐ If these notes helped you, star the repo and keep building!**

![Made with](https://img.shields.io/badge/Made_with-❤️_for_AI%2FML_Engineers-red?style=flat-square)

</div>
