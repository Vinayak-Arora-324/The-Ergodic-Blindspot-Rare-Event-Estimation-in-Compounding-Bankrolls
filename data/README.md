# Data cache

`blindspot.sp500_memory` stores downloaded S&P 500 prices here. CSV files are
ignored because they are reproducible caches rather than source artifacts.

Downloaded prices are validated before an atomic cache replacement. Empty or
invalid downloads and interrupted writes preserve the existing cache.
Both downloads and cached reads include `--start` and exclude `--end`.
An empty selection fails with a message describing the requested date range.
