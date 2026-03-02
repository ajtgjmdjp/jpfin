# jpfin

Unified CLI for Japanese equity factor analysis.

Integrates [japan-finance-factors](https://pypi.org/project/japan-finance-factors/), [edinet-mcp](https://pypi.org/project/edinet-mcp/), [stockprice-mcp](https://pypi.org/project/stockprice-mcp/), and [japan-finance-codes](https://pypi.org/project/japan-finance-codes/) into a single command.

## Install

```bash
pip install jpfin
```

## Usage

```bash
# Analyze a single ticker
jpfin analyze 7203

# Multiple tickers
jpfin analyze 7203 6758 9984

# JSON output
jpfin analyze 7203 --format json

# Specify fiscal year
jpfin analyze 7203 --year 2024
```

## Features

- **18 quantitative factors** across 5 categories (value, quality, momentum, risk, size)
- **Point-in-time (PIT) safety** — no lookahead bias
- **Auto-resolution** of ticker → EDINET code
- **Market cap** from yfinance merged into financial data
- **Table and JSON** output formats

## Environment Variables

- `EDINET_API_KEY` — Required for financial data from EDINET

## License

Apache-2.0
