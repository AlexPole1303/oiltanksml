"""Capture the daily contract data loaded by public TradingCharts year pages."""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

from collect_public import write

MONTH_CODES = "FGHJKMNQUVXZ"


async def collect_year(browser, year, output):
    expected = {f"CL{code}{year % 100:02d}" for code in MONTH_CODES}
    received = {}
    done = asyncio.Event()
    page = await browser.new_page()
    pending = []

    async def consume(response):
        if not response.ok:
            return
        try:
            payload = await response.json()
        except (ValueError, TypeError):
            return
        rows = payload.get("results") or []
        symbols = {row.get("symbol") for row in rows}
        if len(symbols) != 1 or not symbols <= expected:
            return
        symbol = symbols.pop()
        # The same page can also fetch a one-bar refresh for the selected chart.
        # Keep the fuller response for each exact dated symbol.
        if len(rows) > received.get(symbol, 0):
            write(output / f"{symbol}.json", json.dumps(rows).encode())
            received[symbol] = len(rows)
        if received.keys() == expected and min(received.values()) > 1:
            done.set()

    def handle(response):
        if "/getHistory.json" in response.url:
            pending.append(asyncio.create_task(consume(response)))

    page.on("response", handle)
    url = f"https://futures.tradingcharts.com/historical/CL_/{year}/C/barchart.html"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
        try:
            await asyncio.wait_for(done.wait(), 20)
        except TimeoutError:
            pass
        await asyncio.gather(*pending)
    finally:
        await page.close()
    thin = sorted(symbol for symbol,count in received.items() if count==1)
    print(f"{year}: {len(received)}/{len(expected)}; missing {', '.join(sorted(expected - received.keys())) or 'none'}; "
          f"single-bar responses {', '.join(thin) or 'none'}",
          flush=True)


async def collect(args):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=args.chrome, headless=True)
        slots = asyncio.Semaphore(2)

        async def limited(year):
            async with slots:
                await collect_year(browser, year, args.output)

        try:
            await asyncio.gather(*(limited(year) for year in args.years))
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("years", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrome", default="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    asyncio.run(collect(parser.parse_args()))
