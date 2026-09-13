"""Capture the README screenshots from the running app.

Prerequisites:
    pip install playwright
    (no browser download needed - it drives your installed Chrome)

Then, with the app running on http://localhost:8501:
    .venv\\Scripts\\python.exe docs/capture_screenshots.py

Writes docs/img/01-overview.png through 04-payload.png, the filenames the
README already references.
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).parent / "img"
URL = "http://localhost:8501"

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("playwright isn't installed. Run:  pip install playwright")


def settle(page, ms: int = 2500) -> None:
    """Streamlit renders over a websocket, so 'load' fires long before the UI
    is actually painted. Wait for the app frame, then give it a beat."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30000)
    page.wait_for_timeout(ms)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        # channel="chrome" reuses the installed browser instead of downloading
        # Playwright's own ~150MB Chromium build.
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000})

        page.goto(URL, wait_until="networkidle")
        settle(page, 4000)

        # 1. The whole app
        page.screenshot(path=OUT / "01-overview.png")
        print("wrote 01-overview.png")

        # 2. Just the sidebar, where the parameter controls live
        sidebar = page.query_selector('[data-testid="stSidebar"]')
        if sidebar:
            sidebar.screenshot(path=OUT / "02-parameters.png")
            print("wrote 02-parameters.png")

        # 3. The help popover, opened
        try:
            page.get_by_text("ℹ️", exact=True).first.click()
            page.wait_for_timeout(1200)
            page.screenshot(path=OUT / "03-help.png")
            print("wrote 03-help.png")
            page.keyboard.press("Escape")
            page.wait_for_timeout(600)
        except Exception as exc:  # noqa: BLE001
            print(f"skipped 03-help.png: {exc}")

        # 4. The payload expander, opened
        try:
            page.get_by_text("Exactly what gets sent").first.click()
            page.wait_for_timeout(1200)
            sidebar = page.query_selector('[data-testid="stSidebar"]')
            (sidebar or page).screenshot(path=OUT / "04-payload.png")
            print("wrote 04-payload.png")
        except Exception as exc:  # noqa: BLE001
            print(f"skipped 04-payload.png: {exc}")

        browser.close()

    print(f"\nDone - see {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
