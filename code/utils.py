"""Web content extraction, screenshots, and monotonic task timing."""

import time

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def url_to_txt(url, output_filename):
    """Save visible body text, raising an HTTP error for unsuccessful requests."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, headers=headers, timeout=10)
    # Do not save an HTTP error page as if it were useful website content.
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Exclude navigation and page code from the text sent to the model.
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    main_content = soup.body
    if main_content is None:
        raise ValueError(f"Could not find body content at {url}")
    plain_text = main_content.get_text(separator="\n", strip=True)
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(plain_text)
    return plain_text


def save_website_as_image(url, output_filename="website.png"):
    """Capture a full page with common consent overlays hidden."""
    from playwright.sync_api import sync_playwright

    # Start the Playwright context
    with sync_playwright() as p:
        # Launch a headless Chrome browser (runs in the background)
        browser = p.chromium.launch(headless=True)

        # Open a new browser tab
        page = browser.new_page()

        print(f"Loading {url}...")
        # Go to the URL. 'networkidle' waits until the page is fully loaded (including most images/scripts)
        page.goto(url, wait_until="networkidle")

        print("Hiding cookie banners...")
        # Inject CSS to hide common cookie popups and sticky headers
        page.add_style_tag(
            content="""
            /* Target common cookie/consent banner IDs and Classes */
            [id*='cookie' i], [class*='cookie' i], 
            [id*='consent' i], [class*='consent' i], 
            [id*='banner' i], [class*='banner' i],
            div[role='dialog'] { 
                display: none !important; 
                opacity: 0 !important;
                visibility: hidden !important;
                z-index: -1 !important;
            }
        """
        )

        print("Taking screenshot...")
        # Take the screenshot. full_page=True captures the whole scrolling page, not just what fits on a screen
        page.screenshot(path=output_filename, full_page=True)

        # Close the browser
        browser.close()
        print(f"Saved successfully as {output_filename}")


class TaskTimer:
    """Timer for tracking task execution time"""

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer"""
        # A monotonic clock is unaffected by system clock adjustments.
        self.start_time = time.perf_counter()
        # Clear the previous stop value when the same timer is reused.
        self.end_time = None

    def stop(self):
        """Stop the timer and return elapsed time"""
        self.end_time = time.perf_counter()
        return self.elapsed()

    def elapsed(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time is not None else time.perf_counter()
        return end - self.start_time
