import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from markdownify import markdownify as md
import sys
from playwright.sync_api import sync_playwright


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}


def url_to_txt(url, output_filename):
    # 1. Fetch the website content
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        print(f"Successfully accessed {url}")
    else:
        print(f"Failed to access {url} with status code {response.status_code}", file=sys.stderr)
    
    # 2. Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 3. Clean up the HTML (Remove navigation, footer, scripts, and styles)
    # This prevents your content from being filled with website code/menus
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
        
    # 4. Fix relative image URLs to make them absolute
    # E.g., changes <img src="/pic.jpg"> to <img src="https://website.com/pic.jpg">
    for img in soup.find_all('img'):
        if img.get('src'):
            img['src'] = urljoin(url, img.get('src'))
            
    # 5. Extract only the main article content if you don't want the whole page
    # main_content = soup.find('article') or soup.find('main') or soup.body
    main_content = soup.body

    if not main_content:
        return "Could not find body content."
    
    # # 6. Convert the cleaned HTML to Markdown
    # # heading_style="ATX" uses '#' for headings instead of underlines
    # markdown_text = md(str(main_content), heading_style="ATX")
    
    # # Clean up excessive blank lines
    # markdown_text = '\n'.join([line for line in markdown_text.splitlines() if line.strip() != ''])

    # Options: You can also return the plain text if you want to feed it to an LLM without markdown formatting
    plain_text = soup.get_text(separator="\n", strip=True)
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(plain_text)
    return plain_text


def save_website_as_image(url, output_filename="../data/webwalkerqa/website.png"):
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
        page.add_style_tag(content="""
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
        """)
        
        print(f"Taking screenshot...")
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
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer and return elapsed time"""
        self.end_time = time.time()
        return self.elapsed()
    
    def elapsed(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
