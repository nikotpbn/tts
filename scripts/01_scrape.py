import os
import time
import random
import json
from playwright.sync_api import sync_playwright

# Configuration
# Base URL without the page parameter
BASE_URL = "https://wago.tools/files?search=sound%2Fcreature%2Fthrall"
OUTPUT_DIR path to os.path.join(PROJECT_ROOT, "data", "raw", "thrall")
MAP_FILE = os.path.join(OUTPUT_DIR, "download_map.json")
MAX_DOWNLOADS = 50

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_download_map():
    if os.path.exists(MAP_FILE):
        try:
            with open(MAP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_download_map(download_map):
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(download_map, f, indent=4)

def scrape_audio():
    download_map = load_download_map()
    
    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=False) 
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        
        success_count = 0
        current_page = 1
        
        while success_count < MAX_DOWNLOADS:
            # Inject the current page number into the URL
            page_url = f"{BASE_URL}&page={current_page}"
            print(f"\n--- Navigating to Page {current_page} ---")
            print(f"URL: {page_url}")
            page.goto(page_url)
            
            print("Waiting for file rows to load...")
            # Wait for the table to render
            try:
                page.wait_for_selector("table tbody tr", timeout=15000)
                time.sleep(3) # Extra buffer for the API to attach download links
            except Exception:
                print(f"No table found on page {current_page}. We might have reached the end of the results.")
                break
            
            rows = page.locator("table tbody tr").all()
            print(f"Found {len(rows)} potential files on page {current_page}.")
            
            if len(rows) == 0:
                print("No rows found. Stopping pagination.")
                break

            for index, row in enumerate(rows):
                if success_count >= MAX_DOWNLOADS:
                    print(f"\nReached target cap of {MAX_DOWNLOADS} downloads. Stopping script.")
                    break
                    
                try:
                    # Target the anchor tag containing the API download endpoint
                    download_btn = row.locator("a[href*='/api/'][href*='download']").first
                    
                    if not download_btn.is_visible():
                        continue
                    
                    # Extract the relative download link (href) to use as a unique key
                    href = download_btn.get_attribute("href")
                    
                    if not href:
                        continue
                    
                    # Check against our local ledger to see if we've handled this resource
                    if href in download_map:
                        existing_file = os.path.join(OUTPUT_DIR, download_map[href])
                        if os.path.exists(existing_file):
                            print(f"[{success_count + 1}] Skipping (Already downloaded: {download_map[href]})")
                            continue
                    
                    # Trigger and intercept the download pipeline
                    with page.expect_download(timeout=5000) as download_info:
                        download_btn.click()
                    
                    download = download_info.value
                    file_name = download.suggested_filename
                    
                    if file_name.endswith((".ogg", ".mp3", ".wav")):
                        file_path = os.path.join(OUTPUT_DIR, file_name)
                        download.save_as(file_path)
                        
                        # Log the successful acquisition in our map
                        download_map[href] = file_name
                        save_download_map(download_map)
                        
                        success_count += 1
                        print(f"[{success_count}/{MAX_DOWNLOADS}] Successfully downloaded: {file_name}")
                        
                        # Implement polite rate-limiting behavior if we aren't done yet
                        if success_count < MAX_DOWNLOADS:
                            delay = random.randint(20, 60)
                            print(f"Waiting for {delay} seconds to safeguard rate limits...")
                            time.sleep(delay)
                            
                except Exception as e:
                    print(f"[{index + 1}] Skipped row due to timeout or navigation friction.")
            
            # If we finish the rows on this page but still need more files, increment the page
            if success_count < MAX_DOWNLOADS:
                current_page += 1
                
        print(f"\nBatch cycle complete. Downloaded {success_count} files into '{OUTPUT_DIR}'.")
        browser.close()

if __name__ == "__main__":
    scrape_audio()