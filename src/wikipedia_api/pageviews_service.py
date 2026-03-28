"""Wikipedia pageviews service."""
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
from urllib.parse import quote
import wikipediaapi

WIKI = wikipediaapi.Wikipedia(user_agent='PopularityBias/1.0', language='en')
BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"

# Rate limiting settings
MAX_CONCURRENT_REQUESTS = 10  # Max parallel requests
RETRY_ATTEMPTS = 5  # Number of retries on 429
RETRY_BASE_DELAY = 2  # Base delay in seconds (doubles each retry)


class PageviewsError(Exception):
    """Exception raised when pageviews API fails."""
    def __init__(self, title: str, message: str, status_code: int = None):
        self.title = title
        self.status_code = status_code
        super().__init__(f"Pageviews error for '{title}': {message}" + (f" (HTTP {status_code})" if status_code else ""))


class RateLimiter:
    """Rate limiter for API requests with retry logic."""
    
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_REQUESTS):
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute(self, coro):
        """Execute a coroutine with rate limiting."""
        async with self._semaphore:
            return await coro


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def _request_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    title: str,
    context: str = ""
) -> dict:
    """Make an HTTP request with retry logic for rate limiting."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with session.get(url, headers={"User-Agent": "PopularityBias/1.0"}) as r:
                if r.status == 200:
                    return await r.json()
                elif r.status == 429:
                    # Rate limited - wait and retry
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise PageviewsError(title, f"Rate limited after {RETRY_ATTEMPTS} retries{context}", 429)
                else:
                    error_text = await r.text()
                    raise PageviewsError(title, f"API error{context}: {error_text[:200]}", r.status)
        except aiohttp.ClientError as e:
            if attempt < RETRY_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_BASE_DELAY)
                continue
            raise PageviewsError(title, f"Network error{context}: {str(e)}")
    
    raise PageviewsError(title, f"Failed after {RETRY_ATTEMPTS} attempts{context}")


async def get_pageviews(title: Optional[str] = None, id: Optional[int] = None, start: datetime = None, end: datetime = None) -> int:
    """Get pageviews for a single Wikipedia page by title or ID."""
    if not title and not id:
        raise PageviewsError("", "Either 'title' or 'id' must be provided")
    
    # Use title if provided, otherwise use id
    identifier = title.replace(' ', '_') if title else str(id)
    url = f"{BASE_URL}/{quote(identifier, safe='')}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "PopularityBias/1.0"}) as r:
                if r.status == 200:
                    data = await r.json()
                    return sum(item.get("views", 0) for item in data.get("items", []))
                else:
                    error_text = await r.text()
                    raise PageviewsError(title or str(id), f"API returned error: {error_text}", r.status)
    except aiohttp.ClientError as e:
        raise PageviewsError(title or str(id), f"Network error: {str(e)}")


async def get_page_info(title: Optional[str] = None, id: Optional[int] = None, days: int = 30) -> Dict:
    """Get info and pageviews for a single Wikipedia page by title or ID."""
    if not title and not id:
        raise PageviewsError("", "Either 'title' or 'id' must be provided")
    
    page = None
    page_id = id
    
    # If title provided, get page and ID
    if title:
        page = WIKI.page(title)
        if not page.exists():
            return {"error": f"Page '{title}' not found"}
        
        # Get page ID from Wikipedia API
        try:
            normalized_title = title.replace(' ', '_')
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={quote(normalized_title, safe='')}&format=json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers={"User-Agent": "PopularityBias/1.0"}) as r:
                    if r.status == 200:
                        data = await r.json()
                        pages = data.get('query', {}).get('pages', {})
                        if pages:
                            for pid, page_data in pages.items():
                                if 'missing' not in page_data:
                                    page_id = int(pid)
                                    break
        except Exception:
            pass
    else:
        # If only ID provided, fetch page info from Wikipedia API
        try:
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&pageids={id}&prop=info|extracts&format=json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, headers={"User-Agent": "PopularityBias/1.0"}) as r:
                    if r.status == 200:
                        data = await r.json()
                        pages = data.get('query', {}).get('pages', {})
                        page_data = pages.get(str(id))
                        if page_data and 'missing' not in page_data:
                            title = page_data.get('title')
                            page = WIKI.page(title)
                        else:
                            return {"error": f"Page with ID '{id}' not found"}
        except Exception as e:
            return {"error": f"Failed to fetch page with ID '{id}': {str(e)}"}
    
    end = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    views = await get_pageviews(title=page.title if page else title, id=page_id, start=start, end=end)
    
    return {
        "title": page.title if page else title,
        "pageid": page_id,
        "url": page.fullurl if page else None,
        "summary": (page.summary[:500] + "..." if len(page.summary) > 500 else page.summary) if page else None,
        "pageviews": views,
        "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
    }


async def get_entities(
    search: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> List[str]:
    """Get Wikipedia entities by search, category, or top viewed pages across a date range."""
    if search:
        return list(WIKI.search(search, results=limit)) if hasattr(WIKI, 'search') else []
    if category:
        cat = WIKI.page(f"Category:{category}")
        return [p.title for p in list(cat.categorymembers.values())[:limit] if p.ns == 0] if cat.exists() else []
    
    # Sample dates across the period to get comprehensive top pages
    if not end:
        end = datetime.utcnow() - timedelta(days=1)
    if not start:
        start = end - timedelta(days=30)
    
    # Sample one date per year in the range, plus start and end
    sample_dates = set()
    current = start
    while current <= end:
        sample_dates.add(datetime(current.year, 1, 15))  # Mid-January each year
        sample_dates.add(datetime(current.year, 7, 15))  # Mid-July each year
        current = datetime(current.year + 1, 1, 1)
    sample_dates.add(start)
    sample_dates.add(end)
    
    # Filter to only dates within range and not in future
    now = datetime.utcnow()
    sample_dates = [d for d in sample_dates if start <= d <= end and d < now]
    sample_dates = sorted(sample_dates)[:10]  # Max 10 samples
    
    # Fetch top pages from each sample date
    all_entities = set()
    async with aiohttp.ClientSession() as session:
        for target_date in sample_dates:
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{target_date.strftime('%Y/%m/%d')}"
            try:
                async with session.get(url, headers={"User-Agent": "PopularityBias/1.0"}) as r:
                    if r.status == 200:
                        data = await r.json()
                        if "items" in data and data["items"]:
                            articles = data["items"][0].get("articles", [])
                            for a in articles[:limit]:
                                if ":" not in a["article"] and a["article"] != "Main_Page":
                                    all_entities.add(a["article"])
            except:
                continue
    
    return list(all_entities)[:limit]


async def get_entities_pageviews(
    entities: List[str],
    start: datetime,
    end: datetime
) -> Dict[str, int]:
    """Get pageviews for multiple entities."""
    async def fetch(session, title):
        url = f"{BASE_URL}/{quote(title.replace(' ', '_'), safe='')}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
        try:
            async with session.get(url, headers={"User-Agent": "PopularityBias/1.0"}) as r:
                if r.status == 200:
                    data = await r.json()
                    return (title, sum(item.get("views", 0) for item in data.get("items", [])))
                else:
                    error_text = await r.text()
                    raise PageviewsError(title, f"API returned error: {error_text}", r.status)
        except aiohttp.ClientError as e:
            raise PageviewsError(title, f"Network error: {str(e)}")
    
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[fetch(session, e) for e in entities])
    return dict(sorted(results, key=lambda x: x[1], reverse=True))


async def print_pageviews(
    search: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 50,
    days: int = 30,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None
) -> None:
    """Fetch and print TOP pageviews for entities (sorted by actual views)."""
    # Determine date range first
    if not end:
        end = datetime.utcnow()
    if not start:
        start = end - timedelta(days=days)
    
    print("\nFetching candidates...")
    # Get MORE candidates than needed (5x limit), then filter to actual top N
    candidates = await get_entities(search, category, limit * 5, start=start, end=end)
    
    if not candidates:
        print("No entities found.")
        return
    
    print(f"Found {len(candidates)} candidates. Fetching pageviews ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})...\n")
    
    # Get pageviews for all candidates
    all_results = await get_entities_pageviews(candidates, start, end)
    
    # Take only top N by pageviews
    top_results = dict(list(all_results.items())[:limit])
    
    print(f"{'Entity':<40} {'Views':>12}")
    print("-" * 52)
    for title, views in top_results.items():
        print(f"{title:<40} {views:>12,}")
    print("-" * 52)
    print(f"{'TOTAL':<40} {sum(top_results.values()):>12,}\n")


async def get_monthly_average(
    title: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    year: Optional[int] = None
) -> Dict[str, Any]:
    """Get monthly average pageviews for a page between two dates or for a year."""
    now = datetime.utcnow()
    
    # Determine date range
    if start and end:
        pass  # Use provided dates
    elif year:
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
    else:
        # Default: last 12 months
        end = now
        start = datetime(now.year - 1, now.month, 1)
    
    # Don't go into future
    if end > now:
        end = now
    
    # Generate list of months in range
    monthly_views = []
    current = datetime(start.year, start.month, 1)
    
    async with aiohttp.ClientSession() as session:
        while current <= end:
            month_start = current
            if current.month == 12:
                month_end = datetime(current.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = datetime(current.year, current.month + 1, 1) - timedelta(days=1)
            
            # Clip to actual range
            if month_start < start:
                month_start = start
            if month_end > end:
                month_end = end
            if month_end > now:
                month_end = now
            
            url = f"{BASE_URL}/{quote(title.replace(' ', '_'), safe='')}/daily/{month_start.strftime('%Y%m%d')}/{month_end.strftime('%Y%m%d')}"
            
            # Use retry logic for rate limiting
            data = await _request_with_retry(session, url, title, f" for {current.strftime('%B %Y')}")
            views = sum(item.get("views", 0) for item in data.get("items", []))
            days_in_period = (month_end - month_start).days + 1
            monthly_views.append({
                "month": current.strftime("%B %Y"),
                "total": views,
                "daily_avg": round(views / days_in_period) if days_in_period > 0 else 0
            })
            
            # Move to next month
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
    
    total = sum(m["total"] for m in monthly_views)
    return {
        "title": title,
        "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
        "months": monthly_views,
        "total_views": total,
        "monthly_avg": round(total / len(monthly_views)) if monthly_views else 0,
        "daily_avg": round(total / ((end - start).days + 1)) if (end - start).days > 0 else 0
    }


async def get_monthly_average_fast(
    title: str,
    start: datetime,
    end: datetime
) -> Dict[str, Any]:
    """
    Get monthly average pageviews with a SINGLE API call (much faster).
    Fetches daily data for full range and calculates monthly stats locally.
    """
    url = f"{BASE_URL}/{quote(title.replace(' ', '_'), safe='')}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
    
    async with aiohttp.ClientSession() as session:
        data = await _request_with_retry(session, url, title)
    
    items = data.get("items", [])
    if not items:
        return {
            "title": title,
            "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
            "months": [],
            "total_views": 0,
            "monthly_avg": 0,
            "daily_avg": 0
        }
    
    # Group daily views by month
    monthly_data: Dict[str, int] = {}
    for item in items:
        timestamp = item.get("timestamp", "")
        views = item.get("views", 0)
        if len(timestamp) >= 6:
            month_key = timestamp[:6]  # YYYYMM
            monthly_data[month_key] = monthly_data.get(month_key, 0) + views
    
    # Convert to monthly stats
    monthly_views = []
    for month_key in sorted(monthly_data.keys()):
        year = int(month_key[:4])
        month = int(month_key[4:6])
        month_date = datetime(year, month, 1)
        views = monthly_data[month_key]
        
        # Calculate days in this month from the data
        days_count = sum(1 for item in items if item.get("timestamp", "").startswith(month_key))
        
        monthly_views.append({
            "month": month_date.strftime("%B %Y"),
            "total": views,
            "daily_avg": round(views / days_count) if days_count > 0 else 0
        })
    
    total = sum(m["total"] for m in monthly_views)
    return {
        "title": title,
        "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
        "months": monthly_views,
        "total_views": total,
        "monthly_avg": round(total / len(monthly_views)) if monthly_views else 0,
        "daily_avg": round(total / len(items)) if items else 0
    }


async def get_monthly_average_batch(
    titles: List[str],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    year: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, str], None]] = None
) -> List[Dict[str, Any]]:
    """
    Get monthly average pageviews for multiple articles with rate limiting.
    Uses FAST method: 1 API call per article instead of 12.
    
    Args:
        titles: List of article titles
        start: Start date
        end: End date
        year: Year (alternative to start/end)
        on_progress: Optional callback(completed, total, current_title)
    
    Returns:
        List of results in same order as titles
    """
    now = datetime.utcnow()
    
    # Determine date range
    if year:
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
    elif not start or not end:
        end = now
        start = datetime(now.year - 1, now.month, 1)
    
    if end > now:
        end = now
    
    rate_limiter = get_rate_limiter()
    results = [None] * len(titles)
    completed = [0]  # Use list to allow modification in nested function
    lock = asyncio.Lock()
    
    async def fetch_one(idx: int, title: str):
        async with rate_limiter._semaphore:
            try:
                result = await get_monthly_average_fast(title, start, end)
                results[idx] = result
            except Exception as e:
                results[idx] = e
            
            async with lock:
                completed[0] += 1
                if on_progress:
                    on_progress(completed[0], len(titles), title)
    
    # Process in batches
    batch_size = MAX_CONCURRENT_REQUESTS * 3
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        batch_tasks = [fetch_one(i + j, title) for j, title in enumerate(batch)]
        await asyncio.gather(*batch_tasks)
        
        # Small delay between batches to be nice to the API
        if i + batch_size < len(titles):
            await asyncio.sleep(0.3)
    
    # Check for errors
    errors = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(f"{titles[idx]}: {result}")
    
    if errors and len(errors) > len(titles) * 0.1:  # More than 10% failed
        raise PageviewsError("batch", f"{len(errors)}/{len(titles)} articles failed. First errors: {errors[:3]}")
    
    # Replace errors with empty results
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            results[idx] = {
                "title": titles[idx],
                "period": f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
                "months": [],
                "total_views": 0,
                "monthly_avg": 0,
                "daily_avg": 0,
                "error": str(result)
            }
    
    return results
