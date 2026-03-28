"""Wikipedia API services."""
from .pageviews_service import (
    PageviewsError,
    get_pageviews,
    get_page_info,
    get_entities,
    get_entities_pageviews,
    print_pageviews,
    get_monthly_average,
    get_monthly_average_fast,
    get_monthly_average_batch,
)

__all__ = [
    "PageviewsError",
    "get_pageviews",
    "get_page_info", 
    "get_entities",
    "get_entities_pageviews",
    "print_pageviews",
    "get_monthly_average",
    "get_monthly_average_fast",
    "get_monthly_average_batch",
]
