
import hashlib
import os
import sys
import time
import logging
import argparse
from typing import Generator, Any

from elasticsearch import Elasticsearch, helpers, NotFoundError
import dotenv
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Configuration
# Remote credentials (source)
ES_REMOTE_URL = "http://52.241.5.104:9200/" 
ES_REMOTE_USER = os.getenv("ELASTICSEARCH_USERNAME")
ES_REMOTE_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
REMOTE_INDEX = "wiki_full_l"

# Local credentials (destination)
ES_LOCAL_URL = "http://localhost:9200"
ES_LOCAL_USER = os.getenv("ELASTICSEARCH_USERNAME") # Using same creds as they seem shared/mirrored
ES_LOCAL_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
LOCAL_INDEX_NAME = "wiki_full_l_local" # New name for local copy

def get_es_client(url, user=None, password=None) -> Elasticsearch:
    """Create an Elasticsearch client."""
    if user and password:
        return Elasticsearch(url, basic_auth=(user, password), request_timeout=300)
    return Elasticsearch(url, request_timeout=300)

def migrate_index():
    """Migrate index from remote to local with specific optimization settings."""
    
    # 1. Connect to both clusters
    logger.info(f"Connecting to Remote: {ES_REMOTE_URL}")
    remote_client = get_es_client(ES_REMOTE_URL, ES_REMOTE_USER, ES_REMOTE_PASSWORD)
    if not remote_client.ping():
        logger.error("Could not connect to remote Elasticsearch!")
        return

    logger.info(f"Connecting to Local: {ES_LOCAL_URL}")
    local_client = get_es_client(ES_LOCAL_URL, ES_LOCAL_USER, ES_LOCAL_PASSWORD)
    if not local_client.ping():
        logger.error("Could not connect to local Elasticsearch!")
        return

    # 2. Get Remote Mapping to replicate structure
    logger.info(f"Fetching mapping for remote index: {REMOTE_INDEX}")
    try:
        mapping_response = remote_client.indices.get_mapping(index=REMOTE_INDEX)
        source_mapping = mapping_response[REMOTE_INDEX]["mappings"]
    except Exception as e:
        logger.error(f"Failed to get remote mapping: {e}")
        return

    # 3. Optimize Mapping for Local Storage
    # - Disable 'vector' in _source to save space
    # - Enable 'store: true' for 'vector' so it can still be retrieved if needed
    
    local_mapping = source_mapping.copy()
    
    # Configure _source exclusion
    if "_source" not in local_mapping:
        local_mapping["_source"] = {}
    
    # Ensure source excludes vector
    current_excludes = local_mapping["_source"].get("excludes", [])    
    if "vector" not in current_excludes:
        current_excludes.append("vector")
    local_mapping["_source"]["excludes"] = current_excludes
    
    # Configure vector field
    props = local_mapping.get("properties", {})
    if "vector" in props:
        # Note: 'store' parameter is not supported for dense_vector in 8.x/9.x
        # We rely on doc_values (which are enabled by default for dense_vector with index:true)
        # to allow retrieval later via docvalue_fields if needed.
        if "store" in props["vector"]:
            del props["vector"]["store"]
            
        props["vector"]["index"] = True 
        logger.info("Configured local mapping: _source excludes 'vector', relying on doc_values/index for retrieval")
    else:
        logger.warning("No 'vector' field found in remote mapping properties!")

    # 4. Create Local Index
    if local_client.indices.exists(index=LOCAL_INDEX_NAME):
        logger.warning(f"Local index '{LOCAL_INDEX_NAME}' already exists. deleting...")
        local_client.indices.delete(index=LOCAL_INDEX_NAME)
    
    logger.info(f"Creating local index '{LOCAL_INDEX_NAME}'...")
    local_client.indices.create(index=LOCAL_INDEX_NAME, mappings=local_mapping)
    
    # 5. Migrate Data (Scan & Bulk)
    logger.info("Starting migration...")
    
    # Get total count for progress bar
    try:
        # NOTE: Using count() can sometimes be slightly off or use a different context depending on permissions.
        # But 'cat indices' showed ~17.7M docs while 'count()' showed ~15.6M.
        # This usually means there are deletes or it's hitting a weird alias/routing issue.
        # Let's trust stats or cat over count() for the progress bar total if they differ.
        
        # Method 1: Standard count
        count_resp = remote_client.count(index=REMOTE_INDEX)
        total_docs = count_resp["count"]
        
        # Method 2: Check stats to be sure (as per user observation of ~17M)
        try:
            stats = remote_client.indices.stats(index=REMOTE_INDEX)
            primaries = stats["indices"][REMOTE_INDEX]["primaries"]["docs"]
            stats_count = primaries["count"]
            if stats_count > total_docs:
                logger.info(f"Stats count ({stats_count:,}) > Count API ({total_docs:,}). Using stats count.")
                total_docs = stats_count
        except:
            pass
            
        logger.info(f"Total documents to migrate: {total_docs:,}")
    except:
        total_docs = None

    # Generator for bulk helper — uses PIT + search_after to avoid scroll context expiry
    # Deduplicates by hashing the 'text' field (since wikipedia_id is not unique per passage)
    PAGE_SIZE = 1000
    PIT_KEEP_ALIVE = "10m"

    def doc_generator():
        seen_hashes: set[str] = set()
        skipped = 0

        # Open a Point-In-Time on the remote cluster — no expiring scroll context
        pit_resp = remote_client.open_point_in_time(index=REMOTE_INDEX, keep_alive=PIT_KEEP_ALIVE)
        pit_id = pit_resp["id"]

        logger.info(f"Opened PIT: {pit_id[:40]}...")

        search_after: list | None = None
        page = 0

        try:
            while True:
                body: dict = {
                    "size": PAGE_SIZE,
                    "query": {"match_all": {}},
                    "sort": [{"_shard_doc": "asc"}],  # deterministic, fast tiebreaker
                    "pit": {"id": pit_id, "keep_alive": PIT_KEEP_ALIVE},
                }
                if search_after is not None:
                    body["search_after"] = search_after

                resp = remote_client.search(body=body, _source=True)

                hits = resp["hits"]["hits"]
                if not hits:
                    break  # exhausted all docs

                # Update PIT id (ES may rotate it)
                pit_id = resp["pit_id"]
                page += 1

                for doc in hits:
                    source = doc["_source"]

                    # Deduplication: hash the text content (primary passage content)
                    raw_text = source.get("text") or source.get("content") or ""
                    # Include wikipedia_id in hash so identical text from different articles
                    # is NOT collapsed (different articles can legitimately share a sentence)
                    wiki_id = str(source.get("wikipedia_id", ""))
                    dedup_key = hashlib.md5(f"{wiki_id}\x00{raw_text}".encode("utf-8", errors="replace")).hexdigest()

                    if dedup_key in seen_hashes:
                        skipped += 1
                        continue
                    seen_hashes.add(dedup_key)

                    yield {
                        "_index": LOCAL_INDEX_NAME,
                        "_id": doc["_id"],
                        "_source": source,
                    }

                search_after = hits[-1]["sort"]

                if page % 50 == 0:
                    logger.info(f"  Page {page:,} — fetched ~{page * PAGE_SIZE:,} docs, "
                                f"skipped {skipped:,} duplicates so far")

        except Exception:
            logger.exception("Error during PIT pagination")
            raise
        finally:
            try:
                remote_client.close_point_in_time(body={"id": pit_id})
                logger.info(f"Closed PIT. Total duplicates skipped: {skipped:,}")
            except Exception:
                pass

    # Run bulk indexing
    success_count = 0
    try:
        progress = tqdm(total=total_docs, unit="docs", desc="Migrating")

        for success, info in helpers.streaming_bulk(
            local_client,
            doc_generator(),
            chunk_size=500,
            max_retries=5,
            initial_backoff=2,
            raise_on_error=False,
        ):
            if success:
                success_count += 1
                progress.update(1)
            else:
                logger.error(f"Failed to index doc: {info}")

        progress.close()

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return

    logger.info(f"Migration complete! {success_count:,} documents indexed to '{LOCAL_INDEX_NAME}'.")
    logger.info("Verifying vector storage...")
    
    # Verification
    time.sleep(1) # Allow refresh
    local_client.indices.refresh(index=LOCAL_INDEX_NAME)
    
    # Check one doc
    res = local_client.search(index=LOCAL_INDEX_NAME, size=1)
    if res["hits"]["hits"]:
        hit = res["hits"]["hits"][0]
        src = hit.get("_source", {})
        
        has_vector_in_source = "vector" in src
        logger.info(f"Verification - 'vector' in _source: {has_vector_in_source} (Should be False)")
        
        # Check retrieval via docvalue_fields (since stored_fields isn't supported for vector)
        dv_res = local_client.search(
            index=LOCAL_INDEX_NAME, 
            size=1, 
            docvalue_fields=["vector"],
            _source=False
        )
        if dv_res["hits"]["hits"]:
            dv_fields = dv_res["hits"]["hits"][0].get("fields", {})
            has_doc_values = "vector" in dv_fields
            logger.info(f"Verification - 'vector' in docvalue_fields: {has_doc_values} (Should be True)")
        else:
            has_doc_values = False
            logger.warning("Verification - Could not perform docvalue fetch")
        
        if not has_vector_in_source and has_doc_values:
            logger.info("SUCCESS: Space saving configuration works (vector not in source, but retrievable).")
        else:
            logger.warning(f"WARNING: Configuration verification failed (source: {has_vector_in_source}, dv: {has_doc_values})")

if __name__ == "__main__":
    migrate_index()
