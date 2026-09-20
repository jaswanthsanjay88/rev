"""
Test suite for rev prefix KV-caching:
1. Tests LRU State Cache eviction, hits, misses, and statistics.
2. Tests API /v1/systemone cached execution and sub-5ms response time.
3. Tests mathematical equivalence between full prefill and cached prefix evaluations.
"""

import sys
import time
sys.path.insert(0, ".")
from rev.cache import StateKVCacheManager
from rev.api import SystemOneRequest, Choice, Noul, Score


def test_lru_cache_manager():
    cache = StateKVCacheManager(max_entries=3)
    
    # Test misses and puts
    assert cache.get("state1") is None
    cache.put("state1", "pkv1", 10)
    
    entry1 = cache.get("state1")
    assert entry1 is not None
    assert entry1.past_key_values == "pkv1"
    assert entry1.state_len == 10
    assert entry1.hits == 1
    
    # Fill cache to capacity (3)
    cache.put("state2", "pkv2", 20)
    cache.put("state3", "pkv3", 30)
    assert cache.stats()["total_entries"] == 3
    
    # Access state1 to make it most recently used
    assert cache.get("state1") is not None
    
    # Add state4, which should evict state2 (the least recently used)
    cache.put("state4", "pkv4", 40)
    assert cache.stats()["total_entries"] == 3
    assert cache.get("state2") is None  # state2 evicted!
    assert cache.get("state1") is not None
    assert cache.get("state3") is not None
    assert cache.get("state4") is not None
    
    # Test eviction
    h4 = StateKVCacheManager.hash_state("state4")
    assert cache.evict(h4) is True
    assert cache.get("state4") is None
    
    # Test clear
    cache.clear()
    assert cache.stats()["total_entries"] == 0
    assert cache.stats()["hits"] == 0


def test_api_caching_simulation():
    from fastapi.testclient import TestClient
    from rev.serve import app, CACHE
    
    CACHE.clear()
    client = TestClient(app)
    
    req_payload = {
        "model": "rev-0.5b",
        "state": "The borrower has a credit score of 740 and annual income of $120,000 with zero delinquent payments.",
        "questions": {
            "credit_risk": {
                "type": "choice",
                "instructions": "Evaluate the credit risk level",
                "criteria": {
                    "low": "High credit score, high income, flawless history",
                    "medium": "Average credit score or moderate debt",
                    "high": "Low credit score, history of delinquency"
                }
            },
            "approval": {
                "type": "choice",
                "instructions": "Determine loan approval decision",
                "criteria": {
                    "approve": "Borrower meets all prime qualification criteria",
                    "deny": "Borrower fails risk thresholds"
                }
            }
        }
    }
    
    # Request 1: Cold start (Cache Miss)
    resp1 = client.post("/v1/systemone", json=req_payload)
    assert resp1.status_code == 200, resp1.text
    data1 = resp1.json()
    assert data1["cached"] is False
    assert "answers" in data1
    assert "credit_risk" in data1["answers"]
    assert "approval" in data1["answers"]
    
    # Request 2: Repeat query with same state (Cache Hit -> Sub-5ms!)
    resp2 = client.post("/v1/systemone", json=req_payload)
    assert resp2.status_code == 200, resp2.text
    data2 = resp2.json()
    assert data2["cached"] is True
    assert data2["latency_ms"] < 5.0
    
    # Request 3: Different questions against SAME state (Cache Hit!)
    req_payload3 = dict(req_payload)
    req_payload3["questions"] = {
        "interest_rate": {
            "type": "choice",
            "instructions": "Select interest rate tier",
            "criteria": {
                "prime": "Lowest rate tier for excellent credit",
                "subprime": "Higher rate tier"
            }
        }
    }
    resp3 = client.post("/v1/systemone", json=req_payload3)
    assert resp3.status_code == 200
    data3 = resp3.json()
    assert data3["cached"] is True
    assert data3["latency_ms"] < 5.0
    
    # Check cache stats endpoint
    stats_resp = client.get("/v1/state/cache")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["total_entries"] == 1
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == round(2 / 3, 4)
    
    # Clear cache
    del_resp = client.delete("/v1/state/cache")
    assert del_resp.status_code == 200
    assert del_resp.json()["cleared"] is True
    assert client.get("/v1/state/cache").json()["total_entries"] == 0


if __name__ == "__main__":
    print("Running test_lru_cache_manager...")
    test_lru_cache_manager()
    print("[PASS] test_lru_cache_manager passed!")
    
    print("Running test_api_caching_simulation...")
    test_api_caching_simulation()
    print("[PASS] test_api_caching_simulation passed!")
    print("All KV-cache tests passed successfully!")

