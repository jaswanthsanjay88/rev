import base64
from fastapi.testclient import TestClient
from rev.serve import app
from rev.api import SystemOneRequest, to_record
from rev import Rev

client = TestClient(app)

def test_vision_schema_validation():
    # 1. Test top-level image field
    payload = {
        "image": "aW1hZ2VfZGF0YQ==",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Is this receipt from a restaurant?",
                "criteria": {"yes": "Restaurant or food service", "no": "Not a restaurant"}
            }
        }
    }
    req = SystemOneRequest.model_validate(payload)
    assert req.image == "aW1hZ2VfZGF0YQ=="
    rec, meta = to_record(req)
    assert rec["image"] == "aW1hZ2VfZGF0YQ=="
    assert len(rec["questions"]) == 1

def test_vision_http_endpoint():
    payload = {
        "image": "dGVzdF9pbWFnZQ==",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "Is this receipt from a restaurant?",
                "criteria": {"yes": "Restaurant or food service", "no": "Not a restaurant"}
            }
        }
    }
    resp = client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answers" in data
    assert "category" in data["answers"]
    assert data["answers"]["category"]["type"] == "choice"
    assert "choice" in data["answers"]["category"]
    assert "probabilities" in data["answers"]["category"]
    assert "yes" in data["answers"]["category"]["probabilities"]
    assert "no" in data["answers"]["category"]["probabilities"]
    assert data["model"] == "jaswanthsanjay88/rev-vision"

def test_mock_engine_with_vision():
    model = Rev.from_pretrained("mock")
    res = model.predict(
        image="dGVzdF9iNjQ=",
        questions={
            "category": {
                "type": "choice",
                "instructions": "Is this receipt from a restaurant?",
                "criteria": {"yes": "Restaurant or food service", "no": "Not a restaurant"}
            }
        }
    )
    assert "answers" in res
    assert "category" in res["answers"]
    assert res["answers"]["category"]["choice"] in ("yes", "no")
