# sub_agents/farmer_profile_agent/agent.py
import os
import json
from typing import Dict, Optional
from google.adk import Agent
from dotenv import load_dotenv
import requests
from pathlib import Path

# Load environment (GEMINI key)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # or GEMINI_SERVICE_TOKEN

# local storage for demo
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)
PROFILE_DB = DATA_DIR / "farmer_profiles.json"

# Initialize empty DB if missing
if not PROFILE_DB.exists():
    PROFILE_DB.write_text(json.dumps({}))


class FarmerProfileAgent(Agent):
    model_config = {"extra": "allow"}
    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__(name="farmer_profile_agent")
        self.storage_path = storage_path or PROFILE_DB

    # ----- Profile storage helpers -----
    def _read_db(self) -> Dict:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_db(self, data: Dict):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ----- Public API -----
    def get_profile(self, farmer_id: str) -> Optional[Dict]:
        db = self._read_db()
        return db.get(farmer_id)

    def create_or_update_profile(self, farmer_id: str, profile: Dict) -> Dict:
        """
        profile example:
        {
          "name": "Ram",
          "age": 45,
          "location": "Kurnool, Andhra Pradesh",
          "farm_size_acres": 2.5,
          "soil_type": "loamy",
          "irrigation": "drip",
          "preferred_crops": ["groundnut"],
          "budget": 20000
        }
        """
        db = self._read_db()
        db[farmer_id] = profile
        self._write_db(db)
        return db[farmer_id]

    def delete_profile(self, farmer_id: str) -> bool:
        db = self._read_db()
        if farmer_id in db:
            del db[farmer_id]
            self._write_db(db)
            return True
        return False

    # ----- Gemini integration for personalised suggestion -----
    def _call_gemini(self, prompt: str, model: str = "models/text-bison-001") -> str:
        """
        Generic wrapper for calling Gemini-like Generative API.
        This example uses a simple POST to a (placeholder) generative endpoint.
        Adjust endpoint & auth method to match your actual Gemini / PaLM setup.
        """
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set in .env")

        # NOTE: Change endpoint & payload to match the exact API you are using.
        # Example endpoint style for generative language:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta2/{model}:generate?key={GEMINI_API_KEY}"

        payload = {
            "prompt": {
                "text": prompt
            },
            "temperature": 0.2,
            "maxOutputTokens": 512
        }

        headers = {"Content-Type": "application/json"}
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        j = resp.json()

        # Response parsing depends on the real API; this is a common shape:
        # - For Google's generative language API responses vary; adapt as needed
        # Try multiple common keys to be resilient:
        text = ""
        if "candidates" in j and isinstance(j["candidates"], list):
            text = j["candidates"][0].get("output", "")
        elif "output" in j:
            text = j["output"].get("text", "")
        else:
            # fallback: flatten known fields
            text = json.dumps(j)
        return text

    def recommend_for_farmer(self, farmer_id: str, context: Dict) -> Dict:
        """
        context: dictionary you can pass from Manager that contains
                 outputs from soil_agent, weather_agent, market_agent, etc.
        Returns a dict with 'recommendation' and 'explaination' fields.
        """
        profile = self.get_profile(farmer_id)
        if not profile:
            return {"error": f"No profile found for farmer_id={farmer_id}"}

        # build prompt with profile + context
        prompt_parts = [
            f"Farmer profile: {json.dumps(profile, ensure_ascii=False)}",
            f"Context: {json.dumps(context, ensure_ascii=False)}",
            "Using the above info, provide a short farming recommendation tailored to the farmer."
            " Include 2-3 practical steps, water/fertilizer considerations, and a short rationale."
        ]
        prompt = "\n\n".join(prompt_parts)

        try:
            output = self._call_gemini(prompt)
        except Exception as e:
            return {"error": f"Failed to call Gemini: {e}"}

        return {"recommendation": output}

# Example quick manual test
if __name__ == "__main__":
    a = FarmerProfileAgent()
    farmer_id = "farmer_001"
    profile = {
        "name": "Raju",
        "age": 38,
        "location": "Guntur, AP",
        "farm_size_acres": 1.8,
        "soil_type": "sandy loam",
        "irrigation": "borewell",
        "preferred_crops": ["chilli", "cotton"],
        "budget": 15000
    }
    a.create_or_update_profile(farmer_id, profile)
    ctx = {
        "soil": {"pH": 6.3, "moisture": "low"},
        "weather": {"condition": "dry", "week_rain_mm": 2},
        "market": {"chilli": "₹160/kg", "cotton": "₹58/kg"}
    }
    print(a.recommend_for_farmer(farmer_id, ctx))
