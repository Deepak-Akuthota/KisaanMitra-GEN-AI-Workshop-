import os
from dotenv import load_dotenv
from google.adk import Agent
import google.generativeai as genai  # <-- FIX: Import genai

# --- sub-agents ---
from sub_agents.soil_agent.agent import SoilAgent
from sub_agents.weather_agent.agent import WeatherAgent
from sub_agents.market_agent.agent import MarketAgent
from sub_agents.farm_practices_agent.agent import FarmPracticesAgent
from sub_agents.farm_profile_agent.agent import FarmerProfileAgent

# load environment
load_dotenv()


class ManagerAgent(Agent):
    """
    🌾 Root (Orchestrator) Agent for KisaanMitra
    Coordinates all sub-agents and produces the final advisory report.
    """

    # allow extra attributes because ADK agents use Pydantic
    model_config = {"extra": "allow"}

    def __init__(self, **data):
        # env check
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_key:
            print("⚠️  GEMINI_API_KEY missing in .env. Agent will likely fail.")
            # Stop initialization if the key is missing
            raise ValueError("GEMINI_API_KEY is not set in the .env file.")

        # --- FIX: Configure genai and create the model instance ---
        try:
            genai.configure(api_key=self.gemini_key)
            # Use a specific model, e.g., "gemini-1.5-flash"
            llm = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            # Handle potential initialization errors
            raise Exception(f"Failed to initialize GenerativeModel: {e}")
        # -----------------------------------------------------------

        # --- FIX: Pass the 'llm' model to the super() constructor ---
        # The base Agent class requires the 'model' argument.
        super().__init__(name="root_manager_agent", model=llm, **data)
        # -----------------------------------------------------------

        self.supports_text_input = True
        
        # --- FIX: Pass the 'llm' model to all sub-agents ---
        # Your sub-agents also inherit from adk.Agent and need the model.
        self.soil_agent = SoilAgent(model=llm)
        self.weather_agent = WeatherAgent(model=llm)
        self.market_agent = MarketAgent(model=llm)
        self.practice_agent = FarmPracticesAgent(model=llm)
        self.farmer_profile_agent = FarmerProfileAgent(model=llm)
        # -----------------------------------------------------------

    # ------------------------------------------------------------------
    # called automatically when you type into ADK Web
    # ------------------------------------------------------------------
    def on_message(self, message: str):
        try:
            farmer_id = "farmer_001"
            location = "Andhra Pradesh"

            soil = self.soil_agent.analyze_soil()
            weather = self.weather_agent.get_forecast(location)
            market = self.market_agent.get_prices()
            advice = self.practice_agent.generate_advice(soil, weather, market)

            context = {"soil": soil, "weather": weather, "market": market}
            personalized = self.farmer_profile_agent.recommend_for_farmer(
                farmer_id, context
            )

            reply = (
                f"🌾 **KisaanMitra Report** 🌾\n\n"
                f"📍 *Region:* {location}\n\n"
                f"**Soil Data:** {soil}\n"
                f"**Weather:** {weather}\n"
                f"**Market:** {market}\n\n"
                f"🧠 *General Advice:* {advice}\n\n"
                f"🤝 *Personalized Recommendation:* "
                f"{personalized.get('recommendation', personalized)}"
            )
            return reply

        except Exception as e:
            return f"❌ Error: {e}"


# --- local CLI test ---
if __name__ == "__main__":
    try:
        agent = ManagerAgent()
        print(agent.on_message("Generate farm advice for farmer_001 in Andhra Pradesh"))
    except Exception as e:
        print(f"❌ Failed to run agent: {e}")

# --- register for ADK Web ---
# This instantiation should also be safe now
try:
    root_agent = ManagerAgent()
except Exception as e:
    print(f"❌ Failed to initialize root_agent for ADK Web: {e}")
    root_agent = None