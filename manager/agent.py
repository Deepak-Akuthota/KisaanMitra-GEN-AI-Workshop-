import os
from dotenv import load_dotenv
from google.adk import Agent
import google.generativeai as genai
from typing import Any
import traceback  # Import traceback for better error logging

# --- sub-agents ---
# (Make sure these paths are correct relative to your root)
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

    # --- THIS IS THE CORRECTED __init__ METHOD ---
    def __init__(self, *, model: genai.GenerativeModel | None = None, **data: Any):
        """
        Initializes the ManagerAgent.
        This version fixes the Pydantic error by loading the API key
        into a local variable first, before calling super().
        """
        
        # --- 1. Load API key into a LOCAL variable (no 'self.') ---
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            print("⚠️  GEMINI_API_KEY missing in .env. Agent will likely fail.")
            raise ValueError("GEMINI_API_KEY is not set in the .env file.")

        # --- 2. Configure genai and create the model ---
        try:
            genai.configure(api_key=gemini_key) # Use the local variable
            llm = genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            raise Exception(f"Failed to initialize GenerativeModel: {e}")
        
        # --- 3. Set up data for super() ---
        data['name'] = "root_manager_agent"

        # --- 4. Call super() FIRST ---
        # This initializes Pydantic and creates '__pydantic_extra__'
        super().__init__(model=llm, **data)

        # --- 5. Now it is SAFE to set custom attributes ---
        self.supports_text_input = True
        self.soil_agent = SoilAgent(model=llm)
        self.weather_agent = WeatherAgent(model=llm)
        self.market_agent = MarketAgent(model=llm)
        self.practice_agent = FarmPracticesAgent(model=llm)
        self.farmer_profile_agent = FarmerProfileAgent(model=llm)
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # --- MODIFIED on_message with DEBUG PRINT STATEMENTS ---
    # ------------------------------------------------------------------
    def on_message(self, message: str):
        try:
            print("\n--- 1. on_message started ---")
            farmer_id = "farmer_001"
            location = "Andhra Pradesh"

            print("--- 2. Calling SoilAgent ---")
            soil = self.soil_agent.analyze_soil()
            print(f"--- 3. SoilAgent OK: {soil} ---")

            print("--- 4. Calling WeatherAgent ---")
            weather = self.weather_agent.get_forecast(location)
            print(f"--- 5. WeatherAgent OK: {weather} ---")

            print("--- 6. Calling MarketAgent ---")
            market = self.market_agent.get_prices()
            print(f"--- 7. MarketAgent OK: {market} ---")

            print("--- 8. Calling PracticeAgent ---")
            advice = self.practice_agent.generate_advice(soil, weather, market)
            print(f"--- 9. PracticeAgent OK: {advice} ---")

            context = {"soil": soil, "weather": weather, "market": market}
            
            print("--- 10. Calling FarmerProfileAgent ---")
            personalized = self.farmer_profile_agent.recommend_for_farmer(
                farmer_id, context
            )
            print(f"--- 11. FarmerProfileAgent OK: {personalized} ---")

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
            
            print("--- 12. Success! Sending reply. ---")
            return reply

        except Exception as e:
            print(f"\n❌❌❌ A SUB-AGENT FAILED! ERROR: {e} ❌❌❌\n") 
            traceback.print_exc() # This prints the full error trace
            return f"❌ Error: An exception occurred: {e}"


# --- local CLI test ---
if __name__ == "__main__":
    try:
        agent = ManagerAgent()
        print(agent.on_message("Generate farm advice for farmer_001 in Andhra Pradesh"))
    except Exception as e:
        print(f"❌ Failed to run agent: {e}")

# --- register for ADK Web ---
try:
    root_agent = ManagerAgent()
    print("\n✅ root_agent initialized successfully for ADK Web.\n")
except Exception as e:
    print(f"\n❌ Failed to initialize root_agent for ADK Web: {e}\n")
    traceback.print_exc()
    root_agent = None