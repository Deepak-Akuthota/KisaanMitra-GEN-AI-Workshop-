from google.adk import Agent

class FarmPracticesAgent(Agent):
    model_config = {"extra": "allow"}

    def __init__(self):
        super().__init__(name="farm_practices_agent")

    def generate_advice(self, soil, weather, market):
        """
        Safely generate general farming practice advice.
        Handles missing or free-form weather responses gracefully.
        """
        try:
            condition = ""
            # weather may be a dict with condition, or raw text, or nested
            if isinstance(weather, dict):
                condition = weather.get("condition", "")
            elif isinstance(weather, str):
                condition = weather
            elif "raw_output" in weather:
                condition = weather["raw_output"]

            if isinstance(condition, str):
                condition = condition.lower()

            if soil.get("pH", 7) > 7 and "rain" in condition:
                return "Plant paddy for high yield due to rain and alkaline soil."
            elif "dry" in condition or "sunny" in condition:
                return "Soil looks dry; consider drought-tolerant crops like millets or pulses."
            elif "humid" in condition:
                return "Conditions are humid; maintain pest control and proper ventilation."
            else:
                return "Proceed with wheat or pulses; monitor moisture regularly."

        except Exception as e:
            return f"Could not generate advice due to: {e}"
