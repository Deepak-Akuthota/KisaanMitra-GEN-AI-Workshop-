from google.adk import Agent

class SoilAgent(Agent):
    model_config = {"extra": "allow"}
    def __init__(self):
        super().__init__(name="soil_agent")

    def analyze_soil(self):
        # Example: static or sensor data
        return {
            "moisture": "25%",
            "pH": 6.8,
            "nitrogen": "moderate",
            "recommendation": "Suitable for wheat and pulses"
        }
