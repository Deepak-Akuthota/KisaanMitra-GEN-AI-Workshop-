# sub_agents/weather_agent/agent.py
from google.adk import Agent
import os, requests
from dotenv import load_dotenv

load_dotenv()

class WeatherAgent(Agent):
    model_config = {"extra": "allow"}
    def __init__(self):
        super().__init__(name="weather_agent")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def get_forecast(self, location="Bengaluru"):
        prompt = f"Give a short realistic daily weather report for {location} in November in India, include temperature, humidity, and condition."
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/text-bison-001:generateText?key={self.gemini_key}"
        data = {"prompt": {"text": prompt}, "temperature": 0.3}
        resp = requests.post(endpoint, json=data)
        return resp.json()
