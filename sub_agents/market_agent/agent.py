from google.adk import Agent

class MarketAgent(Agent):
    model_config = {"extra": "allow"}
    def __init__(self):
        super().__init__(name="market_agent")

    def get_prices(self):
        # Mock example data
        return {
            "wheat": "₹2400/quintal",
            "rice": "₹2600/quintal",
            "soybean": "₹5400/quintal"
        }
