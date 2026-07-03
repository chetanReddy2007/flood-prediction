from locust import HttpUser, task, between

class FloodPredictionUser(HttpUser):
    wait_time = between(1, 3)

    @task(2)
    def home_page(self):
        self.client.get("/")

    @task(3)
    def predict_flood(self):
        self.client.post("/predict", data={
            "MonsoonIntensity": 8,
            "TopographyDrainage": 7,
            "RiverManagement": 6,
            "Deforestation": 8,
            "Urbanization": 9,
            "ClimateChange": 8,
            "DamsQuality": 7,
            "Siltation": 6,
            "AgriculturalPractices": 7,
            "Encroachments": 8,
            "IneffectiveDisasterPreparedness": 9,
            "DrainageSystems": 7,
            "CoastalVulnerability": 8,
            "Landslides": 6,
            "Watersheds": 7,
            "DeterioratingInfrastructure": 8,
            "PopulationScore": 9,
            "WetlandLoss": 7,
            "InadequatePlanning": 8,
            "PoliticalFactors": 6
        })

    @task(1)
    def about_page(self):
        self.client.get("/about")