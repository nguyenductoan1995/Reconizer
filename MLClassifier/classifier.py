import turicreate as turi

# Load data
url = "dataset/"
data = turi.image_analysis.load_images(url)
# Create a model
data["personType"] = data["path"].apply(lambda path: "Quang" if "Quang" in path else "TOAN")
# Make predictions
data.save("NameWho.sframe")

# Export to Core ML
data.explore()
dataBuffer = turi.SFrame("NameWho.sframe")
trainingBuffers, testingBuffers = dataBuffer.random_split(0.1)
model = turi.image_classifier.create(trainingBuffers, target="personType", model="squeezenet_v1.1")
model = turi.image_classifier.create(trainingBuffers, target="personType", model="resnet-50")
evaluations = model.evaluate(testingBuffers)
print evaluations["accuracy"]
model.save("NameWho.model")
model.export_coreml("NameModel.mlmodel")
