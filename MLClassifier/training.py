import turicreate as tc

# Load data
data = tc.SFrame('photoLabel.sframe')

# Create a model
model = tc.image_classifier.create(data, target='photoLabel')

# Make predictions
predictions = model.predict(data)

# Export to Core ML
model.export_coreml('MyClassifier.mlmodel')
