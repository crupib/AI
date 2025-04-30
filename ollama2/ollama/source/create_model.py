import pandas as pd
from ollama import OllamaModel

# Load and preprocess data
data = pd.read_csv('router_config.csv')

# Instantiate and configure the model
model = OllamaModel()
model.config(batch_size=32, epochs=10, learning_rate=0.001)

# Train the model
model.train(data)

# Evaluate the model
accuracy = model.evaluate(validation_data)
print(f'Model Accuracy: {accuracy}%')

# Save the model
model.save('trained_model.ollama')

# Load and use the trained model
trained_model = OllamaModel.load('trained_model.ollama')
predictions = trained_model.predict(test_data)

