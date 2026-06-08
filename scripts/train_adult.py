from torch._inductor import compiler_bisector
import os
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data_path = "data/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv(data_path, names=columns, skipinitialspace= True)

#preprocessing 
#making it simpler, maping sex to binary 0 for female and 1 for male

df['sex_binary'] = df['sex'].map({'Female' : 0, 'Male' : 1})

#6 for now might expand later based on model accuracy 
feature_cols = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "sex_binary",
]

X = df[feature_cols].copy()
# Target label: 1 if income >50K, 0 otherwise
y = df['income'].map({'<=50K': 0, '>50K': 1}).values

#scalling it 

scaler = StandardScaler()
X.iloc[:, :5] = scaler.fit_transform(X.iloc[:, :5])
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
#(can changet he train test split ratio later)
#converting it to tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test) 


class AdultClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AdultClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)  # 2 output classes (<=50K vs >50K)
        )

    def forward(self, x): 
        return self.network(x)

input_dim = len(feature_cols) 
model = AdultClassifier(input_dim)

criterion = nn.CrossEntropyLoss()
    #using ADAM as optimizer if u guys think there are better alternatives let me know
optimizer = optim.Adam(model.parameters(), lr= 0.01)

print("Training model")
for epoch in range(15):
     model.train()
     optimizer.zero_grad()
     outputs = model(X_train_t)
     loss = criterion(outputs, y_train_t)
     loss.backward()
     optimizer.step()
        
        # Evaluate accuracy
     model.eval()
     with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1)
        accuracy = (preds == y_test_t).float().mean().item()
            
     print(f"Epoch {epoch+1}/15 - Loss: {loss.item():.4f} - Test Accuracy: {accuracy*100:.2f}%")


#converting it into well onnx for now (not json)
model.eval()
    
dummy_input = torch.randn(1, input_dim)

onnx_path = "models/adult_classifier.onnx"
os.makedirs("models", exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params= True,
    opset_version= 14, 
    input_names= ["input"],
    output_names= ["output"]
)

print(f"Model saved to {onnx_path}")


