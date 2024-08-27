import os

# Step 1: Train the model
print("Training the model...")
os.system('python train.py')

# Step 2: Evaluate the model
print("Evaluating the model...")
os.system('python eval.py')
