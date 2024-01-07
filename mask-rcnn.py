import torch
from transformers import MaskRCNNForInstanceSegmentation, MaskRCNNProcessor, DefaultDataCollator
from torch.utils.data import DataLoader

# Ensure you have the necessary packages installed
# pip install transformers torch torchvision

# Constants and configurations
MODEL_NAME = "blesot/Mask-RCNN"
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

train_dataset = YourDataset("/Users/pranavmacbookpro/Desktop/kelp/Archive/test")
val_dataset = YourDataset("/Users/pranavmacbookpro/Desktop/kelp/Archive/train")

# Data collator 
data_collator = DefaultDataCollator(return_tensors="pt")

# DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)

# Load the model and processor
model = MaskRCNNForInstanceSegmentation.from_pretrained(MODEL_NAME)
processor = MaskRCNNProcessor.from_pretrained(MODEL_NAME)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training Loop
model.train()
for epoch in range(NUM_EPOCHS):
    for batch in train_loader:
        # Process your inputs here
        inputs = processor(images=batch["images"], return_tensors="pt")
        
        # Forward pass
        outputs = model(**inputs)

        # Compute loss
        loss = outputs.loss

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the final model
model.save_pretrained("/Users/pranavmacbookpro/Desktop/kelp")
