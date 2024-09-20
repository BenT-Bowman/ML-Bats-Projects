#!/usr/bin/env python
# coding: utf-8

# # Pytorch Implementation

# In[1]:


import torch
import torch.nn as nn
from torchvision import datasets, transforms


# In[2]:


from modules.CNN import CNN


# In[3]:


from tqdm import tqdm


# ## Data

# In[4]:


data_1_4 = fr"\Data\1_4"
data_5_8 = fr"\Data\5_8"
data_9_12= fr"\Data\9_12"

data_1_12= fr"Data\Final Testing Dataset\Final Testing Dataset"


# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[6]:


transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((40, 40)).to(device),
    transforms.RandomRotation(45).to(device)
])


# ## Model

# ### 1_12

# In[7]:


root_dir = r'Data\Final Testing Dataset\Final Testing Dataset'
dataset = datasets.ImageFolder(root=root_dir, transform=transform)


# In[8]:


model = CNN(12).to(device)


# In[9]:


val_size = int(0.2 * len(dataset))
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size - test_size

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


# In[10]:


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# In[11]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[12]:


length = len(train_dataloader)
for epoch in range(50):  # loop over the dataset multiple times
    print("Epoch: ", epoch+1)
    pbar = tqdm(train_dataloader)
    running_loss = 0.0
    correct = 0
    total_seen = 0
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        # total_seen += labels.size(0)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(labels.shape, outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted.size(0))
        total_seen+=predicted.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_description(f"Running Accuracy: { correct/total_seen},  \t Batch Loss:  {loss}")

    # Testing the network
    correct = 0
    total = 0
    total_loss=0
    with torch.no_grad():
        for (images, labels) in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy: ", correct/total, "\tLoss: ", total_loss/len(val_dataloader))


# In[13]:


torch.save(model, '1_12_bats.pth')


# ### 1_4

# In[14]:


root_dir = r'Data\Top_level\1_4'
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
model_1_4 = CNN(4).to(device)
val_size = int(0.2 * len(dataset))
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size - test_size

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1_4.parameters(), lr=0.0001)

length = len(train_dataloader)
for epoch in range(50):  # loop over the dataset multiple times
    print("Epoch: ", epoch+1)
    pbar = tqdm(train_dataloader)
    running_loss = 0.0
    correct = 0
    total_seen = 0
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        # total_seen += labels.size(0)

        optimizer.zero_grad()

        outputs = model_1_4(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(labels.shape, outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted.size(0))
        total_seen+=predicted.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_description(f"Running Accuracy: { correct/total_seen},  \t Batch Loss:  {loss}")

    # Testing the network
    correct = 0
    total = 0
    total_loss=0
    with torch.no_grad():
        for (images, labels) in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_1_4(images)
            val_loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy: ", correct/total, "\tLoss: ", total_loss/len(val_dataloader))


# ### 5_8

# In[15]:


root_dir = r'Data\Top_level\5_8'
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
model_5_8 = CNN(4).to(device)
val_size = int(0.2 * len(dataset))
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size - test_size

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_5_8.parameters(), lr=0.0001)

length = len(train_dataloader)
for epoch in range(50):  # loop over the dataset multiple times
    print("Epoch: ", epoch+1)
    pbar = tqdm(train_dataloader)
    running_loss = 0.0
    correct = 0
    total_seen = 0
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        # total_seen += labels.size(0)

        optimizer.zero_grad()

        outputs = model_5_8(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(labels.shape, outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted.size(0))
        total_seen+=predicted.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_description(f"Running Accuracy: { correct/total_seen},  \t Batch Loss:  {loss}")

    # Testing the network
    correct = 0
    total = 0
    total_loss=0
    with torch.no_grad():
        for (images, labels) in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_5_8(images)
            val_loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy: ", correct/total, "\tLoss: ", total_loss/len(val_dataloader))


# ### 9_12

# In[16]:


root_dir = r'Data\Top_level\9_12'
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
model_9_12 = CNN(4).to(device)
val_size = int(0.2 * len(dataset))
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size - test_size

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_9_12.parameters(), lr=0.0001)

length = len(train_dataloader)
for epoch in range(50):  # loop over the dataset multiple times
    print("Epoch: ", epoch+1)
    pbar = tqdm(train_dataloader)
    running_loss = 0.0
    correct = 0
    total_seen = 0
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        # total_seen += labels.size(0)

        optimizer.zero_grad()

        outputs = model_9_12(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(labels.shape, outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted.size(0))
        total_seen+=predicted.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_description(f"Running Accuracy: { correct/total_seen},  \t Batch Loss:  {loss}")

    # Testing the network
    correct = 0
    total = 0
    total_loss=0
    with torch.no_grad():
        for (images, labels) in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_9_12(images)
            val_loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy: ", correct/total, "\tLoss: ", total_loss/len(val_dataloader))


# ### Top

# In[17]:


root_dir = r'Data\Top_level'
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
model_top = CNN(3).to(device)
val_size = int(0.2 * len(dataset))
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size - test_size

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_top.parameters(), lr=0.0001)

length = len(train_dataloader)
for epoch in range(50):  # loop over the dataset multiple times
    print("Epoch: ", epoch+1)
    pbar = tqdm(train_dataloader)
    running_loss = 0.0
    correct = 0
    total_seen = 0
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        # total_seen += labels.size(0)

        optimizer.zero_grad()

        outputs = model_top(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(labels.shape, outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted.size(0))
        total_seen+=predicted.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_description(f"Running Accuracy: { correct/total_seen},  \t Batch Loss:  {loss}")

    # Testing the network
    correct = 0
    total = 0
    total_loss=0
    with torch.no_grad():
        for (images, labels) in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_top(images)
            val_loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy: ", correct/total, "\tLoss: ", total_loss/len(val_dataloader))


# # Issue Here?

# In[18]:


# Forward pass function
def forward(input_data):
    # Use the top-level model to decide which specialized model to use
    with torch.no_grad():
        decision = model_top(input_data)
        # Get the index (decision) for each input in the batch
        decisions = torch.argmax(decision, dim=1)  # This returns a tensor of decisions for each input in the batch
        print(decisions)

    # Initialize a batch of target tensors
    batch_size = input_data.size(0)
    target_tensor = torch.zeros(batch_size, 12).to(device)  # Initialize the target tensor for the entire batch

    for i in range(batch_size):
        # Route the input to the correct specialized model based on the decision for each element
        if decisions[i] == 0:
            output_tensor = model_1_4(input_data[i].unsqueeze(0))  # Process the ith element
            # target_size = min(output_tensor.size(1), 12)  # Ensure it doesn't exceed target tensor size
            target_tensor[i, :4] = output_tensor.squeeze(0)[:4]  # Overlay on the target tensor
        
        elif decisions[i] == 1:
            output_tensor = model_5_8(input_data[i].unsqueeze(0))  # Process the ith element
            # target_size = min(output_tensor.size(1), 7)  # Ensure it fits in the available space (12-5=7)
            target_tensor[i, 4:4+4] = output_tensor.squeeze(0)[:4]  # Overlay on the target tensor
            
        elif decisions[i] == 2:
            output_tensor = model_9_12(input_data[i].unsqueeze(0))  # Process the ith element
            # target_size = min(output_tensor.size(1), 3)  # Ensure it fits in the available space (12-9=3)
            target_tensor[i, 8:8+4] = output_tensor.squeeze(0)[:4]  # Overlay on the target tensor

    return target_tensor



# In[19]:


root_dir = r'Data\Final Testing Dataset\Final Testing Dataset'
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
val_size = int(0.2 * len(dataset))
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size - test_size

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


# In[20]:


correct = 0
total = 0
total_loss=0
with torch.no_grad():
    for (images, labels) in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = forward(images)
        val_loss = nn.CrossEntropyLoss()(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Accuracy: ", correct/total, "\tLoss: ", total_loss/len(test_dataloader))


# In[21]:


predicted


# In[22]:


labels

