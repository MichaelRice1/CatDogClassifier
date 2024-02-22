import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights


#data augmentations 
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#loading in training dataset of 4000 cat images and 4000 dog images
train_dataset = torchvision.datasets.ImageFolder(root='./dataset/training_set', transform=transform)

#creating a training loader object which sets bacth size to 64 and opts to shuffle the batches
#abstracts the complexity of creating and shuffling mini batches every epoch
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

#loading in testing dataset of 1000 cat images and 1000 dog images
test_dataset = torchvision.datasets.ImageFolder(root='./dataset/test_set', transform=transform)

#creating a test loader object which sets batch size to 64 and opts to not shuffle the batches
#abstracts the complexity of creating mini batches every epoch
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

#allows the model to deploy my laptops gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


 
#Custom Neural Network Class
class CatDogClassifier(nn.Module):

    
    #defines the constructor method for NN Class
    def __init__(self):
        super(CatDogClassifier, self).__init__()
        #creates an instance of the ResNet 18 model with default weights ( trained on ImageNet)
        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        #gets number of input features for fully connected layer in the NN
        num_features = self.resnet.fc.in_features
        #replaces the Fully Connected Layer with one with two output units, one each for classifying Cats and Dogs
        self.resnet.fc = nn.Linear(num_features, 2)
 
    def forward(self, x):
        return self.resnet(x)

'''
class customCatDogClassifier(nn.Module):
    def __init__(self):
        super(customCatDogClassifier,self).__init__()
        self.hidden = nn.Linear(32, 180)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 2)  # Change the output to have 2 units for cat and dog
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.output(x)
        x = self.softmax(x)
        return x
'''



#creation of a model object, using GPU as the device
model = CatDogClassifier().to(device)
#model = customCatDogClassifier().to(device)

#defines the loss function (cross entropy loss)
criterion = nn.CrossEntropyLoss()

#creates an Optimizer object, using SGD as the algorithm, with a learning rate of 0.001 and momentum of 0.9
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.95)

# Initialize an empty list to store the loss at each epoch
train_losses = []

#train model method, takes model,criterion,optimzer objects as inputs as well as number of epochs
#loops over the set number of epochs

def train_model(model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        #iterates over batches from the training dataset
        for i, (inputs, labels) in enumerate(train_loader):
            #moves input and labels to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            #zero the gradients to avoid accumulation from previous iterations
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)#getting predictions
            loss = criterion(outputs, labels)#calculating loss

            # backward pass
            loss.backward()#computing gradients
            optimizer.step()#updating parameters using the optimizer

            #computes a running loss metric
            running_loss += loss.item()

        # Calculate and store the average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}')

#method to evaluate the models performance against the test dataset
#tracks the number of correct predictions against the total number of test examples
def eval_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    #turns off gradient computation for the evaluation process
    with torch.no_grad():
        #iteerates over the batches in the testing data
        for data in dataloader:
            images, labels = data#unpack the data into images and labels 
            images, labels = images.to(device), labels.to(device)#loads the images and labels onto the GPU
            outputs = model(images)#forward pass to obtain predictions
            _, predicted = torch.max(outputs.data, 1)#get the index with the maximum value to be the predicted class of image 
            #update totals
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the evaluation set: {100 * correct / total:.2f}%')

#calling the training method
train_model(model, criterion, optimizer, num_epochs=2)

#calling the evaluation method on the test dataset
eval_model(model,test_loader)


# Plot the training loss against the number of epochs
plt.plot(range(1,3), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

