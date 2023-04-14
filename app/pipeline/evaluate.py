import torch

class FruitModelTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        
    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            
            # Set model to train mode
            self.model.train()
            
            for i, (inputs, labels) in enumerate(train_loader):
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update running loss and accuracy
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).sum().item()
            
            # Calculate average loss and accuracy
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_acc / len(train_loader.dataset)
            
            # Evaluate model on validation set
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Print epoch statistics
            print("Epoch {}/{}".format(epoch+1, epochs))
            print("Train loss: {:.4f}, Train accuracy: {:.2f}%".format(train_loss, train_acc*100))
            print("Val loss: {:.4f}, Val accuracy: {:.2f}%".format(val_loss, val_acc*100))
            print()
            
    def evaluate(self, loader):
        loss = 0.0
        acc = 0.0
        
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
            for inputs, labels in loader:
                # Forward pass
                outputs = self.model(inputs)
                loss += self.criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                acc += (predicted == labels).sum().item()
                
        # Calculate average loss and accuracy
        loss = loss / len(loader.dataset)
        acc = acc / len(loader.dataset)
        
        return loss, acc
