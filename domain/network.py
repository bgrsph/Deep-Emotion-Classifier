import torch.nn as nn
from torch.backends import cudnn
from domain.modelFactory import *
from torch.autograd import Variable
import time
from domain.dataLoader import *
from sklearn.metrics import accuracy_score
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from domain.debugger import Debug

# Make this true if debug
verbose = True

class Network:

    def __init__(self, args):
        Debug(verbose=verbose,message="Network is initializing with args : {}".format(args))
        self.args = args
        modelFactory = ModelFactory(args)
        Debug(verbose=verbose, message=modelFactory)
        self.model = modelFactory.get_model()
        Debug(verbose=verbose, message="Model has been selected")
        self.data_loader = DataLoader(args=self.args)

    def train(self):
        Debug(verbose=verbose,message="train method has been started.")

        device = torch.device(self.args["device"])

        Debug(verbose=verbose, message="Device has been detected: {}".format(device.type))

        num_epochs = int(self.args["epochs"])

        Debug(verbose=verbose, message="Number of epochs has been detected: {}".format(num_epochs))

        loss_fn = nn.CrossEntropyLoss()

        Debug(verbose=verbose, message="Loss function has been determined.")

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args["learning_rate"], weight_decay=self.args["weight_decay"], momentum=self.args["momentum"])  # Later get this info from args

        Debug(verbose=verbose, message="Optimizer has been determined.")

        # Get train and validation data sets
        train_dataset = self.data_loader.get_data_set("train")
        valid_dataset = self.data_loader.get_data_set("validation")

        # Get iterable data sets
        train_load = self.data_loader.get_iterable_data_set("train")
        valid_load = self.data_loader.get_iterable_data_set("validation")

        Debug(verbose=verbose,message="Datasets has been collected.")

        self.model = self.model.to(device)
        if device.type == 'cuda':
            cudnn.benchmark = True

        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []

        Debug(verbose=verbose, message="Entering in the epoch loop")
        # Training
        for epoch in range(num_epochs):
            Debug(verbose=verbose, message="In epoch # {}".format(epoch))
            # Reset these below variables to 0 at the begining of every epoch
            start = time.time()
            correct = 0
            iterations = 0
            iter_loss = 0.0

            self.model.train()  # Put the network into training mode
            Debug(verbose=verbose, message="Network is now in training mode")
            Debug(verbose=verbose, message="Entering the train data_load loop")
            for i, (inputs, labels, path) in enumerate(train_load):
                # Convert torch tensor to Variable
                inputs = Variable(inputs)
                labels = Variable(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                Debug(verbose=verbose, message="Inputs and labels are converted to device type")
                optimizer.zero_grad()  # Clear off the gradient in (w = w - gradient)
                Debug(verbose=verbose, message="Optimizer has cleaned the gradient")
                outputs = self.model(inputs)
                Debug(verbose=verbose, message="Inputs has been fed to network and outputs were collected.")
                loss = loss_fn(outputs, labels)
                Debug(verbose=verbose, message="Loss calculated")
                iter_loss += loss.data  # Accumulate the loss
                loss.backward()  # Backpropagation
                Debug(verbose=verbose, message="Backpropagated")
                optimizer.step()  # Update the weights
                #print("path: " + str(path) + " i: " + str(i))
                Debug(verbose=verbose, message="Weights had been updated")
                # Record the correct predictions for training data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
                iterations += 1

            # Record the training loss
            train_loss.append(iter_loss / iterations)

            # Record the training accuracy
            train_accuracy.append((100 * correct / len(train_dataset)))

            # Testing
            loss = 0.0
            correct = 0
            iterations = 0
            self.model.eval()  # Put the network into evaluation mode
            for i, (inputs, labels, path) in enumerate(valid_load):
                # Convert torch tensor to Variable
                inputs = Variable(inputs)
                labels = Variable(labels)

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)  # Calculate the loss
                loss += loss.data
                # Record the correct predictions for training data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()

                iterations += 1

            # Record the Testing loss
            test_loss.append(loss / iterations)
            # Record the Testing accuracy
            test_accuracy.append((100 * correct / len(valid_dataset)))
            stop = time.time()

            print(
                'Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}, Time: {}s'
                    .format(epoch + 1, num_epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1],
                            stop - start))

        # Run this if you want to save the model
        torch.save(self.model.state_dict(), self.args["output_name"] + ".pth")

        Debug(verbose=True,message="Trained model has been saved.")


    def test(self):
        Debug(verbose=verbose,message="Starting to test...")

        device = torch.device(self.args["device"])

        if device.type == 'cuda':
            cudnn.benchmark = True

        self.model.to(device)

        Debug(verbose=verbose, message="Device Selected")

        self.model.load_state_dict(torch.load(self.args["pretrained_data_path"],map_location=device))

        Debug(verbose=verbose, message="Model Loaded")

        self.model.eval()

        Debug(verbose=verbose, message="Model is now in evaluation mode")

        test_load2 = self.data_loader.get_iterable_data_set("test")

        loss_fn = nn.CrossEntropyLoss()

        output_classes_counts = {}  # Holds each output class with their counts

        Debug(verbose=verbose, message="Loss Function has been selected")

        # Initialize all counts to zero
        for out_class in self.args["output_classes"]:
            output_classes_counts[out_class] = 0

        Debug(verbose=verbose, message="All Counts of output classes: {}".format(output_classes_counts))

        Debug(verbose=verbose, message="Starting to iterate over test dataset")
        total = 0
        correct = 0
        for i, (inputs, labels, path) in enumerate(test_load2):
            Debug(verbose=verbose, message="Iteration #{}".format(i))
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = self.model(inputs)
            Debug(verbose=verbose, message="Outputs had been collected from model")

            # Calculate the loss
            loss = loss_fn(outputs, labels)
            loss += loss.data
            Debug(verbose=verbose, message="Loss has been calculated")

            # Record the correct predictions for training data
            _, predicted = torch.max(outputs, 1)
            # Calculate accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 100 test images: %d %%' % (
                100 * correct / total))
        Debug(verbose=verbose, message="End of testing")