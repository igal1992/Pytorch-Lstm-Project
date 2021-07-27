import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import utils
from ExcelClass import ExcelClass

num_epochs = 2000
exlObject = ExcelClass(2, "data15")


class lstmModel(nn.Module):
    def __init__(self):
        super(lstmModel, self).__init__()
        self.input_dim = 6  # represents the size of the input at each time step
        self.hidden_dim = 500  # represents the size of the hidden state and cell state at each time step
        self.num_layers = 1  # represents the num of layers stacked upon each other in lstm activation step

        self.lstm_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
        h0 = torch.zeros(self.num_layers, len(x), self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers, len(x), self.hidden_dim).cuda()
        out, _ = self.lstm_layer(x, (h0, c0))
        out = self.relu(out[:, -1, :])
        out = self.fc(out)
        return out


# Train the model
def train_model(model, loader, criterion, optimizer):
    accum_loss_end = []
    for epoch in tqdm(range(num_epochs),desc="Transfer progress", ncols=100, bar_format='{l_bar}{bar}|'):
        accum_loss = []
        random.shuffle(loader)
        for i, (feature, label) in enumerate(loader):
            if label != 0:
                _input = torch.tensor(feature).cuda()
                _label = torch.tensor(label).cuda()

                # Forward pass
                output = model(_input.float())
                output = torch.squeeze(torch.squeeze(output, 0), 0)
                loss = criterion(output, _label.float())
                accum_loss.append(loss.item())
                accum_loss_end.append(loss.item())

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if epoch % 50 == 0:
            print(f'\n[Epoch [{epoch + 1}/{num_epochs}]], [Max Loss: {max(accum_loss):.4f}] , [Min Loss: {min(accum_loss):.4f}]')
    return model


# Test the model
def accuracy_on_dataset(model, test_loader, ghz):
    # In test phase, we don't need to compute gradients (for memory efficiency)
    real_labels = []
    predicted_labels = []
    flag = False
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for feature, label in test_loader:
            if label != 0:
                _input = torch.tensor(feature).cuda()
                _label = torch.tensor(float(label)).cuda()
                _label = torch.unsqueeze(_label, 0)  # added a dimension
                output = model(_input.float())
                real_labels.append(_label)
                predicted_labels.append(round(output.item(), 3))
                # max returns (value ,index)
                n_samples += 1
                if abs(round(output.item(), 3) - _label.item()) < 0.005:
                    n_correct += 1
            elif flag is False and label == 0:
                # print the first graph and reset arrays
                plt.plot(real_labels)
                plt.title("First Building(Real loss - Blue Graph, Predicted loss - Orange Graph)")
                plt.xlabel("Time(s)")
                plt.ylabel("Building Entry Loss(dB)")
                plt.plot(predicted_labels)
                plt.savefig(ghz+'Ghz approximately Graph Figure 1.png')
                plt.clf()
                real_labels = []
                predicted_labels = []
                flag = True
            else:
                continue
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the data_Set: {acc} %')
    # print the second graph and
    plt.plot(real_labels)
    plt.title("First Building(Real loss - Blue Graph, Predicted loss - Orange Graph)")
    plt.xlabel("Time(s)")
    plt.ylabel("Building Entry Loss(dB)")
    plt.plot(predicted_labels)
    plt.savefig(ghz+'Ghz approximately Graph Figure 2.png')
    return acc


# Generate new Excel loss prediction columns
def setup_excel(model, train_loader, randomBuilding):
    _, firstBuilding = utils.calculateAngle(random.randint(0, 250), random.randint(0, 3000), False)
    _, secondBuilding = utils.calculateAngle(random.randint(0, 250), random.randint(0, 3000), True)
    set_params = exlObject.get_data_matrix(exlObject.startingIndex)
    outputs = []
    for i, (feature, label) in enumerate(set_params):
        # checks if generating random building or not, accordingly checks the train labels if not random building
        if (train_loader[i][1] != 0 and train_loader[i][1] is not None and randomBuilding is False) or (
                firstBuilding > feature[1] or secondBuilding < feature[1] and randomBuilding):
            _input = torch.tensor(feature).cuda()
            _label = torch.tensor(label).cuda()
            # Forward pass
            output = model(_input.float())
            outputs.append(output.cpu().detach().numpy()[0][0])
        # else if not random building checks train labels or if is random building checks nothing
        elif (train_loader[i][1] == 0 and randomBuilding is False) or randomBuilding:
            outputs.append(0)
    return outputs


if __name__ == '__main__':
    lr = 0.00005
    train_loader = exlObject.get_data_matrix(3)
    copied_trained_loader = copy.deepcopy(train_loader)
    model = lstmModel().cuda()
    criterion = nn.modules.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    trained_model = train_model(model, copied_trained_loader, criterion, optimizer)
    accuracy_on_dataset(trained_model, train_loader, "15")
    exlObject.setStartingRow("range", 0)
    temp0 = setup_excel(trained_model, train_loader, False)
    exlObject.setDataInExcel("urbanTerresLossPred", temp0, "data15", True)
    exlObject.setStartingRow("range", 1)
    temp1 = setup_excel(trained_model, train_loader, False)
    exlObject.setDataInExcel("urbanTerresLossPred", temp1, "data15", True)
    exlObject.setStartingRow("range", 2)
    temp2 = setup_excel(trained_model, train_loader, True)
    exlObject.setDataInExcel("urbanTerresLossPred", temp2, "data15", True)
