

class stackedResidualLSTM:
    def __init__(self, n_layers=4):
        self.n_layers = n_layers

    def createStackedLSTM(self):

        rnn1 = nn.LSTM(10, 20, 1)
        input = Variable(torch.randn(5, 3, 10))
        output1, hn = rnn1(input)

        rnn2 = nn.LSTM(20, 30, 1)
        output2, hn2 = rnn2(output1)

        residual1 = torch.add(output2, input)
        rnn3 = nn.LSTM(20, 30, 1)
        output3, hn2 = rnn2(residual1)

        residual2 = torch.add(output3, output1)
        rnn4 = nn.LSTM(20, 30, 1)
        output4, hn2 = rnn2(output3+output1)
