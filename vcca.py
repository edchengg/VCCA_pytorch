from torch import nn, cat
from torch.autograd import Variable
ZDIMS=20
PDIMS=30
class VCCA(nn.Module):
    def __init__(self, private):
        super(VCCA, self).__init__()
        self.private = private
        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.en_z_1 = nn.Linear(784, 1024)
        self.en_z_2 = nn.Linear(1024, 1024)
        self.en_z_3 = nn.Linear(1024, 1024)

        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.en_z_4_mu = nn.Linear(1024, ZDIMS)  # mu layer
        self.en_z_4_sigma = nn.Linear(1024, ZDIMS)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections
        if self.private:
            self.en_x_1 = nn.Linear(784, 1024)
            self.en_x_2 = nn.Linear(1024, 1024)
            self.en_x_3 = nn.Linear(1024, 1024)
            self.en_x_4_mu = nn.Linear(1024, PDIMS)
            self.en_x_4_sigma = nn.Linear(1024, PDIMS)

            self.en_y_1 = nn.Linear(784, 1024)
            self.en_y_2 = nn.Linear(1024, 1024)
            self.en_y_3 = nn.Linear(1024, 1024)
            self.en_y_4_mu = nn.Linear(1024, PDIMS)
            self.en_y_4_sigma = nn.Linear(1024, PDIMS)

        # DECODER 1
        # from bottleneck to hidden 400
        if self.private:
            self.de_x_1 = nn.Linear(PDIMS+ZDIMS, 1024)
        else:
            self.de_x_1 = nn.Linear(ZDIMS, 1024)
        self.de_x_2 = nn.Linear(1024, 1024)
        self.de_x_3 = nn.Linear(1024, 1024)
        self.de_x_4 = nn.Linear(1024, 784)

        # DECODER 2
        if self.private:
            self.de_y_1 = nn.Linear(PDIMS+ZDIMS, 1024)
        else:
            self.de_y_1 = nn.Linear(ZDIMS, 1024)
        self.de_y_2 = nn.Linear(1024, 1024)
        self.de_y_3 = nn.Linear(1024, 1024)
        self.de_y_4 = nn.Linear(1024, 784)

        self.sigmoid = nn.Sigmoid()

    def encode(self, x: Variable) -> (Variable, Variable):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected

        """
        h1 = self.relu(self.en_z_1(self.dropout(x)))
        h1 = self.relu(self.en_z_2(self.dropout(h1)))
        h1 = self.relu(self.en_z_3(self.dropout(h1)))
        return self.en_z_4_mu(self.dropout(h1)), self.en_z_4_sigma(self.dropout(h1))

    def private_encoder1(self, x:Variable):
        h1 = self.relu(self.en_x_1(self.dropout(x)))
        h1 = self.relu(self.en_x_2(self.dropout(h1)))
        h1 = self.relu(self.en_x_3(self.dropout(h1)))
        return self.en_x_4_mu(self.dropout(h1)), self.en_x_4_sigma(self.dropout(h1))

    def private_encoder2(self, y:Variable):
        h1 = self.relu(self.en_y_1(self.dropout(y)))
        h1 = self.relu(self.en_y_2(self.dropout(h1)))
        h1 = self.relu(self.en_y_3(self.dropout(h1)))
        return self.en_y_4_mu(self.dropout(h1)), self.en_y_4_sigma(self.dropout(h1))

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:

        """

        if self.training:

            std = logvar.mul(0.5).exp_()  # type: Variable

            eps = Variable(std.data.new(std.size()).normal_())

            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode_1(self, z: Variable) -> Variable:
        h3 = self.relu(self.de_x_1(self.dropout(z)))
        h3 = self.relu(self.de_x_2(self.dropout(h3)))
        h3 = self.relu(self.de_x_3(self.dropout(h3)))
        return self.sigmoid(self.de_x_4(self.dropout(h3)))

    def decode_2(self, z: Variable) -> Variable:
        h3 = self.relu(self.de_y_1(self.dropout(z)))
        h3 = self.relu(self.de_y_2(self.dropout(h3)))
        h3 = self.relu(self.de_y_3(self.dropout(h3)))
        return self.sigmoid(self.de_y_4(self.dropout(h3)))

    def forward(self, x: Variable, y: Variable) -> (Variable, Variable, Variable):
        mu, log_var = self.encode(x.view(-1, 784))

        if self.private:
            mu1, log_var1 = self.private_encoder1(x.view(-1, 784))
            mu2, log_var2 = self.private_encoder2(y.view(-1, 784))
            mu1_tmp = cat((mu,mu1), 1)
            log_var1_tmp = cat((log_var,log_var1), 1)
            mu2_tmp = cat((mu, mu2), 1)
            log_var2_tmp = cat((log_var, log_var2), 1)
            z1 = self.reparameterize(mu1_tmp, log_var1_tmp)
            z2 = self.reparameterize(mu2_tmp, log_var2_tmp)
            return self.decode_1(z1), self.decode_2(z2), mu, log_var, mu1, log_var1, mu2, log_var2

        z1 = self.reparameterize(mu, log_var)
        z2 = self.reparameterize(mu, log_var)
        return self.decode_1(z1), self.decode_2(z2), mu, log_var

