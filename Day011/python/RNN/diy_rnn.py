import torch
from torch import nn

# 1. RNNCell: 保持原样，但注意它是一个线性单元
class RNNCell(nn.Module):
    def __init__(self, input_dim, bias=True):
        super().__init__()
        # 注意：标准的RNNCell通常会有激活函数，如tanh或ReLU
        # self.layer = nn.Linear(2*input_dim, input_dim, bias=bias)
        # 为了更像标准RNN，我们加上tanh
        self.layer = nn.Linear(input_dim + input_dim, input_dim, bias=bias)
        self.activation = nn.Tanh()

    def forward(self, x, h):
        # x: (batch_size, input_dim)
        # h: (batch_size, hidden_dim)
        combined = torch.cat((x, h), dim=1)
        return self.activation(self.layer(combined))

# 2. RNN (单层)
class RNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # hidden_dim 在这里等于 input_dim
        self.rnncell = RNNCell(input_dim)

    def forward(self, xs, h):
        """
        @param xs: (batch_size, seq_len, input_dim)
        @param h: (batch_size, hidden_dim)
        @return outputs: (batch_size, seq_len, hidden_dim)
        @return h_n: (batch_size, hidden_dim)
        """
        outputs = []
        _h = h
        for t in range(xs.shape[1]):
            current_x = xs[:, t, :]
            _h = self.rnncell(current_x, _h)
            outputs.append(_h)

        # 将输出列表堆叠成一个张量
        output_seq = torch.stack(outputs, dim=1)
        return output_seq, _h

# 3. RNNNet (多层)
class RNNNet(nn.Module):
    def __init__(self, num_layers, input_dim):
        super().__init__()
        self.num_layers = num_layers
        self._layers = nn.ModuleList(
            [RNN(input_dim) for _ in range(num_layers)]
        )

    def forward(self, xs, h):
        """
        @param xs: (batch_size, seq_len, input_dim)
        @param h: (num_layers, batch_size, hidden_dim)
        @return output_seq: (batch_size, seq_len, hidden_dim)
        @return h_n: (num_layers, batch_size, hidden_dim)
        """
        current_input = xs
        final_hiddens = []

        for i, layer in enumerate(self._layers):
            h_t = h[i]
            output_seq, h_n = layer(current_input, h_t)
            final_hiddens.append(h_n)
            current_input = output_seq # 下一层的输入是当前层的输出

        all_final_hiddens = torch.stack(final_hiddens, dim=0)
        return current_input, all_final_hiddens

# --- 使用示例 ---
if __name__ == '__main__':
    batch_size = 4
    seq_len = 10
    input_dim = 32
    num_layers = 2

    # 实例化网络
    model = RNNNet(num_layers=num_layers, input_dim=input_dim)

    # 创建随机输入和初始隐藏状态
    inputs = torch.randn(batch_size, seq_len, input_dim)
    h0 = torch.zeros(num_layers, batch_size, input_dim) # hidden_dim == input_dim

    # 前向传播
    output_seq, hn = model(inputs, h0)

    print("Input shape:", inputs.shape)
    print("Output sequence shape:", output_seq.shape) # 应该是 (4, 10, 32)
    print("Final hidden state shape:", hn.shape)      # 应该是 (2, 4, 32)
