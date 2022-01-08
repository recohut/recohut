class SparseFeat(namedtuple('SparseFeat', ['name', 'vocabulary_size', 'embedding_dim', 'use_hash',
                                           'dtype', 'embedding_name', 'group_name'])):
    __slots__ = ()
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name='default_group'):
        if embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype, embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()
    def __new__(cls, name, dimension=1, dtype='float32'):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)
    def __hash__(self):
        return self.name.__hash__()

def activation_layer(act_name, hidden_size=None, dice_dim=2):
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    return act_layer

def get_auc(loader, model):
    pred, target = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).float()
            y_hat = model(x)
            pred += list(y_hat.cpu().numpy())
            target += list(y.cpu().numpy())
    auc = roc_auc_score(target, pred)
    return auc

class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)
        ])
        if use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)
            ])
        self.activation_layer = nn.ModuleList([
            activation_layer(activation, hidden_units[i+1], dice_dim) for i in range(len(hidden_units)-1)
        ])
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)
    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layer[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

class BiInteractionPooling(nn.Module):
    def __init__(self):
        super(BiInteractionPooling, self).__init__()
    def forward(self, inputs):
        concated_embeds_value = inputs
        square_of_sum = torch.pow(torch.sum(concated_embeds_value, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(concated_embeds_value * concated_embeds_value, dim=1, keepdim=True)
        cross_term = 0.5 * (square_of_sum - sum_of_square)
        return cross_term

class NFM(nn.Module):
    def __init__(self, feat_sizes, embedding_size, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128),
                 l2_reg_embedding=1e-5, l2_reg_linear=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, bi_dropout=1,
                 dnn_dropout=0, dnn_activation='relu', task='binary', device='cpu', gpus=None):
        super(NFM, self).__init__()
        self.dense_features_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
        dense_input_dim = sum(map(lambda x: x.dimension, self.dense_features_columns))

        self.sparse_features_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []

        self.feat_sizes = feat_sizes
        self.embedding_size = embedding_size
        self.embedding_dic = nn.ModuleDict({feat.name:nn.Embedding(self.feat_sizes[feat.name], self.embedding_size, sparse=False)
                                            for feat in self.sparse_features_columns})
        for tensor in self.embedding_dic.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

        self.feature_index = defaultdict(int)
        start = 0
        for feat in self.feat_sizes:
            if feat in self.feature_index:
                continue
            self.feature_index[feat] = start
            start += 1

        self.dnn = DNN(dense_input_dim+self.embedding_size, dnn_hidden_units, activation=dnn_activation,
                       l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

        dnn_hidden_units = [len(self.feature_index)] + list(dnn_hidden_units) + [1]
        self.Linears = nn.ModuleList(
            [nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i + 1]) for i in range(len(dnn_hidden_units) - 1)])
        self.relu = nn.ReLU()
        self.bi_pooling = BiInteractionPooling()
        self.bi_dropout = bi_dropout
        if self.bi_dropout > 0:
            self.dropout  = nn.Dropout(bi_dropout)
        self.to(device)

    def forward(self, X):
        sparse_embedding = [self.embedding_dic[feat.name](X[:, self.feature_index[feat.name]].long()).reshape(X.shape[0], 1, -1)
                            for feat in self.sparse_features_columns]
        dense_values = [X[:, self.feature_index[feat.name]].reshape(-1, 1) for feat in self.dense_features_columns]
       # print('sparse_embedding shape', sparse_embedding[0].shape)
        dense_input = torch.cat(dense_values, dim=1)
       # print('densn_input shape', dense_input.shape)
        fm_input = torch.cat(sparse_embedding, dim=1)
       # print('fm_input_shape', fm_input.shape)
        bi_out = self.bi_pooling(fm_input)
       # print('bi_out shape', bi_out.shape)
        if self.bi_dropout:
            bi_out = self.dropout(bi_out)

        bi_out = torch.flatten(torch.cat([bi_out], dim=-1), start_dim=1)

        dnn_input = torch.cat((dense_input, bi_out), dim=1)
        dnn_output = self.dnn(dnn_input)
        dnn_output = self.dnn_linear(dnn_output)

       # print('X shape', X.shape)
        for i in range(len(self.Linears)):
            fc = self.Linears[i](X)
            fc = self.relu(fc)
            fc = self.dropout(fc)
            X = fc

        logit = X + dnn_output
        y_pred = torch.sigmoid(logit)
        return y_pred