import pandas as pd
import rtdl
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

data = pd.read_csv('./jiayouzhongke6.csv',header=None)

features = data.columns[:-1]
target = data.columns[-1]

X_ = data[features]
y_ = data[target]
model = LinearRegression()
rfe = RFE(model, n_features_to_select=150)  # 特征数的定义
fit = rfe.fit(X_, y_)
selected_features = X_.columns[fit.support_]

X_all = data.iloc[:, 0:224].values
X_all = X_all[:, selected_features]

y_all = data.iloc[:, -1].values
n_classes = int(max(y_all)) + 1

X = {}
y = {}
X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(X_all, y_all, train_size=0.8)
X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(X['train'], y['train'], train_size=0.8)

device = torch.device('cpu')
preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
X = {k: torch.Tensor(preprocess.transform(v)) for k, v in X.items()}
y = {k: torch.tensor(v,dtype = torch.long) for k, v in y.items()}
y_std = y_mean = None

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.conv1 = nn.Conv1d(150, 128, kernel_size=3, padding=1)  # 150与特征数必须相同
        self.fc = nn.Linear(128, 150) # 150与特征数必须相同
        self.proj = nn.Linear(150, 3) # 150与特征数必须相同 3表示3分类，2表示2分类
    def forward(self, x_num, x_cat=None):
        x = x_num.unsqueeze(2)
        x = self.conv1(x)
        x = torch.mean(x, 2)
        x = self.fc(x)
        x = self.model(x_num,x_cat)
        output = self.proj(x)
        return output

d_out = 150  # 150与特征数必须相同
model_ = rtdl.FTTransformer.make_default(n_num_features=X['train'].shape[1],cat_cardinalities=None,last_layer_query_idx=[-1],d_out=d_out,)
model = Model(model_)
optimizer = (torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)) #lr学习率
loss_fn = (F.cross_entropy)

def apply_model(x_num, x_cat=None):
    return model(x_num, x_cat)

@torch.no_grad()
def evaluate(part):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 1024):
        feat = apply_model(batch)
        prediction.append(feat)
        
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    
    target = y[part].cpu().numpy()

    prediction = prediction.argmax(1)
    score = sklearn.metrics.accuracy_score(target, prediction)

    return score

batch_size = 256
train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)
progress = zero.ProgressTracker(patience=100)

n_epochs = 1000

for epoch in range(1, n_epochs + 1):
    for iteration, batch_idx in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx]
        feat = apply_model(x_batch)
        loss = loss_fn(feat.squeeze(1), y_batch)
        loss.backward()
        optimizer.step()

        print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

    val_score = evaluate('val')
    test_score = evaluate('test')
    print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
    progress.update((1) * val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
    print()
    if progress.fail:
        break
"""
2分类：        
yongyou9 95.00 特征120 lr = 0.1
xiushui121 96.88 特征105 lr = 0.1
ning84 96.25  特征120 lr = 0.1
jiayouzhongke6 95.00 特征100 lr = 0.001

3分类： 
jiayouzhongke6 91.87 特征150 lr = 0.001
ning84 85.62  特征125 lr = 0.01
yongyou9 93.00 特征115 lr = 0.01
xiushui121  dinner！！！！！！！！！！！！！！！！！！！！！！！！！
"""