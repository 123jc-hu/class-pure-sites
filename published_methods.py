"""T19 Parisot Chebyshev GCN and Kunda TPE–MIDA–ridge."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

import numpy as np
from nilearn.connectome import ConnectivityMeasure
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
import torch
from torch import nn
from torch.nn import functional as F

import abide_class_pure_sites as a
import bridge_design as bridge

OUT = ROOT/'outputs/T19'
UPSTREAM = Path.home()/'external_code'
KUNDA = UPSTREAM/'fMRI-site-adaptation'
PARISOT = UPSTREAM/'population-gcn'
PARISOT_FORK = UPSTREAM/'parisot-gcn'


def main_jobs():
    base = [j|dict(method={'P':'B0','M':'M'}[j['arm']]) for j in a.read_jobs()
            if not j['no_zero_roi'] and j['n']==40 and j['arm'] in ('P','M') and j['k'] in (2,4)]
    anchors = {j['job_id']:j for j in base}
    path = ROOT/'outputs/T13B/splits.jsonl'
    from_t13 = [j for j in map(json.loads,path.read_text().splitlines()) if j['feature']=='HO'
              and j['method']=='C0' and j['b']==5 and j['direction']=='twoway' and j['k'] in (2,4)]
    reference = {j['job_id']:j for j in bridge.read_plans() if j['method']=='C0'}
    linked = []
    for existing in from_t13:
        j = reference[existing['source_job_id']]
        assert all(j[k]==existing[k] for k in ('train','test','reverse','test_sites'))
        original = anchors[j['T12A_job_id']]
        assert all(j[k]==original[k] for k in ('test','test_sites','repeat','k'))
        assert j['pure_train']==original['train'] and set(j['train'])-set(original['train'])==set(original['reverse'])-set(j['reverse'])
        linked.append(j)
    jobs = sorted(base+linked,key=lambda j:(j['repeat'],j['k'],j['method']))
    assert len(jobs)==300 and len({(j['repeat'],j['k'],j['method']) for j in jobs})==300
    return jobs


def parisot_selector(x, train, labels, n_features):
    rfe = RFE(RidgeClassifier(),n_features_to_select=n_features,step=100)
    return rfe.fit(x[train],labels)


def parisot_graph(selected, site, sex, bandwidth=None):
    ds = squareform(pdist(selected,metric='correlation'))
    if bandwidth is None:
        bandwidth = ds.mean()
    phenotype = (site[:,None]==site).astype(float)+(sex[:,None]==sex)
    np.fill_diagonal(phenotype,0)
    return phenotype*np.exp(-ds**2/(2*bandwidth**2)),bandwidth


def parisot_supports(adjacency):
    n = len(adjacency)
    degree = adjacency.sum(axis=1)**-.5
    identity = np.eye(n)
    laplacian = identity-degree[:,None]*adjacency*degree
    largest = eigh(laplacian,eigvals_only=True,subset_by_index=[n-1,n-1])[0]
    step = 2*laplacian/largest-identity
    second = 2*step@step-identity
    return np.stack([identity,step,second,2*step@second-step])


def parisot_normalize(selected):
    row_sum = selected.sum(axis=1,keepdims=True)
    return np.divide(selected,row_sum,out=np.zeros_like(selected),where=row_sum!=0)


def parisot_design(x, site, sex, train, labels, n_features=2000):
    selected = parisot_selector(x,train,labels,n_features).transform(x)
    adjacency,_ = parisot_graph(selected,site,sex)
    return parisot_normalize(selected),parisot_supports(adjacency)


class ParisotGCN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.first = nn.Parameter(torch.empty(4,n_features,16))
        self.second = nn.Parameter(torch.empty(4,16,2))
        for bank in (self.first,self.second):
            for weight in bank:
                nn.init.xavier_uniform_(weight)

    def forward(self, x, supports):
        x = F.dropout(x,.3,self.training)
        hidden = (supports@(x@self.first)).sum(dim=0).relu()
        hidden = F.dropout(hidden,.3,self.training)
        return (supports@(hidden@self.second)).sum(dim=0)


def train_parisot(z, s, train, labels):
    assert np.isfinite(z).all() and np.isfinite(s).all()
    z,s = [torch.as_tensor(m,dtype=torch.float32) for m in (z,s)]
    y = torch.as_tensor(labels,dtype=torch.long)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = ParisotGCN(z.shape[1])
    optimizer = torch.optim.Adam(model.parameters(),lr=.005)
    for epoch in range(150):
        # TF1 Adam applies epsilon before bias correction.
        optimizer.param_groups[0]['eps'] = 1e-8/np.sqrt(1-.999**(epoch+1))
        optimizer.zero_grad()
        logits = model(z,s)
        loss = F.cross_entropy(logits[train],y)+5e-4*.5*model.first.square().sum()
        loss.backward()
        optimizer.step()
    return model.eval()


def predict_parisot(model, z, s):
    z,s = [torch.as_tensor(m,dtype=torch.float32) for m in (z,s)]
    with torch.no_grad():
        score = model(z,s).softmax(dim=1)[:,1].numpy()
    assert np.isfinite(score).all()
    return score


def parisot_fit(x, site, sex, train, labels, n_features=2000):
    z,s = parisot_design(x,site,sex,train,labels,n_features)
    score = predict_parisot(train_parisot(z,s,train,labels),z,s)
    return score,score>.5


def read_timeseries(ids, atlas):
    folder = a.ROOT/f'rois_{atlas}'
    return [np.loadtxt(folder/f'{i}_rois_{atlas}.1D',skiprows=1) for i in ids]


def correlation(timeseries):
    return ConnectivityMeasure(kind='correlation').fit_transform(timeseries)


def fit_tpe(corr, train):
    # Fold-specific fit and vectorization follow the upstream TPE path.
    return ConnectivityMeasure(kind='tangent',vectorize=True,discard_diagonal=False).fit(corr[train])


def tpe(corr, train):
    return fit_tpe(corr,train).transform(corr)


def fit_kunda_ridge(x, site, phenotypes, train, labels, dimensions):
    # Same unsupervised MIDA objective as upstream MIDA.py (augment=False, y=None).
    n = len(x)
    distances = squareform(pdist(x,metric='sqeuclidean'))
    width2 = .5*np.median(distances[distances>0])
    kernel = np.exp(-distances/(2*width2))
    center = np.eye(n)-1/n
    domain = (site[:,None]==site).astype(float)
    centered_domain = center@domain@center
    alphas = {'alpha':[.25,.5,.75]}
    best = -np.inf
    for mu in (.5,.75,1.):
        operator = kernel@(mu*center-centered_domain)@kernel
        _,vectors = eigh(operator,subset_by_index=[n-max(dimensions),n-1])
        projected = kernel@vectors[:,::-1]
        for h in dimensions:
            design = np.column_stack([projected[:,:h],phenotypes])
            selector = GridSearchCV(RidgeClassifier(random_state=1234),alphas,cv=5)
            selector.fit(design[train],labels)
            if selector.best_score_>best:
                best = selector.best_score_
                fitted,chosen = selector.best_estimator_,design
                projection = vectors[:,::-1][:,:h]
                params = dict(mu=mu,h=h,alpha=fitted.alpha,inner_accuracy=best)
    return fitted,chosen,projection,width2,kernel.mean(axis=0),params


def kunda_ridge(x, site, phenotypes, train, labels, dimensions=(50,150,300)):
    fitted,chosen,_,_,_,params = fit_kunda_ridge(x,site,phenotypes,train,labels,dimensions)
    score = fitted.decision_function(chosen)
    assert np.isfinite(score).all()
    return score,fitted.predict(chosen),params
