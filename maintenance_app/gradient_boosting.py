"""Portable classic Gradient Boosting pipelines and exact tree-path explanations."""
import maintenance_ml as ml
import numpy as np
import pandas as pd
import joblib
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

ALGORITHM='GradientBoostingClassifier'
EXPLANATION='Additive tree-path attribution (not SHAP); path order and correlated predictors affect allocations. Contributions describe model log-odds, not physical causes.'

def make_pipeline(names,config,seed,trees):
    numeric=[c for c in names if c not in ml.CAT]
    preprocessing=ColumnTransformer([
        ('numeric',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),numeric),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),ml.CAT)],sparse_threshold=0)
    classifier=GradientBoostingClassifier(loss='log_loss',learning_rate=config['learning_rate'],n_estimators=trees,
        max_depth=config['max_depth'],min_samples_leaf=config['min_samples_leaf'],subsample=config['subsample'],
        random_state=seed,n_iter_no_change=None)
    return Pipeline([('preprocessing',preprocessing),('classifier',classifier)])

def sample_weights(y,mode):
    y=np.asarray(y,dtype=int)
    if mode=='none':return np.ones(len(y))
    counts=np.bincount(y,minlength=2)
    if (counts==0).any():raise ValueError('Both outcomes are required in each training fold.')
    weight=len(y)/(2*counts.astype(float))
    if mode=='sqrt_balanced':weight=np.sqrt(weight)
    elif mode!='balanced':raise ValueError('Unknown weighting policy')
    result=weight[y];return result/result.mean()

def fit(model,X,y,weight_mode):
    model.fit(X,y,classifier__sample_weight=sample_weights(y,weight_mode));return model

def load_models(directory,manifest):
    from pathlib import Path
    # Load only this application's locally generated, trusted pipeline files.
    return [joblib.load(Path(directory)/name) for name in manifest['model_files']]

def predict_models(models,X,calibration):
    raw=np.mean([model.decision_function(X) for model in models],axis=0)
    return raw,ml.calibration_predict(raw,calibration)

def _path_values(model,X):
    prep=model.named_steps['preprocessing'];classifier=model.named_steps['classifier']
    matrix=prep.transform(X).astype(np.float32)
    numeric=prep.transformers_[0][2];imputer=prep.named_transformers_['numeric']
    sources=list(numeric)+[numeric[i] for i in imputer.indicator_.features_]
    encoder=prep.named_transformers_['categorical']
    for col,categories in zip(ml.CAT,encoder.categories_):sources.extend([col]*len(categories))
    positions={name:i for i,name in enumerate(X.columns)};mapping=np.array([positions[name] for name in sources])
    assert matrix.shape[1]==len(mapping)
    attribution=np.zeros((len(X),len(X.columns)))
    baseline=logit(np.clip(classifier.init_.predict_proba(matrix)[:,1],np.finfo(np.float32).eps,1-np.finfo(np.float32).eps))
    for estimator in classifier.estimators_[:,0]:
        tree=estimator.tree_;values=tree.value.reshape(-1);parent=np.full(tree.node_count,-1,dtype=int)
        for branch in [tree.children_left,tree.children_right]:
            mask=branch>=0;parent[branch[mask]]=np.flatnonzero(mask)
        path=estimator.decision_path(matrix).tocoo();keep=path.col!=0
        nodes=path.col[keep];rows=path.row[keep];parents=parent[nodes]
        np.add.at(attribution,(rows,mapping[tree.feature[parents]]),classifier.learning_rate*(values[nodes]-values[parents]))
        baseline+=classifier.learning_rate*values[0]
    np.testing.assert_allclose(baseline+attribution.sum(axis=1),model.decision_function(X),atol=1e-6,rtol=1e-6)
    return baseline,attribution

def driver_explanations(models,X,limit=3):
    parts=[_path_values(model,X) for model in models]
    baseline=np.mean([p[0] for p in parts],axis=0);contributions=np.mean([p[1] for p in parts],axis=0)
    raw=np.mean([m.decision_function(X) for m in models],axis=0)
    np.testing.assert_allclose(baseline+contributions.sum(axis=1),raw,atol=1e-6,rtol=1e-6)
    descriptions=[]
    for i,values in enumerate(contributions):
        row=[]
        for j in np.argsort(-np.abs(values))[:limit]:
            value=X.iloc[i,j];display='not measured / not applicable' if pd.isna(value) else f'{value:.2f}' if isinstance(value,(float,np.floating)) else str(value)
            row.append({'feature':X.columns[j],'value':display,'direction':'raises raw score' if values[j]>0 else 'lowers raw score',
                'contribution':float(values[j]),'method':'tree_path'})
        descriptions.append(row)
    return np.column_stack([contributions,baseline]),descriptions
