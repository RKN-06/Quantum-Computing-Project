"""Reproducible binary/three-class extension; legacy evidence remains unchanged."""
from pathlib import Path
import hashlib
import json
import time
import platform
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

THRESHOLD = 0.0025
TASKS = {'binary': ('target_up', [0, 1]), 'three_class': ('target_three', [0, 1, 2])}
CONFIG = dict(tickers=['AAPL','MSFT','JPM','XOM','JNJ'], train_dates=200,
              validation_dates=80, evaluation_dates=100, threshold=THRESHOLD,
              start='2015-01-01', validation_start='2021-01-01',
              evaluation_start='2023-01-01', end='2025-01-01', seed=42)

def labels(returns):
    """Simple-return bands: DOWN=0, NEUTRAL=1, UP=2; missing stays missing."""
    r = pd.Series(returns)
    out = pd.Series(np.select([r < -THRESHOLD, r > THRESHOLD], [0, 2], default=1), index=r.index, dtype='Int64')
    return out.mask(r.isna())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def read_selected(path, columns):
    return pd.concat([c.loc[c.Ticker.isin(CONFIG['tickers'])] for c in
        pd.read_csv(path, usecols=columns, parse_dates=['Date'], chunksize=200000)], ignore_index=True).sort_values(['Ticker','Date'])

def prepare(root):
    features = json.loads((root/'data/processed/feature_columns.json').read_text())
    assert len(features)==16 and not any('target' in f or 'forward' in f for f in features)
    data = read_selected(root/'data/processed/SP500_features.csv', ['Ticker','Date',*features,'target_up'])
    target = read_selected(root/'data/processed/SP500_targets.csv', ['Ticker','Date','label_end','forward_return_simple','target_three'])
    target['label_end'] = pd.to_datetime(target.label_end)
    data = data.merge(target, on=['Ticker','Date'], validate='one_to_one')
    assert not data.duplicated(['Ticker','Date']).any()
    assert np.isfinite(data[features].to_numpy()).all()
    assert (data.label_end > data.Date).all()
    assert np.array_equal(data.target_up, (data.forward_return_simple>0).astype(int))
    assert np.array_equal(data.target_three, labels(data.forward_return_simple).astype(int))
    frames = {}
    for name, lo, hi, count in [
        ('train',CONFIG['start'],CONFIG['validation_start'],CONFIG['train_dates']),
        ('validation',CONFIG['validation_start'],CONFIG['evaluation_start'],CONFIG['validation_dates']),
        ('evaluation',CONFIG['evaluation_start'],CONFIG['end'],CONFIG['evaluation_dates'])]:
        frame = data.loc[(data.Date>=lo)&(data.Date<hi)&(data.label_end<hi)]
        dates = np.sort(frame.Date.unique())
        chosen = dates[np.linspace(0,len(dates)-1,min(count,len(dates)),dtype=int)]
        frames[name] = frame.loc[frame.Date.isin(chosen)].sort_values(['Date','Ticker']).reset_index(drop=True)
    frames['development'] = pd.concat([frames['train'],frames['validation']],ignore_index=True).sort_values(['Date','Ticker'])
    assert frames['train'].label_end.max()<frames['validation'].Date.min()
    assert frames['development'].label_end.max()<frames['evaluation'].Date.min()
    for frame in frames.values():
        assert set(frame.Ticker)==set(CONFIG['tickers'])
        for target_name, classes in TASKS.values():
            assert set(frame[target_name])==set(classes), 'Missing class: revise protocol before drawing conclusions'
            assert frame[target_name].value_counts().min()>=20, 'Insufficient per-class support'
    return features, frames

def metrics(y, pred):
    return dict(accuracy=accuracy_score(y,pred), balanced_accuracy=balanced_accuracy_score(y,pred),
        macro_f1=f1_score(y,pred,average='macro',zero_division=0),
        macro_precision=precision_score(y,pred,average='macro',zero_division=0),
        macro_recall=recall_score(y,pred,average='macro',zero_division=0))

def grids(quantum):
    result = {'dummy':[{}], 'linear':list(ParameterGrid({'C':[0.1,1,10]})),
        'rbf':list(ParameterGrid({'C':[0.1,1,10],'gamma':['scale',0.01,0.1]})),
        'polynomial':list(ParameterGrid({'C':[0.1,1,10],'degree':[2,3],'gamma':['scale']}))}
    if quantum: result['qsvm'] = list(ParameterGrid({'C':[0.1,1,10]}))
    return result

def model(name, params):
    if name=='dummy': return DummyClassifier(strategy='most_frequent')
    return SVC(kernel={'linear':'linear','rbf':'rbf','polynomial':'poly','qsvm':'precomputed'}[name],
               class_weight='balanced', decision_function_shape='ovr', **params)

def transforms(d):
    steps=[('scale',StandardScaler())]
    if d != 16:
        steps += [('pca',PCA(n_components=d,svd_solver='full')),('angles',MinMaxScaler(feature_range=(-np.pi,np.pi),clip=True))]
    return Pipeline(steps)

def kernel(A,B,d):
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    from qiskit.quantum_info import Statevector
    circuit=zz_feature_map(d,reps=1,entanglement='linear')
    k=FidelityStatevectorKernel(feature_map=circuit,shots=None)
    start=time.perf_counter(); K=k.evaluate(A); fit_time=time.perf_counter()-start
    start=time.perf_counter(); cross=k.evaluate(B,A); cross_time=time.perf_counter()-start
    assert K.shape==(len(A),len(A)) and cross.shape==(len(B),len(A))
    assert np.isfinite(K).all() and np.isfinite(cross).all()
    assert np.allclose(K,K.T,atol=1e-9) and np.allclose(np.diag(K),1,atol=1e-9)
    assert min(K.min(),cross.min())>=-1e-9 and max(K.max(),cross.max())<=1+1e-9
    # Check all eigenvalues on the actual training Gram matrix.
    eigen_min=float(np.linalg.eigvalsh(K).min()); assert eigen_min>=-1e-8
    a=Statevector.from_instruction(circuit.assign_parameters(A[0])).data
    b=Statevector.from_instruction(circuit.assign_parameters(B[0])).data
    assert np.isclose(cross[0,0],abs(np.vdot(a,b))**2,atol=1e-9)
    return K,cross,dict(train_kernel_seconds=fit_time,test_kernel_seconds=cross_time,
        kernel_bytes=K.nbytes+cross.nbytes,min_eigenvalue=eigen_min,depth=circuit.depth(),
        cx_gates=circuit.count_ops().get('cx',0))

def run(root, stage):
    root=Path(root); quantum=stage==4; out=root/f'results/extension/stage{stage}'; out.mkdir(parents=True,exist_ok=True)
    features, frames=prepare(root)
    for name,frame in frames.items():
        frame[['Ticker','Date','label_end','target_up','target_three']].to_csv(out/f'{name}_sample.csv',index=False)
    support=[]
    for split,frame in frames.items():
        for task,(target_name,classes) in TASKS.items():
            for c in classes:
                support.append(dict(split=split,task=task,label=c,rows=int((frame[target_name]==c).sum()),dates=frame.Date.nunique()))
    pd.DataFrame(support).to_csv(out/'class_support.csv',index=False)
    rows=[]; predictions=[]; trials=[]; diagnostics=[]; preprocessing=[]
    train,valid,dev,test=[frames[k] for k in ['train','validation','development','evaluation']]
    for d in ([2,4] if quantum else [16]):
        pre=transforms(d); A=pre.fit_transform(train[features]); V=pre.transform(valid[features])
        assert np.allclose(pre.named_steps['scale'].mean_,train[features].mean())
        if quantum:
            K,KV,diag=kernel(A,V,d); diagnostics.append(dict(phase='selection',dimension=d,**diag))
        chosen={}
        for task,(target_name,classes) in TASKS.items():
            for name,grid in grids(quantum).items():
                best=-np.inf
                for params in grid:
                    m=model(name,params); m.fit(K if name=='qsvm' else A,train[target_name])
                    pred=m.predict(KV if name=='qsvm' else V)
                    score=balanced_accuracy_score(valid[target_name],pred)
                    trials.append(dict(task=task,dimension=d,model=name,params=json.dumps(params),validation_balanced_accuracy=score))
                    if score>best: best=score; chosen[(task,name)]=(params.copy(),score)
        # No evaluation labels or scores enter the selection loop above.
        start=time.perf_counter(); pre=transforms(d); A=pre.fit_transform(dev[features])
        before=pre.named_steps['scale'].mean_.copy(); B=pre.transform(test[features]); prep=time.perf_counter()-start
        assert np.array_equal(before,pre.named_steps['scale'].mean_)
        assert np.allclose(before,dev[features].mean())
        pre_record=dict(dimension=d,scaler_mean=before.tolist(),scaler_scale=pre.named_steps['scale'].scale_.tolist())
        if quantum:
            assert np.max(np.abs(A))<=np.pi+1e-9 and np.max(np.abs(B))<=np.pi+1e-9
            reduced=pre.named_steps['pca'].transform(pre.named_steps['scale'].transform(test[features]))
            angle=pre.named_steps['angles']
            pre_record.update(pca_components=pre.named_steps['pca'].components_.tolist(),
                variance=pre.named_steps['pca'].explained_variance_ratio_.tolist(),angle_min=angle.data_min_.tolist(),
                angle_max=angle.data_max_.tolist(),clip_fraction=float(((reduced<angle.data_min_)|(reduced>angle.data_max_)).mean()))
            K,KT,diag=kernel(A,B,d); diagnostics.append(dict(phase='final',dimension=d,**diag))
        preprocessing.append(pre_record)
        for task,(target_name,classes) in TASKS.items():
            for name in grids(quantum):
                params,score=chosen[(task,name)]; m=model(name,params)
                start=time.perf_counter(); m.fit(K if name=='qsvm' else A,dev[target_name]); fit=time.perf_counter()-start
                start=time.perf_counter(); pred=m.predict(KT if name=='qsvm' else B); predict=time.perf_counter()-start
                assert set(m.classes_)==set(classes)
                rows.append(dict(task=task,dimension=d,model=name,**metrics(test[target_name],pred),
                    validation_balanced_accuracy=score,params=json.dumps(params),fit_seconds=fit,predict_seconds=predict,
                    preprocessing_seconds=prep,train_kernel_seconds=diag['train_kernel_seconds'] if name=='qsvm' else 0,
                    test_kernel_seconds=diag['test_kernel_seconds'] if name=='qsvm' else 0))
                p=test[['Ticker','Date']].copy(); p['task']=task; p['dimension']=d; p['model']=name
                p['actual']=test[target_name].to_numpy(); p['prediction']=pred; predictions.append(p)
    result=pd.DataFrame(rows); result.to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(trials).to_csv(out/'validation_trials.csv',index=False)
    metadata=dict(config=CONFIG,features=features,python=platform.python_version(),
        packages={p:version(p) for p in ['numpy','pandas','scikit-learn','qiskit','qiskit-machine-learning']},
        input_sha256={p:digest(root/'data/processed'/p) for p in ['SP500_features.csv','SP500_targets.csv','feature_columns.json']},
        code_sha256=digest(__file__), evaluation_status='Exploratory, previously inspected period',
        kernel_reuse='Same label-independent kernel reused across tasks; per-model costs allocate full kernel build, do not sum to actual workflow time')
    for name,obj in [('metadata',metadata),('kernel_diagnostics',diagnostics),('preprocessing',preprocessing)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    return result, pd.DataFrame(support)

def audit(root):
    root=Path(root); evidence=root/'reports/multiclass/evidence'; evidence.mkdir(parents=True,exist_ok=True)
    all_metrics=[]; matches=[]; manifest={}
    for stage in [3,4]:
        out=root/f'results/extension/stage{stage}'
        table=pd.read_csv(out/'metrics.csv'); pred=pd.read_csv(out/'predictions.csv')
        for row in table.to_dict('records'):
            p=pred.loc[(pred.task==row['task'])&(pred.dimension==row['dimension'])&(pred.model==row['model'])]
            assert len(p)==500 and not p.duplicated(['Ticker','Date']).any()
            for metric,value in metrics(p.actual,p.prediction).items():
                assert np.isclose(value,row[metric],atol=1e-12)
        all_metrics.append(table.assign(stage=stage))
        # Freeze compact results for reproducible review without rerunning model fits.
        for filename in ['metrics.csv','predictions.csv','class_support.csv','validation_trials.csv','metadata.json','kernel_diagnostics.json','preprocessing.json','train_sample.csv','validation_sample.csv','evaluation_sample.csv']:
            source=out/filename; dest=evidence/f'stage{stage}_{filename}'
            dest.write_bytes(source.read_bytes()); manifest[dest.name]=digest(dest)
        if stage==4:
            for (task,d),group in table.groupby(['task','dimension']):
                classical=group.loc[group.model.isin(['linear','rbf','polynomial'])].sort_values('validation_balanced_accuracy',ascending=False,kind='stable').iloc[0]
                q=group.loc[group.model=='qsvm'].iloc[0]
                matches.append(dict(task=task,dimension=int(d),classical_model=classical.model,
                    classical_balanced_accuracy=classical.balanced_accuracy,qsvm_balanced_accuracy=q.balanced_accuracy,
                    difference_pp=100*(q.balanced_accuracy-classical.balanced_accuracy)))
    (evidence/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    comparison=pd.DataFrame(matches); comparison.to_csv(evidence.parent/'comparison.csv',index=False)
    return pd.concat(all_metrics,ignore_index=True), comparison

def figures(root):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    root=Path(root); out=root/'reports/multiclass'; out.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv(out/'evidence/stage4_metrics.csv'); pred=pd.read_csv(out/'evidence/stage4_predictions.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,task in zip(axes,TASKS):
        table.loc[table.task==task].pivot(index='model',columns='dimension',values='balanced_accuracy').plot.bar(ax=ax)
        ax.set_ylim(0,1); ax.set_title(task.replace('_',' ').title()); ax.set_ylabel('Balanced accuracy')
    fig.tight_layout(); fig.savefig(out/'matched_balanced_accuracy.png',dpi=150); plt.show()
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    for i,(task,(_,classes)) in enumerate(TASKS.items()):
        for j,d in enumerate([2,4]):
            p=pred.loc[(pred.task==task)&(pred.dimension==d)&(pred.model=='qsvm')]
            names=['Down/flat','Up'] if task=='binary' else ['Down','Neutral','Up']
            ConfusionMatrixDisplay(confusion_matrix(p.actual,p.prediction,labels=classes),display_labels=names).plot(ax=axes[i,j],colorbar=False)
            axes[i,j].set_title(f'{task}, {d} qubits')
    fig.tight_layout(); fig.savefig(out/'qsvm_confusion_matrices.png',dpi=150); plt.show()

def self_test():
    r=pd.Series([-0.01,-THRESHOLD,0,THRESHOLD,0.01,np.nan])
    assert labels(r).iloc[:5].tolist()==[0,1,1,1,2]
    assert pd.isna(labels(r).iloc[-1])
    assert np.isclose(metrics([0,1,2],[0,1,2])['macro_f1'],1)

self_test()
