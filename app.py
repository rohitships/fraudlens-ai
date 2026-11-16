"""FraudLens AI: Streamlit + Spark MLlib fraud analytics platform."""
import math
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_auc_score)

st.set_page_config(page_title="FraudLens AI | Big Data Fraud Intelligence",page_icon="🛡️",layout="wide",initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
FEAT = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],div[data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:#f4f7fb;}
[data-testid="stSidebar"] {background:#0d1b30;}
[data-testid="stSidebar"] * {color:#e3eef9 !important;}
[data-testid="stSidebar"] div[data-baseweb="select"] * {color:#0d1b30 !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {border-radius:10px;padding:6px 10px;}
.block-container {padding-top:1.6rem;max-width:1420px;}
h1,h2,h3 {letter-spacing:-.035em;color:#10233c;}
h1 {font-size:2.25rem !important;}
section[data-testid="stSidebar"] .block-container {padding-top:1rem;}
div[data-testid="stMetric"] {background:white;padding:18px 20px;border:1px solid #e3ebf4;border-radius:16px;box-shadow:0 5px 20px #1e38550b;}
div[data-testid="stMetricValue"] {font-weight:800;color:#132d4b;font-size:1.65rem;}
div[data-testid="stMetricLabel"] {color:#71839c;}
div[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#e1e9f3;}
.stButton button[kind="primary"] {background:#1f72d7;border:0;border-radius:10px;}
.hero {padding:29px 32px;border-radius:19px;background:linear-gradient(112deg,#102a4b,#164b77 65%,#0b6f9b);color:white;margin-bottom:22px;}
.hero h1,.hero h2,.hero p {color:white !important;margin:0;}
.hero p {opacity:.83;margin-top:9px;}
.eyebrow {text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;font-weight:800;color:#70caff;}
.info {padding:15px 18px;background:#eaf3fd;border-left:4px solid #338bea;border-radius:10px;color:#274969;margin:12px 0;}
.kicker {font-weight:700;color:#44779f;font-size:.85rem;letter-spacing:.035em;}
.small {font-size:.85rem;color:#74849b;}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def example_data():
    """Clearly labeled synthetic data ONLY for UI walkthroughs, never real model validation."""
    rng=np.random.default_rng(8)
    n=2200
    x=pd.DataFrame({f"V{i}":rng.normal(0,1,n) for i in range(1,29)})
    x.insert(0,"Time",np.sort(rng.uniform(0,172800,n)))
    x["Amount"]=np.round(rng.lognormal(3.3,1.25,n),2)
    x["Class"]=(rng.random(n)<.035).astype(int)
    x.loc[x.Class.eq(1),"V14"]-=3
    x.loc[x.Class.eq(1),"Amount"]*=1.9
    return x

def validate_data(df):
    missing=[c for c in FEAT+["Class"] if c not in df.columns]
    if missing: raise ValueError("Required columns missing: "+", ".join(missing))
    out=df[FEAT+["Class"]].copy()
    for c in out.columns: out[c]=pd.to_numeric(out[c],errors="coerce")
    out=out.replace([np.inf,-np.inf],np.nan)
    dropped=int(out.isna().any(axis=1).sum())
    out=out.dropna()
    out=out[out["Class"].isin([0,1])]
    out["Class"]=out["Class"].astype(int)
    if out.empty: raise ValueError("No valid rows after numeric validation")
    return out,dropped

@st.cache_resource(show_spinner="Initializing a local Spark runtime...")
def get_spark():
    os.environ.setdefault("SPARK_LOCAL_IP","127.0.0.1")
    os.environ.setdefault("PYSPARK_PYTHON",os.environ.get("PYTHON","python3"))
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[2]")
            .appName("FraudLens-Fraud-Analytics")
            .config("spark.driver.memory","1g")
            .config("spark.sql.shuffle.partitions","4")
            .config("spark.ui.enabled","false")
            .getOrCreate())

def make_spark_data(pdf):
    spark=get_spark()
    # Schema inferred from validated numeric dataframe; bounded size recommended on Cloud.
    return spark.createDataFrame(pdf)

def train_ml(pdf, algo, seed, test_pct):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.sql import functions as F
    spark=get_spark()
    sdf=make_spark_data(pdf)
    # Split with a seeded per-class randomized order to retain rare positives.
    from pyspark.sql.window import Window
    ranked=sdf.withColumn("_rnd",F.rand(seed)).withColumn("_rn",F.row_number().over(Window.partitionBy("Class").orderBy("_rnd")))
    counts=sdf.groupBy("Class").count().collect()
    cnt={int(r['Class']):int(r['count']) for r in counts}
    if any(cnt.get(c,0)<10 for c in [0,1]):
        raise ValueError("Need at least 10 examples of each class for an illustrative split; upload real labeled dataset")
    limits={k:max(1,min(v-1,int(v*(1-test_pct)))) for k,v in cnt.items()}
    train=ranked.filter(((F.col("Class")==0)&(F.col("_rn")<=limits[0]))|((F.col("Class")==1)&(F.col("_rn")<=limits[1]))).drop("_rnd","_rn")
    test=ranked.filter(((F.col("Class")==0)&(F.col("_rn")>limits[0]))|((F.col("Class")==1)&(F.col("_rn")>limits[1]))).drop("_rnd","_rn")
    # No target Class among features. Preprocessing fit only on train.
    assembler=VectorAssembler(inputCols=FEAT,outputCol="raw_features")
    if algo=="Logistic Regression":
        classifier=LogisticRegression(featuresCol="features",labelCol="Class",maxIter=35,regParam=.02)
        stages=[assembler,StandardScaler(inputCol="raw_features",outputCol="features",withMean=False,withStd=True),classifier]
    else:
        classifier=RandomForestClassifier(featuresCol="features",labelCol="Class",numTrees=35,maxDepth=6,seed=seed)
        stages=[assembler,RandomForestClassifier(featuresCol="raw_features",labelCol="Class",numTrees=35,maxDepth=6,seed=seed)]
    start=time.time()
    pipe=Pipeline(stages=stages).fit(train)
    preds=pipe.transform(test).select("Class","prediction","probability","Amount","Time")
    # Test set is not used to choose parameters or threshold.
    outs=preds.collect()
    y=np.array([int(r['Class']) for r in outs]); scores=np.array([float(r['probability'][1]) for r in outs])
    yhat=np.array([int(r['prediction']) for r in outs])
    tn,fp,fn,tp=confusion_matrix(y,yhat,labels=[0,1]).ravel()
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    auc=roc_auc_score(y,scores) if len(set(y))==2 else float('nan')
    ap=average_precision_score(y,scores) if len(set(y))==2 else float('nan')
    records=pd.DataFrame([{'actual':int(r['Class']),'prediction':int(r['prediction']), 'risk_score':float(r['probability'][1]),'Amount':float(r['Amount']),'Time':float(r['Time'])} for r in outs])
    return {'model':pipe,'algo':algo,'elapsed':round(time.time()-start,2),'train_rows':sum(limits.values()),'test_rows':len(y),
            'tp':int(tp),'fp':int(fp),'tn':int(tn),'fn':int(fn),'precision':precision,'recall':recall,'f1':f1,'roc_auc':auc,'pr_auc':ap,
            'records':records,'y':y,'scores':scores,'yhat':yhat,'source':st.session_state.get('source','Dataset')}

def chartstyle(fig, height=350):
    fig.update_layout(height=height,margin=dict(l=8,r=8,t=28,b=10),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(family='DM Sans, sans-serif',color='#344e69'),legend_title_text='')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#e8eef6')
    return fig

def page_header(title,subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">FRAUDLENS AI / BIG DATA ANALYTICS</div><h1>{title}</h1><p>{subtitle}</p></div>',unsafe_allow_html=True)

def kpis(items):
    cols=st.columns(len(items))
    for col,(name,value) in zip(cols,items): col.metric(name,value)

def status():
    return "REAL DATASET" if st.session_state.source=="Uploaded CSV" else "SYNTHETIC DEMO DATA"

if 'source' not in st.session_state:
    st.session_state.source='Synthetic demo'
    st.session_state.df=example_data()
if 'run' not in st.session_state: st.session_state.run=None

with st.sidebar:
    st.markdown("## 🛡️ FraudLens AI")
    st.caption("BIG DATA FRAUD INTELLIGENCE")
    st.divider()
    page=st.radio("Navigation",["📊 Overview","🗂️ Dataset Explorer","⚡ Spark & BDA Lab","🧠 MLlib Training","🎯 Fraud Prediction","🌊 Stream Mining"],label_visibility="collapsed")
    st.divider()
    st.markdown("**Dataset source**")
    st.caption(status())
    upload=st.file_uploader("Upload ULB creditcard.csv",type=['csv'],help='The actual dataset is not bundled. Upload your own public ULB data file.')
    if upload is not None:
        if st.button('Load uploaded dataset',type='primary',use_container_width=True):
            try:
                incoming=pd.read_csv(upload)
                clean,dropped=validate_data(incoming)
                st.session_state.df=clean
                st.session_state.source='Uploaded CSV'
                st.session_state.run=None
                st.success(f"Loaded {len(clean):,} rows; dropped {dropped} invalid rows")
            except Exception as e: st.error(str(e))
    if st.button('Reset to synthetic preview',use_container_width=True):
        st.session_state.df=example_data();st.session_state.source='Synthetic demo';st.session_state.run=None;st.rerun()
    st.caption('For real MLlib results, upload the ULB dataset and train a model.')
    st.markdown('---')
    st.caption('Apache Spark MLlib • Fraud Analytics')

df=st.session_state.df
n=len(df); n_fraud=int(df.Class.sum());n_normal=n-n_fraud
is_real=st.session_state.source=='Uploaded CSV'
if not is_real:
    st.warning('**Synthetic preview mode:** All displayed transactions and charts are generated examples for exploring the interface. They are not the ULB dataset or measured fraud detection performance.',icon='🧪')

if page=="📊 Overview":
    page_header("Fraud Intelligence Overview","A single command center for transaction patterns, class imbalance and risk analytics.")
    kpis([('Transactions',f'{n:,}'),('Legitimate',f'{n_normal:,}'),('Fraud',f'{n_fraud:,}'),('Fraud rate',f'{100*n_fraud/n:.3f}%')])
    st.write('')
    a,b=st.columns([1,1.35])
    with a:
        st.subheader('Transaction classification')
        counts=pd.DataFrame({'Class':['Legitimate','Fraud'],'Count':[n_normal,n_fraud]})
        fig=px.pie(counts,names='Class',values='Count',hole=.69,color='Class',color_discrete_map={'Legitimate':'#2b81d6','Fraud':'#ec6572'})
        st.plotly_chart(chartstyle(fig),use_container_width=True)
    with b:
        st.subheader('Transaction amount distribution')
        fig=px.histogram(df,x='Amount',color=df.Class.map({0:'Legitimate',1:'Fraud'}),nbins=55,log_y=True,opacity=.8,color_discrete_map={'Legitimate':'#2b81d6','Fraud':'#ec6572'})
        fig.update_layout(xaxis_title='Amount',yaxis_title='Number of transactions (log scale)')
        st.plotly_chart(chartstyle(fig),use_container_width=True)
    a,b=st.columns([1.2,1])
    with a:
        st.subheader('Transaction activity by time')
        tmp=df[['Time','Class']].copy();tmp['Hour']=(tmp.Time//3600).astype(int)
        trend=tmp.groupby('Hour').agg(Transactions=('Class','size'),Frauds=('Class','sum')).reset_index()
        fig=go.Figure();fig.add_trace(go.Scatter(x=trend.Hour,y=trend.Transactions,name='Transactions',mode='lines',line=dict(color='#2d82cf',width=3)))
        fig.add_trace(go.Scatter(x=trend.Hour,y=trend.Frauds,name='Frauds',mode='lines',line=dict(color='#e35e6e',width=2),yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y',side='right',showgrid=False,title='Fraud events'),xaxis_title='Elapsed hour')
        st.plotly_chart(chartstyle(fig),use_container_width=True)
    with b:
        st.subheader('What the dataset teaches')
        st.markdown('''<div class="info"><b>Volume</b> — transaction records analyzed with structured operations.<br><b>Variety</b> — numeric transaction features and labels.<br><b>Veracity</b> — validate schema, missing data and class imbalance.<br><b>Value</b> — identify risky transaction patterns.<br><b>Velocity</b> — relevant to production streams; this dataset is historical batch data.</div>''',unsafe_allow_html=True)
        st.info('With the original ULB dataset, a model predicting **every transaction as legitimate** gets ~99.827% accuracy yet 0% fraud recall.')
    st.subheader('Transaction preview')
    st.dataframe(df[['Time','Amount','Class','V1','V2','V14']].head(12),use_container_width=True,hide_index=True)

elif page=="🗂️ Dataset Explorer":
    page_header("Dataset Explorer","Inspect fraud prevalence, features, missing values and transaction distributions.")
    kpis([('Rows',f'{n:,}'),('Features',str(len(FEAT))),('Fraud labels',f'{n_fraud:,}'),('Missing values',f'{int(df.isna().sum().sum()):,}')])
    a,b,c=st.columns([1,1,1])
    with a: kind=st.selectbox('Transaction class',['Both','Legitimate','Fraud'])
    with b: amt=st.slider('Maximum amount',0.0,float(max(1,df.Amount.max())),float(max(1,df.Amount.max())))
    with c: limit=st.selectbox('Preview rows',[20,50,100,250])
    filtered=df[df.Amount<=amt]
    if kind!='Both':filtered=filtered[filtered.Class==(1 if kind=='Fraud' else 0)]
    st.caption(f'{len(filtered):,} matching records; showing up to {limit} rows')
    st.dataframe(filtered.head(limit),use_container_width=True,hide_index=True)
    st.download_button('Download current preview CSV',filtered.head(limit).to_csv(index=False),file_name='fraudlens_preview.csv',mime='text/csv')
    a,b=st.columns(2)
    with a:
        st.subheader('Feature distributions')
        feature=st.selectbox('Numeric feature',FEAT,index=FEAT.index('Amount'))
        f=px.histogram(df,x=feature,color=df.Class.map({0:'Legitimate',1:'Fraud'}),nbins=50,opacity=.7,color_discrete_map={'Legitimate':'#2b81d6','Fraud':'#ec6572'})
        st.plotly_chart(chartstyle(f),use_container_width=True)
    with b:
        st.subheader('Descriptive statistics')
        st.dataframe(df[['Amount','Time','V1','V2','V14']].describe().round(2),use_container_width=True)
    with st.expander('📘 BDA concept: Structured data and PCA'):
        st.write('The ULB features V1–V28 were anonymized through PCA by the data creators. Loading them does **not** mean this app itself performed PCA. Spark DataFrames offer schema-aware, partitionable processing for this structured dataset.')

elif page=="⚡ Spark & BDA Lab":
    page_header('Spark & BDA Laboratory','Demonstrate real PySpark DataFrames, partitioning, transformations, actions and query planning.')
    st.info('This tab executes real Spark operations on the currently loaded data when Java and PySpark are available. Local Spark is a **single-machine** setup, not a multi-node cluster.')
    if st.button('Initialize Spark and analyze data',type='primary'):
        try:
            with st.spinner('Starting Spark and executing jobs...'):
                spark=get_spark(); sdf=make_spark_data(df)
                t=time.time(); grouped=sdf.groupBy('Class').count().orderBy('Class').collect(); secs=time.time()-t
                plan=sdf.filter('Class = 1').groupBy('Class').count()._jdf.queryExecution().toString()
                st.session_state.spark_result={'version':spark.version,'master':spark.sparkContext.master,'partitions':sdf.rdd.getNumPartitions(),'grouped':[(int(r['Class']),int(r['count'])) for r in grouped], 'seconds':secs,'plan':plan}
        except Exception as e: st.error(f'Spark was not available: {e}. See README for Java troubleshooting. This is not a successful Spark run.')
    if 'spark_result' in st.session_state:
        r=st.session_state.spark_result
        kpis([('Spark version',r['version']),('Master',r['master']),('Partitions',str(r['partitions'])),('Aggregation time',f"{r['seconds']:.2f}s")])
        st.write('**Actual Spark groupBy count output:**')
        st.dataframe(pd.DataFrame(r['grouped'],columns=['Class','Count']),hide_index=True)
        with st.expander('Actual Spark query plan'):
            st.code(r['plan'][:16000],language='text')
    else: st.caption('No Spark job executed in this session yet.')
    st.subheader('Transformations vs actions')
    a,b=st.columns(2)
    with a:
        st.success('**Transformations** — build a plan')
        st.code('fraud_df = sdf.filter("Class = 1")\nselected = fraud_df.select("Amount")',language='python')
    with b:
        st.warning('**Actions** — trigger execution')
        st.code('number_of_frauds = selected.count()\nselected.show(5)',language='python')
    st.subheader('Partition processing concept')
    p=st.slider('Conceptual partitions',2,12,4)
    blocks=[sum(1 for i in range(n) if i%p==k) for k in range(p)]
    fig=px.bar(x=[f'P{i+1}' for i in range(p)],y=blocks,labels={'x':'Simulated partition','y':'Record count'},color_discrete_sequence=['#3d87d0'])
    st.plotly_chart(chartstyle(fig,260),use_container_width=True)
    st.caption('Partition chart above is a conceptual even-distribution simulation, not the actual Spark partition sizes. Use actual Spark details above for verified execution.')
    with st.expander('🔎 Why use Spark instead of standard Python processing?'):
        st.write('Spark enables partitioned computation, parallel execution, lazy transformations, DataFrames and scalable MLlib algorithms. A 150 MB dataset does not require a production cluster; this is an educational demonstration of scalable methods.')

elif page=="🧠 MLlib Training":
    page_header('Spark MLlib Model Lab','Train a genuine Spark MLlib classifier and inspect held-out evaluation results.')
    if not is_real: st.error('Real model training is disabled in synthetic preview mode. Upload creditcard.csv to train and report genuine MLlib results.')
    a,b,c=st.columns(3)
    with a:algo=st.selectbox('Algorithm',['Logistic Regression','Random Forest'])
    with b:seed=st.number_input('Random seed',0,99999,42)
    with c:test_pct=st.slider('Test split (%)',15,35,20,5)
    st.caption('Stratified, seeded split. Preprocessing fitted on training data. No resampling or threshold tuning is implemented. Results are educational holdout metrics.')
    if st.button('Train Spark MLlib model',type='primary',disabled=not is_real):
        try:
            with st.spinner('Training real MLlib model on Spark. This may use significant CPU/RAM...'):
                st.session_state.run=train_ml(df,algo,int(seed),test_pct/100)
            st.success('Actual Spark MLlib training completed.')
        except Exception as e: st.error('Training failed: '+str(e))
    r=st.session_state.run
    if r:
        st.markdown(f'**Active model: {r["algo"]}** | Training: {r["train_rows"]:,} rows | Test: {r["test_rows"]:,} rows | Fit & evaluation: {r["elapsed"]:.1f}s')
        kpis([('Precision',f'{r["precision"]:.3f}'),('Recall',f'{r["recall"]:.3f}'),('F1',f'{r["f1"]:.3f}'),('PR-AUC (AP)',f'{r["pr_auc"]:.3f}')])
        a,b=st.columns(2)
        with a:
            cm=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
            fig=px.imshow(cm,text_auto=True,x=['Predicted normal','Predicted fraud'],y=['Actual normal','Actual fraud'],color_continuous_scale='Blues')
            st.plotly_chart(chartstyle(fig,335),use_container_width=True)
        with b:
            prec,rec,_=precision_recall_curve(r['y'],r['scores'])
            fig=go.Figure(go.Scatter(x=rec,y=prec,mode='lines',line=dict(color='#2386d6',width=3)))
            fig.update_layout(xaxis_title='Recall',yaxis_title='Precision',title='Precision-recall curve')
            st.plotly_chart(chartstyle(fig,335),use_container_width=True)
        st.caption('PR-AUC card reports average precision (AP), which is a stepwise summary of the precision-recall curve; it may differ from trapezoidal PR area.')
        with st.expander('Actual test predictions'):
            st.dataframe(r['records'].head(100),use_container_width=True,hide_index=True)
    else: st.info('No trained MLlib model available. Upload the dataset and run a training job.')

elif page=="🎯 Fraud Prediction":
    page_header('Fraud Prediction Studio','Inspect genuine held-out model predictions; never confuse a fraud alert with proof of fraud.')
    r=st.session_state.run
    if not r:
        st.info('Upload the genuine dataset and train a model in **MLlib Training** first. This page does not invent predictions or fraud scores.')
    else:
        records=r['records'];kpis([('Held-out records',f'{len(records):,}'),('Actual fraud',f'{records.actual.sum():,}'),('Predicted fraud',f'{int(records.prediction.sum()):,}'),('Model',r['algo'])])
        choice=st.selectbox('Show test cases',['All test predictions','Actual fraud cases','Incorrect predictions','Predicted fraud'])
        filtered=records
        if choice=='Actual fraud cases':filtered=records[records.actual==1]
        elif choice=='Incorrect predictions':filtered=records[records.actual!=records.prediction]
        elif choice=='Predicted fraud':filtered=records[records.prediction==1]
        st.dataframe(filtered.head(100),use_container_width=True,hide_index=True)
        st.subheader('Interactive score threshold — educational test-set exploration')
        threshold=st.slider('Risk score threshold',0.05,0.95,0.50,0.05)
        pred=(r['scores']>=threshold).astype(int)
        tn,fp,fn,tp=confusion_matrix(r['y'],pred,labels=[0,1]).ravel()
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
        kpis([('Flagged fraud',f'{int(pred.sum()):,}'),('Frauds caught',str(int(tp))),('False alarms',str(int(fp))),('Recall',f'{recall:.2%}')])
        st.warning('Do not tune thresholds using the final test set for a reported unbiased score. This slider is for educational exploration only.')

elif page=="🌊 Stream Mining":
    page_header('Stream Mining Lab','Interactive historical replay and classic Big Data streaming algorithms.')
    t1,t2,t3=st.tabs(['🔄 Transaction replay','🧩 Bloom Filter','📉 Approximation concepts'])
    with t1:
        st.caption('Historical replay, not live financial events, and not Spark Structured Streaming.')
        count=st.slider('Replay first N transactions',10,min(400,n),min(80,n))
        flow=df[['Time','Amount','Class']].head(count).copy()
        flow['event_no']=np.arange(1,len(flow)+1)
        flow['rolling_fraud_count']=flow.Class.rolling(window=20,min_periods=1).sum()
        fig=px.line(flow,x='event_no',y='rolling_fraud_count',labels={'event_no':'Transaction number','rolling_fraud_count':'Fraud labels in recent 20 events'})
        st.plotly_chart(chartstyle(fig),use_container_width=True)
        st.dataframe(flow.tail(20),hide_index=True,use_container_width=True)
    with t2:
        st.markdown('**Bloom Filter**: approximate membership, not a fraud classifier.')
        m=96;k=3
        if 'bloom_values' not in st.session_state:st.session_state.bloom_values=[]
        v=st.text_input('Demo identifier','DEVICE-104')
        if st.button('Insert identifier') and v:
            st.session_state.bloom_values.append(v);st.success('Inserted into educational Bloom Filter')
        import hashlib
        def positions(s):return [int.from_bytes(hashlib.sha256((str(i)+'/'+s).encode()).digest()[:8],'big')%m for i in range(k)]
        bits=[0]*m
        for value in st.session_state.bloom_values:
            for index in positions(value):bits[index]=1
        if v:
            exists=all(bits[index] for index in positions(v))
            st.markdown('**Membership test:** '+('Possibly present' if exists else 'Definitely absent'))
        st.code(' '.join(str(bit) for bit in bits),language='text')
        n_ins=len(st.session_state.bloom_values)
        st.caption(f'Approximate false-positive rate: {(1-math.exp(-k*n_ins/m))**k:.4%} (assumes ideal independent hashes; actual collision behavior varies).')
    with t3:
        st.markdown('**Flajolet–Martin:** estimates distinct identifiers from hashed bit patterns, rather than storing every unique item.')
        st.markdown('**DGIM:** approximates the count of 1s in a recent binary window using power-of-two buckets and timestamps.')
        st.info('FM and DGIM are explained here but are **not implemented** in this version. The working stream demo provides a sliding-window fraud count and Bloom Filter membership check.')

st.markdown('---')
st.caption(f'FraudLens AI • {status()} • Spark MLlib analytics prototype • Never use model predictions as real payment decisions.')
