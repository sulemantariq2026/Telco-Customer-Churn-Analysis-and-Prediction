import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Churn Risk Advisor', layout='wide')
st.markdown(
    """<style>
    div[data-testid="stMetric"] {background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.2); padding: 12px 16px;
        border-radius: 10px;}
    </style>""", unsafe_allow_html=True)

RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges',
            'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
PAYMENTS = ['Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)']
COLORS = {'HIGH': '#d64545', 'WATCH': '#e0a030', 'LOW': '#2e9e6b'}


@st.cache_resource            # load once per server, not on every click
def load_artifacts():
    model = joblib.load('churn_model.joblib')
    with open('model_meta.json') as f:
        meta = json.load(f)
    return model, meta


def prepare_input(raw, columns):
    X = pd.get_dummies(raw[RAW_COLS])              # no drop_first at serving
    return X.reindex(columns=columns, fill_value=0).astype(float)


model, meta = load_artifacts()
COLS = meta['feature_columns']


def score(raw):
    return model.predict_proba(prepare_input(raw, COLS))[:, 1]


def band_of(p, t):
    return 'HIGH' if p >= t else 'WATCH' if p >= t / 2 else 'LOW'


def pkr(x):
    return f'PKR {x:,.0f}'


def style(fig, height=300):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    return fig


# ---------------- Sidebar: one customer ----------------
with st.sidebar:
    st.header('Customer profile')
    tenure = st.slider('Tenure (months)', 0, 72, 4)
    contract = st.selectbox('Contract',
                            ['Month-to-month', 'One year', 'Two year'])
    monthly = st.number_input('Monthly charges (USD)', 18.0, 120.0, 85.0,
                              step=0.5)
    internet = st.selectbox('Internet service', ['Fiber optic', 'DSL', 'No'])
    services = []
    if internet != 'No':
        services = st.multiselect('Add-on services', SERVICES,
                                  default=['StreamingTV'])
    phone = st.radio('Phone service', ['Yes', 'No'], horizontal=True)
    multi = 'No phone service'
    if phone == 'Yes':
        multi = st.radio('Multiple lines', ['No', 'Yes'], horizontal=True)
    payment = st.selectbox('Payment method', PAYMENTS)
    paperless = st.radio('Paperless billing', ['Yes', 'No'], horizontal=True)
    with st.expander('Demographics'):
        gender = st.radio('Gender', ['Female', 'Male'], horizontal=True)
        senior = st.checkbox('Senior citizen')
        partner = st.checkbox('Has partner')
        dependents = st.checkbox('Has dependents')
    st.divider()
    st.subheader('Decision settings')
    threshold = st.slider('Contact threshold', 0.05, 0.95,
                          float(meta['threshold']), 0.05)
    with st.expander('Cost assumptions (PKR)'):
        LOSS = st.number_input('Cost of losing a customer', 0, 100000, 6000,
                               step=500)
        CALL = st.number_input('Cost of one retention call', 0, 50000, 1000,
                               step=100)
        SAVE = st.slider('Share of called customers saved', 0.05, 0.95,
                         0.50, 0.05)
        if SAVE * LOSS > 0:
            st.caption(f'Break-even probability for a call: '
                       f'{CALL / (SAVE * LOSS):.0%}')

row = {'gender': gender, 'SeniorCitizen': int(senior),
       'Partner': 'Yes' if partner else 'No',
       'Dependents': 'Yes' if dependents else 'No', 'tenure': tenure,
       'PhoneService': phone, 'MultipleLines': multi,
       'InternetService': internet, 'Contract': contract,
       'PaperlessBilling': paperless, 'PaymentMethod': payment,
       'MonthlyCharges': monthly,
       'TotalCharges': tenure * monthly}           # derived, not typed
for s in SERVICES:
    row[s] = ('No internet service' if internet == 'No'
              else 'Yes' if s in services else 'No')
customer = pd.DataFrame([row])

# ---------------- Main area ----------------
st.title('Customer Churn Risk Advisor')
st.caption(f"{meta['model_name']}  |  CV AUC {meta['cv_auc']:.3f} "
           f"+/- {meta['cv_auc_std']:.3f}  |  Decision support only")

tab1, tab2, tab3 = st.tabs(['Dashboard', 'Batch scoring', 'Model & About'])

# ======================= Tab 1: one customer =======================
with tab1:
    p = float(score(customer)[0])
    band = band_of(p, threshold)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric('Churn probability', f'{p:.0%}')
    k2.metric('Risk band', band)
    k3.metric('Action', 'Contact now' if band == 'HIGH' else 'No action')
    k4.metric('Expected loss if ignored', pkr(p * LOSS))
    k5.metric('Expected value of a call', pkr(p * SAVE * LOSS - CALL))

    left, right = st.columns([1, 1])
    with left:
        gauge = go.Figure(go.Indicator(
            mode='gauge+number', value=p * 100,
            number={'suffix': '%', 'valueformat': '.0f'},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': COLORS[band]},
                   'steps': [
                       {'range': [0, threshold * 50],
                        'color': 'rgba(46,158,107,.25)'},
                       {'range': [threshold * 50, threshold * 100],
                        'color': 'rgba(224,160,48,.25)'},
                       {'range': [threshold * 100, 100],
                        'color': 'rgba(214,69,69,.25)'}],
                   'threshold': {'line': {'color': 'gray', 'width': 4},
                                 'value': threshold * 100}}))
        st.plotly_chart(style(gauge, 280), use_container_width=True)
        st.caption(f'Grey marker = contact threshold ({threshold:.0%}). '
                   f'Probability does not change when you move it; '
                   f'only the band does.')

    with right:
        st.subheader('Risk flags in this profile')
        flags = []
        if contract == 'Month-to-month':
            flags.append('Month-to-month contract: no commitment')
        if tenure < 12:
            flags.append(f'Short tenure ({tenure} months)')
        if payment == 'Electronic check':
            flags.append('Pays by electronic check')
        if internet == 'Fiber optic':
            flags.append('Fiber optic plan (higher churn in the Telco data)')
        if internet != 'No' and 'TechSupport' not in services:
            flags.append('No tech support')
        if internet != 'No' and 'OnlineSecurity' not in services:
            flags.append('No online security')
        if monthly >= 80:
            flags.append(f'High monthly bill (USD {monthly:.0f})')
        if senior:
            flags.append('Senior citizen')
        if flags:
            for f_ in flags:
                st.markdown(f':red[●] {f_}')
        else:
            st.markdown(':green[●] No major risk flags in this profile')
        st.caption('Flags describe the profile; they are not the model\'s '
                   'explanation.')

    # ---------------- what-if ----------------
    st.subheader('What would change the risk?')
    changes = [('Contract', c, f'Move to {c} contract')
               for c in ['One year', 'Two year'] if c != contract]
    if payment != 'Credit card (automatic)':
        changes.append(('PaymentMethod', 'Credit card (automatic)',
                        'Switch to automatic credit card'))
    if internet != 'No':
        for svc, label in [('TechSupport', 'Add tech support'),
                           ('OnlineSecurity', 'Add online security')]:
            if svc not in services:
                changes.append((svc, 'Yes', label))
    changes.append(('MonthlyCharges', round(monthly * 0.9, 2),
                    '10% lower monthly charge'))

    rows = []
    for feature, value, label in changes:
        alt = customer.copy()
        alt[feature] = value
        if feature == 'MonthlyCharges':
            alt['TotalCharges'] = alt['tenure'] * value
        q = float(score(alt)[0])
        rows.append({'Change': label, 'New probability': round(q, 3),
                     'Difference': round(q - p, 3)})
    rows.sort(key=lambda r: r['Difference'])

    w1, w2 = st.columns([1, 1])
    with w1:
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)
    with w2:
        fig = go.Figure(go.Bar(
            x=[r['Difference'] * 100 for r in rows],
            y=[r['Change'] for r in rows], orientation='h',
            marker_color=[COLORS['LOW'] if r['Difference'] < 0
                          else COLORS['HIGH'] for r in rows]))
        fig.update_layout(title='Change in churn probability (points)',
                          yaxis={'autorange': 'reversed'})
        st.plotly_chart(style(fig, 260), use_container_width=True)
    st.caption('Associations learned from data, not guaranteed effects.')

    best = rows[0]
    if band == 'HIGH':
        advice = (f'Priority call this week. Strongest tested lever: '
                  f'**{best["Change"]}** ({best["Difference"] * 100:+.1f} '
                  f'points).' if best['Difference'] < 0 else
                  'Priority call this week. No tested lever lowers the risk.')
        st.error(advice)
    elif band == 'WATCH':
        st.warning('Add to the watch list and review at the next billing '
                   'cycle.')
    else:
        st.success('No action needed. Keep standard service.')

    # ---------------- tenure curve ----------------
    st.subheader('Risk over tenure for this plan')
    curve = pd.DataFrame([row] * 73)
    curve['tenure'] = range(73)
    curve['TotalCharges'] = curve['tenure'] * curve['MonthlyCharges']
    curve['p'] = score(curve)
    line = go.Figure(go.Scatter(x=curve['tenure'], y=curve['p'],
                                mode='lines', name='Churn probability'))
    line.add_trace(go.Scatter(x=[tenure], y=[p], mode='markers',
                              marker=dict(size=12, color=COLORS[band]),
                              name='This customer'))
    line.add_hline(y=threshold, line_dash='dash', line_color='gray')
    line.update_layout(xaxis_title='Tenure (months)',
                       yaxis=dict(title='P(churn)', tickformat='.0%',
                                  range=[0, 1]))
    st.plotly_chart(style(line, 300), use_container_width=True)
    st.caption('Same customer, same plan, only tenure changes. '
               'Dashed line = contact threshold.')

    # ---------------- LR explanation ----------------
    final = model[-1] if hasattr(model, 'steps') else model
    if hasattr(model, 'steps') and hasattr(final, 'coef_'):
        with st.expander('Why this score? (Logistic Regression)'):
            z = model[:-1].transform(prepare_input(customer, COLS))[0]
            contrib = pd.Series(final.coef_[0] * z, index=COLS)
            top = contrib.reindex(
                contrib.abs().sort_values(ascending=False).index).head(8)
            bars = go.Figure(go.Bar(
                x=top.values, y=top.index, orientation='h',
                marker_color=[COLORS['HIGH'] if v > 0 else COLORS['LOW']
                              for v in top.values]))
            bars.update_layout(yaxis={'autorange': 'reversed'})
            st.plotly_chart(style(bars, 330), use_container_width=True)
            st.caption('Positive bars push toward churn, negative away.')

# ======================= Tab 2: batch scoring =======================
with tab2:
    st.write('Upload a CSV with the original Telco columns, or use the '
             'bundled sample.')
    file = st.file_uploader('Customer file', type='csv')
    source = file
    if source is None and os.path.exists('sample_customers.csv'):
        if st.checkbox('Use bundled sample_customers.csv (50 customers)'):
            source = 'sample_customers.csv'

    if source is not None:
        data = pd.read_csv(source)
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f'Missing columns: {missing}')
        else:
            for c in ['TotalCharges', 'MonthlyCharges', 'tenure',
                      'SeniorCitizen']:
                data[c] = pd.to_numeric(data[c], errors='coerce').fillna(0)
            data['p_churn'] = score(data).round(3)
            data['band'] = [band_of(v, threshold) for v in data['p_churn']]
            data['contact'] = np.where(data['p_churn'] >= threshold,
                                       'Yes', '')
            flagged = data[data['contact'] == 'Yes']

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric('Customers scored', f'{len(data):,}')
            b2.metric('Above threshold',
                      f'{len(flagged):,} ({len(flagged) / len(data):.0%})')
            b3.metric('Expected churners', f'{data["p_churn"].sum():.1f}')
            b4.metric('Monthly revenue flagged (USD)',
                      f'{flagged["MonthlyCharges"].sum():,.0f}')
            b5.metric('Net value of calling flagged',
                      pkr((flagged['p_churn'] * SAVE * LOSS - CALL).sum()))

            h1, h2, h3 = st.columns(3)
            hist = px.histogram(data, x='p_churn', nbins=20,
                                title='Risk score distribution')
            hist.add_vline(x=threshold, line_dash='dash', line_color='gray')
            hist.update_xaxes(range=[0, 1], title='P(churn)')
            h1.plotly_chart(style(hist), use_container_width=True)

            counts = (data['band'].value_counts()
                      .reindex(['HIGH', 'WATCH', 'LOW'], fill_value=0))
            bar = px.bar(x=counts.index, y=counts.values, color=counts.index,
                         color_discrete_map=COLORS, title='Customers by band')
            bar.update_layout(showlegend=False, xaxis_title='',
                              yaxis_title='Customers')
            h2.plotly_chart(style(bar), use_container_width=True)

            seg = data.groupby('Contract', as_index=False)['p_churn'].mean()
            segbar = px.bar(seg, x='Contract', y='p_churn',
                            title='Average risk by contract')
            segbar.update_yaxes(tickformat='.0%', title='Mean P(churn)')
            h3.plotly_chart(style(segbar), use_container_width=True)

            st.subheader('Highest-risk customers')
            show = ([c for c in ['customerID'] if c in data.columns] +
                    ['Contract', 'tenure', 'MonthlyCharges', 'p_churn',
                     'band', 'contact'])
            st.dataframe(data.sort_values('p_churn', ascending=False)
                         [show].head(50), hide_index=True,
                         use_container_width=True)
            st.download_button('Download scored CSV',
                               data.to_csv(index=False),
                               'scored_customers.csv', 'text/csv')

# ======================= Tab 3: model & about =======================
with tab3:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Model', meta['model_name'])
    m2.metric('CV AUC', f"{meta['cv_auc']:.3f} +/- {meta['cv_auc_std']:.3f}")
    m3.metric('Test AUC', f"{meta.get('test_auc', float('nan')):.3f}")
    m4.metric('Default threshold', f"{meta['threshold']:.2f}")

    a1, a2 = st.columns(2)
    with a1:
        st.subheader('How to use it')
        st.markdown(
            '- Rank customers by churn risk so a retention team can '
            'prioritise calls.\n'
            '- Use the what-if table to choose a retention offer to test.\n'
            '- Use batch scoring for a whole customer file.\n\n'
            '**Not for:** credit, pricing, or any decision that denies a '
            'service.')
    with a2:
        st.subheader('Limitations')
        st.markdown(
            '- Trained on one US telecom dataset from one period; not '
            'validated for other markets.\n'
            '- What-if results are associations, not causal effects. Test '
            'offers with a controlled experiment.\n'
            '- Performance may drift as plans and prices change.\n'
            '- Cost figures are editable assumptions, not measurements.')

    st.subheader('Model metadata')
    st.json({k: v for k, v in meta.items() if k != 'feature_columns'})
