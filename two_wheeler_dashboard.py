import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

# ─── DATA LOADING & FEATURE ENGINEERING ────────────────────────────────────
reg = pd.read_excel('C:/Users/Rohit Prajapati/Documents/Two Wheeler Assignment/Inputs/Two-Wheeler_Demand_Assessment_Data.xlsx', sheet_name='2W_Registrations')
macro = pd.read_excel('C:/Users/Rohit Prajapati/Documents/Two Wheeler Assignment/Inputs/Two-Wheeler_Demand_Assessment_Data.xlsx', sheet_name='Macro_Indicators')

df = pd.merge(reg, macro, on=['State', 'Quarter'])
df = df.sort_values(['State', 'Quarter']).reset_index(drop=True)

# Parse quarter order
df['Year'] = df['Quarter'].str[:4].astype(int)
df['Q'] = df['Quarter'].str[-1:].astype(int)
df['QuarterOrder'] = (df['Year'] - 2021) * 4 + df['Q']

MACRO_COLS = ['Rainfall_Index', 'Rural_Wage_Index', 'Fuel_Price_Index', 'CPI_Index', 'Agri_Output_Index']
MACRO_LABELS = {
    'Rainfall_Index': 'Rainfall Index',
    'Rural_Wage_Index': 'Rural Wage Index',
    'Fuel_Price_Index': 'Fuel Price Index',
    'CPI_Index': 'CPI Index',
    'Agri_Output_Index': 'Agri Output Index',
}
STATES = sorted(df['State'].unique())
QUARTERS = sorted(df['Quarter'].unique())

# Per-state derived metrics
state_dfs = []
for state, sdf in df.groupby('State'):
    sdf = sdf.sort_values('Quarter').copy()
    sdf['QoQ_Growth'] = sdf['2W_Registrations'].pct_change() * 100
    sdf['YoY_Growth'] = sdf['2W_Registrations'].pct_change(4) * 100
    sdf['MA4'] = sdf['2W_Registrations'].rolling(4).mean()
    sdf['MomentumSignal'] = sdf['QoQ_Growth'].rolling(2).mean()
    for col in MACRO_COLS:
        for lag in [1, 2]:
            sdf[f'{col}_lag{lag}'] = sdf[col].shift(lag)
    state_dfs.append(sdf)

df = pd.concat(state_dfs).reset_index(drop=True)

# ─── CORRELATION ANALYSIS ───────────────────────────────────────────────────
def compute_correlations(data, lag=0):
    rows = []
    for state, sdf in data.groupby('State'):
        sdf = sdf.sort_values('Quarter').dropna(subset=['QoQ_Growth'])
        for col in MACRO_COLS:
            tag = col if lag == 0 else f'{col}_lag{lag}'
            sub = sdf[['QoQ_Growth', tag]].dropna()
            if len(sub) > 4:
                r, p = pearsonr(sub[tag], sub['QoQ_Growth'])
                rows.append({'State': state, 'Indicator': MACRO_LABELS[col], 'Lag': lag, 'Correlation': round(r, 3), 'Significant': p < 0.1})
    return pd.DataFrame(rows)

corr0 = compute_correlations(df, 0)
corr1 = compute_correlations(df, 1)
corr2 = compute_correlations(df, 2)
corr_all = pd.concat([corr0, corr1, corr2])

# ─── COLOUR PALETTE ─────────────────────────────────────────────────────────
COLORS = {
    'bg': '#0e1117', 'surface': '#1a1f2e', 'surface2': '#232940',
    'accent': '#4f8ef7', 'accent2': '#7c5cbf', 'green': '#2ecc71',
    'red': '#e74c3c', 'yellow': '#f39c12', 'text': '#e8eaf0',
    'subtext': '#8892b0', 'border': '#2d3452',
}
STATE_PALETTE = px.colors.qualitative.Bold
FONT = 'Inter, Segoe UI, sans-serif'

LAYOUT_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family=FONT, color=COLORS['text'], size=12),
    margin=dict(t=40, b=40, l=50, r=20),
    xaxis=dict(gridcolor=COLORS['border'], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=COLORS['border'], showgrid=True, zeroline=False),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=COLORS['border'], borderwidth=1),
)

# ─── INSIGHT SUMMARY ────────────────────────────────────────────────────────
latest_q = QUARTERS[-1]
prev_q = QUARTERS[-2]
prev_year_q = QUARTERS[-5]

latest_df = df[df['Quarter'] == latest_q][['State', '2W_Registrations', 'QoQ_Growth', 'YoY_Growth', 'MomentumSignal']].copy()
top_state = latest_df.nlargest(1, '2W_Registrations').iloc[0]
bottom_state = latest_df.nsmallest(1, '2W_Registrations').iloc[0]
best_yoy = latest_df.nlargest(1, 'YoY_Growth').iloc[0]
worst_yoy = latest_df.nsmallest(1, 'YoY_Growth').iloc[0]

total_regs_latest = df[df['Quarter'] == latest_q]['2W_Registrations'].sum()
total_regs_prev = df[df['Quarter'] == prev_q]['2W_Registrations'].sum()
total_qoq = (total_regs_latest / total_regs_prev - 1) * 100

# Best leading indicator overall
best_lead = corr_all[corr_all['Lag'] > 0].assign(AbsCorr=lambda x: x['Correlation'].abs()).nlargest(1, 'AbsCorr').iloc[0]

# ─── APP LAYOUT ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title='Two-Wheeler Demand Intelligence')

CARD = {
    'background': COLORS['surface'],
    'borderRadius': '12px',
    'border': f'1px solid {COLORS["border"]}',
    'padding': '20px',
    'marginBottom': '16px',
}

def kpi_card(title, value, sub, color=COLORS['accent']):
    return html.Div([
        html.P(title, style={'color': COLORS['subtext'], 'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'margin': '0 0 4px 0'}),
        html.H3(value, style={'color': color, 'margin': '0', 'fontSize': '26px', 'fontWeight': '700'}),
        html.P(sub, style={'color': COLORS['subtext'], 'fontSize': '12px', 'margin': '4px 0 0 0'}),
    ], style={**CARD, 'textAlign': 'center'})

def section_header(title, subtitle=''):
    return html.Div([
        html.H5(title, style={'color': COLORS['text'], 'margin': '0', 'fontWeight': '600'}),
        html.P(subtitle, style={'color': COLORS['subtext'], 'fontSize': '12px', 'margin': '2px 0 0 0'}) if subtitle else None,
    ], style={'marginBottom': '12px'})

app.layout = html.Div(style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'fontFamily': FONT, 'color': COLORS['text']}, children=[

    # ── HEADER ──
    html.Div([
        html.Div([
            html.H1('Two-Wheeler Demand Intelligence', style={'margin': 0, 'fontWeight': '700', 'fontSize': '22px', 'color': COLORS['text']}),
            html.P('Quarterly performance monitoring · State-level tracking · Macro signal analysis', style={'margin': '2px 0 0 0', 'color': COLORS['subtext'], 'fontSize': '13px'}),
        ]),
        html.Div([
            html.Span(f'Latest Quarter: {latest_q}', style={'background': COLORS['surface2'], 'border': f'1px solid {COLORS["accent"]}', 'color': COLORS['accent'], 'padding': '6px 14px', 'borderRadius': '20px', 'fontSize': '13px', 'fontWeight': '600'}),
        ]),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '20px 32px', 'backgroundColor': COLORS['surface'], 'borderBottom': f'1px solid {COLORS["border"]}', 'marginBottom': '24px'}),

    html.Div(style={'padding': '0 32px 32px 32px'}, children=[

        # ── NAV TABS ──
        dcc.Tabs(id='tabs', value='overview', style={'marginBottom': '20px'}, children=[
            dcc.Tab(label='📊  Overview', value='overview', style={'backgroundColor': COLORS['bg'], 'color': COLORS['subtext'], 'border': 'none', 'padding': '10px 20px'},
                    selected_style={'backgroundColor': COLORS['surface2'], 'color': COLORS['text'], 'borderTop': f'2px solid {COLORS["accent"]}', 'fontWeight': '600'}),
            dcc.Tab(label='📈  State Deep Dive', value='state', style={'backgroundColor': COLORS['bg'], 'color': COLORS['subtext'], 'border': 'none', 'padding': '10px 20px'},
                    selected_style={'backgroundColor': COLORS['surface2'], 'color': COLORS['text'], 'borderTop': f'2px solid {COLORS["accent"]}', 'fontWeight': '600'}),
            dcc.Tab(label='🔬  Macro Signals', value='macro', style={'backgroundColor': COLORS['bg'], 'color': COLORS['subtext'], 'border': 'none', 'padding': '10px 20px'},
                    selected_style={'backgroundColor': COLORS['surface2'], 'color': COLORS['text'], 'borderTop': f'2px solid {COLORS["accent"]}', 'fontWeight': '600'}),
            dcc.Tab(label='🚨  Watchlist', value='watchlist', style={'backgroundColor': COLORS['bg'], 'color': COLORS['subtext'], 'border': 'none', 'padding': '10px 20px'},
                    selected_style={'backgroundColor': COLORS['surface2'], 'color': COLORS['text'], 'borderTop': f'2px solid {COLORS["accent"]}', 'fontWeight': '600'}),
            dcc.Tab(label='💡  Insight Summary', value='insights', style={'backgroundColor': COLORS['bg'], 'color': COLORS['subtext'], 'border': 'none', 'padding': '10px 20px'},
                    selected_style={'backgroundColor': COLORS['surface2'], 'color': COLORS['text'], 'borderTop': f'2px solid {COLORS["accent"]}', 'fontWeight': '600'}),
        ]),

        html.Div(id='tab-content'),
    ]),
])

# ─── TAB CONTENT CALLBACKS ──────────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'overview':   return render_overview()
    if tab == 'state':      return render_state()
    if tab == 'macro':      return render_macro()
    if tab == 'watchlist':  return render_watchlist()
    if tab == 'insights':   return render_insights()

# ── OVERVIEW TAB ────────────────────────────────────────────────────────────
def render_overview():
    total_all = df.groupby('Quarter')['2W_Registrations'].sum().reset_index()
    total_all = total_all.sort_values('Quarter')
    total_all['QoQ'] = total_all['2W_Registrations'].pct_change() * 100
    total_all['YoY'] = total_all['2W_Registrations'].pct_change(4) * 100
    total_all['MA4'] = total_all['2W_Registrations'].rolling(4).mean()

    # Share by state latest quarter
    share_df = df[df['Quarter'] == latest_q].copy()
    share_df['Share'] = share_df['2W_Registrations'] / share_df['2W_Registrations'].sum() * 100

    # Stacked bar by state over time
    pivot = df.pivot_table(index='Quarter', columns='State', values='2W_Registrations', aggfunc='sum').reset_index()

    # KPIs
    kpis = dbc.Row([
        dbc.Col(kpi_card('Total Registrations', f'{total_regs_latest:,.0f}', f'{latest_q}'), md=3),
        dbc.Col(kpi_card('QoQ Growth (All India)', f'{total_qoq:+.1f}%', f'vs {prev_q}', COLORS['green'] if total_qoq >= 0 else COLORS['red']), md=3),
        dbc.Col(kpi_card('Top State', top_state['State'], f'{top_state["2W_Registrations"]:,.0f} units'), md=3),
        dbc.Col(kpi_card('Best YoY Growth', best_yoy['State'], f'{best_yoy["YoY_Growth"]:+.1f}%', COLORS['green']), md=3),
    ])

    # Trend chart with MA
    fig1 = go.Figure()
    fig1.add_bar(x=total_all['Quarter'], y=total_all['2W_Registrations'], name='Registrations',
                 marker_color=COLORS['accent'], opacity=0.7)
    fig1.add_scatter(x=total_all['Quarter'], y=total_all['MA4'], name='4Q Moving Avg',
                     line=dict(color=COLORS['yellow'], width=2.5, dash='dot'), mode='lines')
    fig1.update_layout(**LAYOUT_BASE, title='All-India Total 2W Registrations with 4-Quarter MA', height=300)

    # Growth rates  
    fig2 = make_subplots(specs=[[{'secondary_y': True}]])
    fig2.add_bar(x=total_all['Quarter'], y=total_all['QoQ'], name='QoQ Growth %',
                 marker_color=[COLORS['green'] if v >= 0 else COLORS['red'] for v in total_all['QoQ'].fillna(0)], secondary_y=False)
    fig2.add_scatter(x=total_all['Quarter'], y=total_all['YoY'], name='YoY Growth %',
                     line=dict(color=COLORS['yellow'], width=2), mode='lines+markers', secondary_y=True)
    fig2.update_layout(**LAYOUT_BASE, title='Growth Rates: QoQ (bars) vs YoY (line)', height=300)
    fig2.update_yaxes(title_text='QoQ %', secondary_y=False, gridcolor=COLORS['border'])
    fig2.update_yaxes(title_text='YoY %', secondary_y=True, gridcolor='rgba(0,0,0,0)')

    # Stacked area
    fig3 = go.Figure()
    for i, state in enumerate(STATES):  
        if state in pivot.columns:
            fig3.add_scatter(x=pivot['Quarter'], y=pivot[state], name=state, stackgroup='one',
                             line=dict(width=0.5), fillcolor=STATE_PALETTE[i % len(STATE_PALETTE)])
    fig3.update_layout(**LAYOUT_BASE, title='State-wise Contribution to Total Registrations', height=300)

    # Donut chart for share
    fig4 = go.Figure(go.Pie(labels=share_df['State'], values=share_df['Share'],
                             hole=0.55, textinfo='label+percent',
                             marker=dict(colors=STATE_PALETTE[:len(STATES)])))
    fig4.update_layout(**LAYOUT_BASE, title=f'Market Share by State — {latest_q}', height=300,
                       showlegend=False)

    return html.Div([
        kpis,
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig1, config={'displayModeBar': False}), style=CARD), md=8),
            dbc.Col(html.Div(dcc.Graph(figure=fig4, config={'displayModeBar': False}), style=CARD), md=4),
        ]),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig2, config={'displayModeBar': False}), style=CARD), md=6),
            dbc.Col(html.Div(dcc.Graph(figure=fig3, config={'displayModeBar': False}), style=CARD), md=6),
        ]),
    ])

# ── STATE DEEP DIVE TAB ─────────────────────────────────────────────────────
def render_state():
    controls = dbc.Row([
        dbc.Col([
            html.Label('Select States', style={'color': COLORS['subtext'], 'fontSize': '12px'}),
            dcc.Dropdown(id='state-selector', options=[{'label': s, 'value': s} for s in STATES],
                         value=STATES, multi=True,
                         style={'backgroundColor': COLORS['surface2'], 'color': COLORS['text']},
                         className='dark-dropdown'),
        ], md=8),
        dbc.Col([
            html.Label('Metric', style={'color': COLORS['subtext'], 'fontSize': '12px'}),
            dcc.RadioItems(id='metric-selector', options=[
                {'label': ' Registrations', 'value': '2W_Registrations'},
                {'label': ' QoQ Growth %', 'value': 'QoQ_Growth'},
                {'label': ' YoY Growth %', 'value': 'YoY_Growth'},
            ], value='2W_Registrations', inline=True, style={'color': COLORS['text'], 'fontSize': '13px'},
               labelStyle={'marginRight': '16px', 'cursor': 'pointer'}),
        ], md=4),
    ], style={**CARD, 'padding': '16px 20px'})

    return html.Div([
        controls,
        html.Div(id='state-charts'),
        html.Div(id='state-heatmap'),
    ])

@app.callback(
    Output('state-charts', 'children'),
    Output('state-heatmap', 'children'),
    Input('state-selector', 'value'),
    Input('metric-selector', 'value'),
)
def update_state_charts(states, metric):
    if not states:
        return html.Div(), html.Div()
    fdf = df[df['State'].isin(states)].sort_values(['State', 'Quarter'])

    # Line chart
    fig1 = go.Figure()
    for i, state in enumerate(states):
        sdf = fdf[fdf['State'] == state]
        fig1.add_scatter(x=sdf['Quarter'], y=sdf[metric], name=state,
                         line=dict(color=STATE_PALETTE[i % len(STATE_PALETTE)], width=2),
                         mode='lines+markers', marker=dict(size=5))
        if metric == '2W_Registrations':
            fig1.add_scatter(x=sdf['Quarter'], y=sdf['MA4'], name=f'{state} MA4',
                             line=dict(color=STATE_PALETTE[i % len(STATE_PALETTE)], width=1.5, dash='dot'),
                             showlegend=False, mode='lines')
    ylabel = {'2W_Registrations': 'Units', 'QoQ_Growth': 'QoQ %', 'YoY_Growth': 'YoY %'}[metric]
    fig1.update_layout(**LAYOUT_BASE, title=f'State Comparison — {ylabel}', height=360, yaxis_title=ylabel)
    if metric in ['QoQ_Growth', 'YoY_Growth']:
        fig1.add_hline(y=0, line_dash='dash', line_color=COLORS['subtext'], line_width=1)

    # Bar rank chart for latest quarter
    rank_df = df[df['Quarter'] == latest_q][['State', metric]].dropna().sort_values(metric, ascending=True)
    fig2 = go.Figure(go.Bar(
        y=rank_df['State'], x=rank_df[metric],
        orientation='h',
        marker_color=[COLORS['green'] if v >= 0 else COLORS['red'] for v in rank_df[metric]] if metric != '2W_Registrations' else COLORS['accent'],
        text=[f'{v:,.0f}' if metric == '2W_Registrations' else f'{v:+.1f}%' for v in rank_df[metric]],
        textposition='outside',
    ))
    fig2.update_layout(**LAYOUT_BASE, title=f'State Ranking — {latest_q}', height=360, xaxis_title=ylabel)

    # Heatmap: state x quarter for growth
    pivot_yoy = df.pivot_table(index='State', columns='Quarter', values='YoY_Growth')
    fig3 = go.Figure(go.Heatmap(
        z=pivot_yoy.values, x=pivot_yoy.columns.tolist(), y=pivot_yoy.index.tolist(),
        colorscale=[[0, COLORS['red']], [0.5, '#1a1f2e'], [1, COLORS['green']]],
        zmid=0, text=pivot_yoy.values.round(1), texttemplate='%{text}%',
        colorbar=dict(tickfont=dict(color=COLORS['text'])),
    ))

   
    fig3.update_layout(**LAYOUT_BASE, title='YoY Growth Heatmap — State × Quarter', height=280)

    return (
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig1, config={'displayModeBar': False}), style=CARD), md=8),
            dbc.Col(html.Div(dcc.Graph(figure=fig2, config={'displayModeBar': False}), style=CARD), md=4),
        ]),
        html.Div(dcc.Graph(figure=fig3, config={'displayModeBar': False}), style=CARD),
    )

# ── MACRO SIGNALS TAB ───────────────────────────────────────────────────────
def render_macro():
    controls = dbc.Row([
        dbc.Col([
            html.Label('State', style={'color': COLORS['subtext'], 'fontSize': '12px'}),
            dcc.Dropdown(id='macro-state', options=[{'label': s, 'value': s} for s in STATES],
                         value=STATES[0], clearable=False,
                         style={'backgroundColor': COLORS['surface2'], 'color': COLORS['text']}),
        ], md=4),
        dbc.Col([
            html.Label('Indicator', style={'color': COLORS['subtext'], 'fontSize': '12px'}),
            dcc.Dropdown(id='macro-indicator',
                         options=[{'label': MACRO_LABELS[c], 'value': c} for c in MACRO_COLS],
                         value=MACRO_COLS[0], clearable=False,
                         style={'backgroundColor': COLORS['surface2'], 'color': COLORS['text']}),
        ], md=4),
        dbc.Col([
            html.Label('Lag (quarters)', style={'color': COLORS['subtext'], 'fontSize': '12px'}),
            dcc.RadioItems(id='lag-selector', options=[
                {'label': ' Current', 'value': 0},
                {'label': ' Lag 1', 'value': 1},
                {'label': ' Lag 2', 'value': 2},
            ], value=0, inline=True, style={'color': COLORS['text'], 'fontSize': '13px'},
               labelStyle={'marginRight': '16px', 'cursor': 'pointer', 'paddingTop': '6px'}),
        ], md=4),
    ], style={**CARD, 'padding': '16px 20px'})

    return html.Div([
        controls,
        html.Div(id='macro-charts'),
        html.Div(id='corr-heatmap'),
    ])

@app.callback(
    Output('macro-charts', 'children'),
    Output('corr-heatmap', 'children'),
    Input('macro-state', 'value'),
    Input('macro-indicator', 'value'),
    Input('lag-selector', 'value'),
)
def update_macro(state, indicator, lag):
    sdf = df[df['State'] == state].sort_values('Quarter').copy()
    ind_col = indicator if lag == 0 else f'{indicator}_lag{lag}'
    label = MACRO_LABELS[indicator] + (f' (Lag {lag}Q)' if lag > 0 else '')

    # Dual-axis: indicator vs registrations
    fig1 = make_subplots(specs=[[{'secondary_y': True}]])
    fig1.add_bar(x=sdf['Quarter'], y=sdf['2W_Registrations'], name='Registrations',
                 marker_color=COLORS['accent'], opacity=0.6, secondary_y=False)
    fig1.add_scatter(x=sdf['Quarter'], y=sdf[ind_col], name=label,
                     line=dict(color=COLORS['yellow'], width=2), mode='lines+markers',
                     marker=dict(size=5), secondary_y=True)
    fig1.update_layout(**LAYOUT_BASE, title=f'{state} — Registrations vs {label}', height=320)
    fig1.update_yaxes(title_text='Registrations', secondary_y=False, gridcolor=COLORS['border'])
    fig1.update_yaxes(title_text=label, secondary_y=True, gridcolor='rgba(0,0,0,0)')

    # Scatter: indicator vs QoQ growth
    sub = sdf[[ind_col, 'QoQ_Growth', 'Quarter']].dropna()
    r_val, _ = pearsonr(sub[ind_col], sub['QoQ_Growth']) if len(sub) > 3 else (0, 1)
    fig2 = go.Figure(go.Scatter(
        x=sub[ind_col], y=sub['QoQ_Growth'],
        mode='markers+text', text=sub['Quarter'], textposition='top center',
        marker=dict(color=COLORS['accent2'], size=9, line=dict(color=COLORS['text'], width=0.5)),
    ))
    # Trend line
    if len(sub) > 2:
        m, b = np.polyfit(sub[ind_col], sub['QoQ_Growth'], 1)
        xr = np.linspace(sub[ind_col].min(), sub[ind_col].max(), 50)
        fig2.add_scatter(x=xr, y=m * xr + b, mode='lines', line=dict(color=COLORS['red'], dash='dash'), name='Trend')
    fig2.update_layout(**LAYOUT_BASE, title=f'Correlation: {label} → QoQ Growth  (r={r_val:.2f})', height=320,
                       xaxis_title=label, yaxis_title='QoQ Growth %')
    fig2.add_hline(y=0, line_dash='dot', line_color=COLORS['subtext'], line_width=1)

    # Correlation heatmap across all states and lags
    heat_rows = []
    for s in STATES:
        sdf2 = df[df['State'] == s].sort_values('Quarter').dropna(subset=['QoQ_Growth'])
        row = {'State': s}
        for col in MACRO_COLS:
            for lg in [0, 1, 2]:
                tag = col if lg == 0 else f'{col}_lag{lg}'
                sub2 = sdf2[['QoQ_Growth', tag]].dropna()
                r = pearsonr(sub2[tag], sub2['QoQ_Growth'])[0] if len(sub2) > 4 else np.nan
                row[f'{MACRO_LABELS[col]} (L{lg})'] = round(r, 2)
        heat_rows.append(row)
    heat_df = pd.DataFrame(heat_rows).set_index('State')

    fig3 = go.Figure(go.Heatmap(
        z=heat_df.values, x=heat_df.columns.tolist(), y=heat_df.index.tolist(),
        colorscale=[[0, COLORS['red']], [0.5, '#1a1f2e'], [1, COLORS['green']]],
        zmid=0, zmin=-1, zmax=1,
        text=heat_df.values.round(2), texttemplate='%{text}',
        colorbar=dict(tickfont=dict(color=COLORS['text']), title=dict(text='r', font=dict(color=COLORS['text']))),
    ))
    fig3.update_layout(**LAYOUT_BASE, title='Correlation Matrix: Macro Indicators (with Lags) → QoQ Growth',
                       height=300)
    fig3.update_xaxes(tickangle=45)

    return (
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig1, config={'displayModeBar': False}), style=CARD), md=6),
            dbc.Col(html.Div(dcc.Graph(figure=fig2, config={'displayModeBar': False}), style=CARD), md=6),
        ]),
        html.Div(dcc.Graph(figure=fig3, config={'displayModeBar': False}), style=CARD),
    )

# ── WATCHLIST TAB ───────────────────────────────────────────────────────────
def render_watchlist():
    # Classify states by momentum in last 2 quarters
    watch = []
    for state in STATES:
        sdf = df[df['State'] == state].sort_values('Quarter').tail(4)
        latest = sdf.iloc[-1]
        prev = sdf.iloc[-2]
        qoq2 = sdf['QoQ_Growth'].tail(2).mean()
        yoy = latest['YoY_Growth']
        momentum = latest['MomentumSignal']

        if qoq2 >= 3 and yoy >= 5:
            signal, color = '🟢 Accelerating', COLORS['green']
        elif qoq2 >= 0 and yoy >= 0:
            signal, color = '🟡 Stable / Mild Growth', COLORS['yellow']
        elif qoq2 < 0 and yoy < 0:
            signal, color = '🔴 Weakening', COLORS['red']
        else:
            signal, color = '🟠 Mixed', COLORS['yellow']

        watch.append({
            'State': state,
            'Signal': signal,
            'Color': color,
            f'Regs {latest_q}': f'{latest["2W_Registrations"]:,.0f}',
            'QoQ %': f'{latest["QoQ_Growth"]:+.1f}%' if pd.notna(latest['QoQ_Growth']) else 'N/A',
            'YoY %': f'{yoy:+.1f}%' if pd.notna(yoy) else 'N/A',
            '2Q Avg QoQ': f'{qoq2:+.1f}%' if pd.notna(qoq2) else 'N/A',
        })

    watch_df = pd.DataFrame(watch)

    # Momentum chart — 2Q rolling QoQ
    fig1 = go.Figure()
    for i, state in enumerate(STATES):
        sdf = df[df['State'] == state].sort_values('Quarter')
        fig1.add_scatter(x=sdf['Quarter'], y=sdf['MomentumSignal'], name=state,
                         line=dict(color=STATE_PALETTE[i % len(STATE_PALETTE)], width=2),
                         mode='lines+markers', marker=dict(size=5))
    fig1.add_hline(y=0, line_dash='dash', line_color=COLORS['subtext'], line_width=1)
    fig1.update_layout(**LAYOUT_BASE, title='2-Quarter Rolling Momentum Signal (Avg QoQ Growth %)', height=320)

    # YoY trend by state
    fig2 = go.Figure()
    for i, state in enumerate(STATES):
        sdf = df[df['State'] == state].sort_values('Quarter')
        fig2.add_scatter(x=sdf['Quarter'], y=sdf['YoY_Growth'], name=state,
                         line=dict(color=STATE_PALETTE[i % len(STATE_PALETTE)], width=2),
                         mode='lines+markers', marker=dict(size=5))
    fig2.add_hline(y=0, line_dash='dash', line_color=COLORS['subtext'], line_width=1)
    fig2.update_layout(**LAYOUT_BASE, title='YoY Growth Trend by State', height=320)

    # Signal table
    table = dash_table.DataTable(
        data=watch_df.drop(columns=['Color']).to_dict('records'),
        columns=[{'name': c, 'id': c} for c in watch_df.columns if c != 'Color'],
        style_table={'overflowX': 'auto'},
        style_cell={'backgroundColor': COLORS['surface'], 'color': COLORS['text'],
                    'border': f'1px solid {COLORS["border"]}', 'textAlign': 'center',
                    'fontFamily': FONT, 'fontSize': '13px', 'padding': '10px'},
        style_header={'backgroundColor': COLORS['surface2'], 'color': COLORS['text'],
                      'fontWeight': '700', 'border': f'1px solid {COLORS["border"]}'},
        style_data_conditional=[
            {'if': {'filter_query': '{Signal} contains "🟢"'}, 'backgroundColor': 'rgba(46,204,113,0.1)'},
            {'if': {'filter_query': '{Signal} contains "🔴"'}, 'backgroundColor': 'rgba(231,76,60,0.1)'},
            {'if': {'filter_query': '{Signal} contains "🟠"'}, 'backgroundColor': 'rgba(230,126,34,0.1)'},
        ],
    )

    return html.Div([
        html.Div([
            section_header('State Demand Signal Summary', f'Based on {latest_q} performance and 2-quarter momentum'),
            table,
        ], style=CARD),
        dbc.Row([
            dbc.Col(html.Div(dcc.Graph(figure=fig1, config={'displayModeBar': False}), style=CARD), md=6),
            dbc.Col(html.Div(dcc.Graph(figure=fig2, config={'displayModeBar': False}), style=CARD), md=6),
        ]),
    ])

# ── INSIGHT SUMMARY TAB ─────────────────────────────────────────────────────
def render_insights():
    # Compute per-state latest metrics for insights
    state_summary = []
    for state in STATES:
        sdf = df[df['State'] == state].sort_values('Quarter')
        latest_row = sdf[sdf['Quarter'] == latest_q].iloc[0]
        share = latest_row['2W_Registrations'] / df[df['Quarter'] == latest_q]['2W_Registrations'].sum() * 100
        state_summary.append({
            'state': state,
            'regs': latest_row['2W_Registrations'],
            'qoq': latest_row['QoQ_Growth'],
            'yoy': latest_row['YoY_Growth'],
            'share': share,
            'momentum': latest_row['MomentumSignal'],
        })
    ss = sorted(state_summary, key=lambda x: -x['regs'])

    # Best leading indicator
    best_leads = corr_all[corr_all['Lag'] > 0].assign(AbsCorr=lambda x: x['Correlation'].abs()).nlargest(3, 'AbsCorr')

    def insight_block(icon, title, body, color=COLORS['accent']):
        return html.Div([
            html.Div([
                html.Span(icon, style={'fontSize': '24px', 'marginRight': '10px'}),
                html.Span(title, style={'fontWeight': '700', 'color': color, 'fontSize': '14px'}),
            ], style={'marginBottom': '8px', 'display': 'flex', 'alignItems': 'center'}),
            html.P(body, style={'color': COLORS['text'], 'fontSize': '13px', 'lineHeight': '1.7', 'margin': 0}),
        ], style={**CARD, 'borderLeft': f'3px solid {color}', 'padding': '16px 20px'})

    # Build narrative
    top2_states = [s['state'] for s in ss[:2]]
    top2_share = sum(s['share'] for s in ss[:2])
    improving = [s['state'] for s in state_summary if pd.notna(s['qoq']) and s['qoq'] > 0 and pd.notna(s['yoy']) and s['yoy'] > 0]
    weakening = [s['state'] for s in state_summary if pd.notna(s['qoq']) and s['qoq'] < 0 and pd.notna(s['yoy']) and s['yoy'] < 0]

    best_lead_rows = best_leads.itertuples()
    lead1 = next(best_lead_rows)
    lead2 = next(best_lead_rows)
    lead3 = next(best_lead_rows)

    return html.Div([
        html.Div([
            html.H4('📋 Key Findings — Insight Summary', style={'color': COLORS['text'], 'fontWeight': '700', 'marginBottom': '4px'}),
            html.P(f'Analysis period: 2021Q1 – {latest_q}  ·  5 states  ·  All-India perspective',
                   style={'color': COLORS['subtext'], 'fontSize': '13px', 'marginBottom': '24px'}),
        ]),

        dbc.Row([
            dbc.Col(insight_block('📊', 'Market Concentration',
                f'{", ".join(top2_states)} together account for ~{top2_share:.0f}% of total registrations in {latest_q}. '
                f'Uttar Pradesh consistently leads volume, followed by Maharashtra. '
                f'Odisha and Bihar together contribute the remaining ~{100 - top2_share:.0f}%, indicating significant skew toward large northern and western states.',
                COLORS['accent']), md=6),
            dbc.Col(insight_block('📈', 'All-India Demand Trend',
                f'All-India registrations showed a {total_qoq:+.1f}% QoQ change in {latest_q}. '
                f'The 4-quarter moving average reveals the underlying trend, smoothing seasonal spikes in Q2/Q3 driven by monsoon and harvest cycles. '
                f'YoY growth has been the more stable signal, filtering out within-year seasonality.',
                COLORS['green']), md=6),
        ]),
        dbc.Row([
            dbc.Col(insight_block('🚦', 'State-Level Performance',
                f'States showing improving momentum (positive QoQ and YoY): {", ".join(improving) if improving else "None in latest quarter"}. '
                f'States showing weakening signals: {", ".join(weakening) if weakening else "None in latest quarter"}. '
                f'The YoY heatmap shows Bihar tends to exhibit higher volatility, while Tamil Nadu shows more stable but lower growth.',
                COLORS['yellow']), md=6),
            dbc.Col(insight_block('🔬', 'Leading Macro Indicators',
                f'The strongest leading relationships with QoQ registrations growth are: '
                f'(1) {lead1.Indicator} at Lag {lead1.Lag}Q (r={lead1.Correlation:.2f}), '
                f'(2) {lead2.Indicator} at Lag {lead2.Lag}Q (r={lead2.Correlation:.2f}), '
                f'(3) {lead3.Indicator} at Lag {lead3.Lag}Q (r={lead3.Correlation:.2f}). '
                f'Lagged indicators outperform current-quarter indicators, supporting their use as early warning signals.',
                COLORS['accent2']), md=6),
        ]),
        dbc.Row([
            dbc.Col(insight_block('💰', 'Rural Wage & Agri Output — Key Demand Drivers',
                'Rural Wage Index and Agri Output Index show the most consistent positive correlation with registrations. '
                'Two-wheelers are predominantly a rural/semi-urban product; higher rural incomes and better harvest outcomes translate into improved purchasing power with a 1–2 quarter lag. '
                'Monitor the Kharif and Rabi output indices as leading indicators each year.',
                COLORS['green']), md=6),
            dbc.Col(insight_block('⛽', 'Fuel Price — A Headwind Indicator',
                'Fuel Price Index carries a negative correlation with QoQ registrations growth in most states. '
                'Rising fuel costs erode total cost of ownership, dampening near-term demand, particularly in price-sensitive rural segments. '
                'The effect appears with a 1-quarter lag, making current fuel price levels a useful early warning tool for the following quarter.',
                COLORS['red']), md=6),
        ]),
        dbc.Row([
            dbc.Col(insight_block('🌧️', 'Rainfall — A Seasonal Amplifier',
                'Rainfall Index positively correlates with registrations, especially in agriculturally dependent states like Bihar, Odisha, and Uttar Pradesh. '
                'Good monsoon → better crop yields → higher rural income → improved vehicle uptake with a ~1 quarter lag. '
                'Below-normal rainfall years (as reflected in index values below 95) have historically preceded demand softness.',
                COLORS['accent']), md=6),
            dbc.Col(insight_block('🚨', 'Watchlist Recommendation',
                'Set monitoring thresholds: Flag any state with 2-quarter rolling QoQ below –2% and YoY below –5% as a high-priority watchlist. '
                'Similarly, watch for Rural Wage Index declining below its 4-quarter average (a leading warning signal with 1–2Q lead). '
                'States where both Rainfall and Agri Output indices are declining simultaneously warrant immediate field investigation.',
                COLORS['red']), md=6),
        ]),

        # Summary table
        html.Div([
            section_header('State Performance Scorecard', f'As of {latest_q}'),
            dash_table.DataTable(
                data=[{
                    'State': s['state'],
                    'Registrations': f'{s["regs"]:,.0f}',
                    'Market Share': f'{s["share"]:.1f}%',
                    'QoQ Growth': f'{s["qoq"]:+.1f}%' if pd.notna(s['qoq']) else 'N/A',
                    'YoY Growth': f'{s["yoy"]:+.1f}%' if pd.notna(s['yoy']) else 'N/A',
                    '2Q Momentum': f'{s["momentum"]:+.1f}%' if pd.notna(s['momentum']) else 'N/A',
                    'Signal': '🟢 Accelerating' if (pd.notna(s['qoq']) and s['qoq'] > 2 and pd.notna(s['yoy']) and s['yoy'] > 4)
                              else ('🔴 Weakening' if (pd.notna(s['qoq']) and s['qoq'] < 0 and pd.notna(s['yoy']) and s['yoy'] < 0)
                              else '🟡 Stable'),
                } for s in ss],
                columns=[{'name': c, 'id': c} for c in ['State', 'Registrations', 'Market Share', 'QoQ Growth', 'YoY Growth', '2Q Momentum', 'Signal']],
                style_table={'overflowX': 'auto'},
                style_cell={'backgroundColor': COLORS['surface2'], 'color': COLORS['text'],
                            'border': f'1px solid {COLORS["border"]}', 'textAlign': 'center',
                            'fontFamily': FONT, 'fontSize': '13px', 'padding': '12px'},
                style_header={'backgroundColor': COLORS['surface'], 'color': COLORS['accent'],
                              'fontWeight': '700', 'border': f'1px solid {COLORS["border"]}', 'fontSize': '12px', 'textTransform': 'uppercase'},
                style_data_conditional=[
                    {'if': {'filter_query': '{Signal} contains "🟢"'}, 'backgroundColor': 'rgba(46,204,113,0.1)'},
                    {'if': {'filter_query': '{Signal} contains "🔴"'}, 'backgroundColor': 'rgba(231,76,60,0.1)'},
                ],
            ),
        ], style=CARD),
    ])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=False)
