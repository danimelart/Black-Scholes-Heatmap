# app.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
import pandas as pd

st.set_page_config(page_title="Black–Scholes Heatmap", layout="wide")

# --- Black-Scholes functions (vectorized) ---
def bs_price_grid(S_grid, sigma_grid, K, T, r, option_type='call'):
    """
    S_grid, sigma_grid are numpy arrays (meshgrid). Returns price grid same shape.
    """
    # avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma_grid, eps)
    T_safe = max(T, eps)

    d1 = (np.log(S_grid / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    d2 = d1 - sigma * np.sqrt(T_safe)

    if option_type == 'call':
        price = S_grid * norm.cdf(d1) - K * np.exp(-r * T_safe) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S_grid * norm.cdf(-d1)
    return price

def bs_greeks_grid(S_grid, sigma_grid, K, T, r):
    eps = 1e-12
    sigma = np.maximum(sigma_grid, eps)
    T_safe = max(T, eps)
    d1 = (np.log(S_grid / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * np.sqrt(T_safe))
    # Greeks
    delta_call = norm.cdf(d1)
    vega = S_grid * norm.pdf(d1) * np.sqrt(T_safe)
    gamma = norm.pdf(d1) / (S_grid * sigma * np.sqrt(T_safe))
    return {'Delta (call)': delta_call, 'Vega': vega, 'Gamma': gamma}


linkedin_url = "https://www.linkedin.com/in/daniel-melkonyan-434522300/"  # Replace with your LinkedIn URL
linkedin_icon_url = "https://cdn-icons-png.flaticon.com/24/174/174857.png"

st.sidebar.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom: 15px;">
        <span>Created by -</span>
        <a href="{linkedin_url}" target="_blank">
            <img src="{linkedin_icon_url}" alt="LinkedIn" style="width:24px; height:24px;">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


# --- Sidebar inputs ---
st.sidebar.header("Parameters")
option_type = st.sidebar.radio("Option type", ('call', 'put'))
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
T = st.sidebar.slider("Time to expiration (years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-free rate (annual, decimal)", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
S_min = st.sidebar.number_input("Stock price min (S)", value=10.0)
S_max = st.sidebar.number_input("Stock price max (S)", value=200.0)
sigma_min = st.sidebar.number_input("Volatility min (σ)", value=0.05, step=0.01, format="%.2f")
sigma_max = st.sidebar.number_input("Volatility max (σ)", value=1.0, step=0.05, format="%.2f")
n_S = st.sidebar.slider("Stock grid points", 20, 200, 80)
n_sigma = st.sidebar.slider("Volatility grid points", 20, 200, 80)

show_greeks = st.sidebar.checkbox("Show Greeks heatmap (Delta / Vega / Gamma)", value=False)
download_csv = st.sidebar.checkbox("Enable CSV download", value=True)

# --- Main layout ---
st.title("Interactive Black–Scholes Heatmap")
st.markdown("""
This app computes European option prices using the Black–Scholes formula and plots an interactive heatmap.
- **X axis:** Volatility (σ)
- **Y axis:** Stock price (S)
- **Color:** Option price (or selected Greek)
""")

# create grid
S_vals = np.linspace(S_min, S_max, n_S)
sigma_vals = np.linspace(sigma_min, sigma_max, n_sigma)
S_grid, sigma_grid = np.meshgrid(S_vals, sigma_vals, indexing='xy')  # shape (n_sigma, n_S)

# compute
Z = bs_price_grid(S_grid.T, sigma_grid.T, K, T, r, option_type=option_type)  # transpose to match axes
# Z shape should be (n_S, n_sigma) where rows map to S, cols to sigma

# Plot price heatmap
fig_price = go.Figure(data=go.Heatmap(
    z=Z,
    x=sigma_vals,       # sigma on x
    y=S_vals,           # S on y
    colorscale='Viridis',
    colorbar=dict(title=f"{option_type.title()} Price")
))

fig_price.update_layout(
    title=f"Black–Scholes {option_type.title()} Price (K={K}, T={T}y, r={r})",
    xaxis_title="Volatility (σ)",
    yaxis_title="Stock Price (S)",
    autosize=True,
    height=700
)

st.plotly_chart(fig_price, use_container_width=True)

# show Greeks if asked
if show_greeks:
    greeks = bs_greeks_grid(S_grid.T, sigma_grid.T, K, T, r)
    cols = st.columns(3)
    for col, (name, grid) in zip(cols, greeks.items()):
        fig = go.Figure(data=go.Heatmap(
            z=grid,
            x=sigma_vals,
            y=S_vals,
            colorscale='Viridis',
            colorbar=dict(title=name)
        ))
        fig.update_layout(title=f"{name} (K={K}, T={T}, r={r})", xaxis_title="σ", yaxis_title="S", height=400)
        col.plotly_chart(fig, use_container_width=True)

# Data download
if download_csv:
    # Convert to DataFrame for download: flatten
    df = pd.DataFrame(Z, index=S_vals, columns=sigma_vals)
    df.index.name = "S"
    csv = df.to_csv()
    st.download_button("Download price grid as CSV", data=csv, file_name="bs_price_grid.csv", mime="text/csv")

# Short explanation and caveats
with st.expander("About / Assumptions"):
    st.write("""
    Black–Scholes assumptions include: continuous trading, log-normal price process, constant volatility (σ), constant risk-free rate (r), frictionless markets, no dividends and European exercise. This is a simple demonstration — in practice volatility is stochastic and there are other pricing models.
    """)

st.markdown("---")
st.caption("Built with Streamlit • Additions: Greeks, CSV export, adjustable grid.")
