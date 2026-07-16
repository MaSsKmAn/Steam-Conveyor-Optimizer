import streamlit as st
import numpy as np
from scipy.optimize import minimize, brentq, minimize_scalar, differential_evolution
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. FIXED CONSTANTS & FACTORY DATA
# ==============================================================================
R = 8.314  # Universal Gas Constant (J/mol K)

BASE_PARAMS = {
    "Lipase": {"Eab": 45945.38, "beta": 0.801218, "dH": 117420.10, "dS": 365.4445},
    "Peroxidase": {"Eab": 32662.39, "beta": 0.650912, "dH": 287284.26, "dS": 841.7062},
}

ENZ_TRAIN_DATA = [
    {"temp": 25.00, "moist": 9.40,  "Lipase": 0,      "Peroxidase": 0},
    {"temp": 27.23, "moist": 9.42,  "Lipase": -2.4,  "Peroxidase": -0.34},
    {"temp": 39.27, "moist": 9.63,  "Lipase": 5.05,  "Peroxidase": -0.13},
    {"temp": 52.48, "moist": 9.82,  "Lipase": 7.72,  "Peroxidase": 4.61},
    {"temp": 63.91, "moist": 9.89,  "Lipase": 23.15, "Peroxidase": 36.58},
    {"temp": 73.88, "moist": 10.10, "Lipase": 33.06, "Peroxidase": 57.59},
    {"temp": 77.25, "moist": 10.62, "Lipase": 35.22, "Peroxidase": 48.31},
    {"temp": 76.00, "moist": 10.30, "Lipase": 59.44, "Peroxidase": 93.98},
    {"temp": 82.32, "moist": 11.14, "Lipase": 74.32, "Peroxidase": 98.89},
    {"temp": 84.07, "moist": 11.25, "Lipase": 73.56, "Peroxidase": 96.13},
    {"temp": 83.73, "moist": 10.92, "Lipase": 84.56, "Peroxidase": 97.64}
]

# Updated 12-point dataset containing the lower temperature control benchmarks
GLUT_TEST_TEMPS = np.array([68.0, 68.0, 62.0, 68.0, 65.0, 72.0, 69.0, 66.0, 59.0, 70.0, 38.0, 25.0])
GLUT_TEST_GLUTENS = np.array([8.66, 4.50, 8.65, 7.86, 7.48, 3.35, 7.27, 8.77, 7.46, 8.30, 9.37, 9.08])

# ==============================================================================
# 2. CALIBRATION ENGINES (Cached for performance)
# ==============================================================================
@st.cache_data
def calibrate_enzymes():
    def mse_enz(params, enzyme):
        ln_A0, gamma = params
        A0, base = np.exp(ln_A0), BASE_PARAMS[enzyme]
        error = 0.0
        for pt in ENZ_TRAIN_DATA:
            Tk, M_pct = pt["temp"] + 273.15, pt["moist"]
            Ea_eff = max(base["Eab"] - gamma * M_pct, base["beta"] * base["Eab"])
            k = A0 * np.exp(-Ea_eff / (R * Tk))
            w = 1.0 / (1.0 + np.exp(base["dH"] / (R * Tk) - base["dS"] / R))
            pred = (1.0 - np.exp(-k * w)) * 100.0
            error += (pred - pt[enzyme]) ** 2
        return error / len(ENZ_TRAIN_DATA)

    opt_params = {}
    for enz in ["Lipase", "Peroxidase"]:
        res = minimize(mse_enz, [10.0, 50.0], args=(enz,), method='L-BFGS-B', bounds=[(1.0, 40.0), (0.1, 800.0)])
        opt_params[enz] = {"A0": np.exp(res.x[0]), "gamma": res.x[1]}
    return opt_params

@st.cache_data
def calibrate_gluten():
    # Global Differential Evolution optimizer with relaxed bounds
    def mse_glut(params):
        w_res, w_drop1, w_drop2, t1, t2, k1, k2 = params
        preds = w_res + (w_drop1 / (1.0 + np.exp(k1 * (GLUT_TEST_TEMPS - t1)))) + (w_drop2 / (1.0 + np.exp(k2 * (GLUT_TEST_TEMPS - t2))))
        return np.sqrt(np.mean((preds - GLUT_TEST_GLUTENS)**2))

    bounds = [(1.0, 4.5), (0.1, 7.0), (0.1, 7.0), (50.0, 85.0), (50.0, 85.0), (0.05, 2.0), (0.05, 2.0)]
    res = differential_evolution(mse_glut, bounds=bounds, strategy='best1bin', popsize=20, seed=42)
    return res.x

def calibrate_system(wheat_grain_kg_h, wheat_moist_pct, wheat_after_temp_pct, water_added_L_h, atta_final_moist_pct):
    wheat_moist_frac = wheat_moist_pct / 100.0
    wheat_after_temp_frac = wheat_after_temp_pct / 100.0
    atta_final_moist_frac = atta_final_moist_pct / 100.0
    
    water_initial = wheat_grain_kg_h * wheat_moist_frac
    water_after_tempering_control = (wheat_grain_kg_h + water_added_L_h) * wheat_after_temp_frac
    
    efficiency = (water_after_tempering_control - water_initial) / water_added_L_h
    moisture_loss_milling = (wheat_after_temp_frac - atta_final_moist_frac) * 100.0
    
    return efficiency, moisture_loss_milling

# ==============================================================================
# 3. PREDICTION ENGINES
# ==============================================================================
def predict_enzyme(temp, moist, enzyme, enz_params):
    base = BASE_PARAMS[enzyme]
    Tk = temp + 273.15
    Ea_eff = max(base["Eab"] - enz_params[enzyme]["gamma"] * moist, base["beta"] * base["Eab"])
    k = enz_params[enzyme]["A0"] * np.exp(-Ea_eff / (R * Tk))
    w = 1.0 / (1.0 + np.exp(base["dH"] / (R * Tk) - base["dS"] / R))
    return (1.0 - np.exp(-k * w)) * 100.0

def predict_gluten(temp, glut_params):
    w_res, w_drop1, w_drop2, t1, t2, k1, k2 = glut_params
    return w_res + (w_drop1 / (1.0 + np.exp(k1 * (temp - t1)))) + (w_drop2 / (1.0 + np.exp(k2 * (temp - t2))))

def get_optimal_window(target_lipase, target_perox, target_gluten, moisture, enz_params, glut_params):
    try: t_lip = brentq(lambda t: predict_enzyme(t, moisture, "Lipase", enz_params) - target_lipase, 10, 150)
    except ValueError: t_lip = None
    try: t_pox = brentq(lambda t: predict_enzyme(t, moisture, "Peroxidase", enz_params) - target_perox, 10, 150)
    except ValueError: t_pox = None
    try: t_glut = brentq(lambda t: predict_gluten(t, glut_params) - target_gluten, 20, 120)
    except ValueError: t_glut = None
    return t_lip, t_pox, t_glut

# ==============================================================================
# 4. THERMODYNAMIC SOLVERS
# ==============================================================================
def simulate_steamer(steam_rate, water_actual_L_h, efficiency, grain_rate_mt_h, rpm, inlet_temp, inlet_moist_pct):
    grain_kg_h = grain_rate_mt_h * 1000.0
    inlet_moist_frac = inlet_moist_pct / 100.0
    
    water_initial = grain_kg_h * inlet_moist_frac
    water_effectively_absorbed = water_actual_L_h * efficiency
    
    total_water_after_temp = water_initial + water_effectively_absorbed
    total_mass_after_temp = grain_kg_h + water_actual_L_h
    tempered_moisture_pct = (total_water_after_temp / total_mass_after_temp) * 100.0
    
    t_res = 1087.6 / rpm
    psi = steam_rate / grain_kg_h if grain_kg_h > 0 else 0
    load_factor = psi / 0.053
    T_env = 30.0 + (116.8 - 30.0) * (1.0 - np.exp(-1.2 * load_factor))
    
    T = inlet_temp
    dt = 0.1
    steps = int(t_res / dt)
    
    m_w = (tempered_moisture_pct / 100.0) / (1.0 - (tempered_moisture_pct / 100.0))
    m_w_start = m_w
    
    for _ in range(steps):
        M_pct_now = (m_w / (1.0 + m_w)) * 100.0
        Cp_dynamic = 1300.0 * (1.0 - M_pct_now/100.0) + 4184.0 * (M_pct_now/100.0)
        
        dT = T_env - T
        if dT > 0:
            q = 152.0 * 0.45 * dT 
            T += (q * dt) / Cp_dynamic
            m_w += (q * dt) / 2100000.0 * 0.05 
            
    dry_weight = grain_kg_h * (1.0 - inlet_moist_frac)
    water_added_by_steam_kg_h = (m_w - m_w_start) * dry_weight
    return T, water_added_by_steam_kg_h

def get_required_inputs_with_steam_moisture(target_temp, target_moist, moist_type, rpm, grain_rate, efficiency, moisture_loss, inlet_temp, inlet_moist_pct):
    grain_kg_h = grain_rate * 1000.0
    inlet_frac = inlet_moist_pct / 100.0
    
    if moist_type.lower() == 'atta':
        moisture_after_temp_pct = target_moist + moisture_loss
    else:
        moisture_after_temp_pct = target_moist
        
    moist_after_temp_frac = moisture_after_temp_pct / 100.0
    
    dry_weight = grain_kg_h * (1.0 - inlet_frac)
    water_before_tempering = grain_kg_h * inlet_frac
    water_after_tempering = dry_weight * moist_after_temp_frac / (1.0 - moist_after_temp_frac)
    total_water_for_tempering = water_after_tempering - water_before_tempering
    
    W_steam = 0.0
    opt_water = 0.0
    opt_steam = 0.0
    
    for _ in range(5):
        liquid_water_needed = max(0, total_water_for_tempering - W_steam)
        water_actual = liquid_water_needed / efficiency
        
        def objective(steam_guess):
            T_sim, _ = simulate_steamer(steam_guess, water_actual, efficiency, grain_rate, rpm, inlet_temp, inlet_moist_pct)
            return (T_sim - target_temp)**2
            
        res = minimize_scalar(objective, bounds=(0, 1500), method='bounded')
        opt_steam = res.x
        
        _, W_steam_new = simulate_steamer(opt_steam, water_actual, efficiency, grain_rate, rpm, inlet_temp, inlet_moist_pct)
        
        if abs(W_steam_new - W_steam) < 0.01:
            W_steam = W_steam_new
            opt_water = water_actual
            break
            
        W_steam = W_steam_new

    return opt_steam, opt_water, W_steam

def predict_outcomes_from_inputs(steam_rate_kg_h, water_added_L_h, grain_rate_mt_h, rpm, inlet_temp, inlet_moist_pct, efficiency, moisture_loss_milling, enz_params, glut_params):
    # 1. Thermodynamics
    final_temp, water_from_steam_kg_h = simulate_steamer(
        steam_rate_kg_h, water_added_L_h, efficiency, grain_rate_mt_h, rpm, inlet_temp, inlet_moist_pct
    )

    # 2. Corrected Mass Balance
    grain_flow_kg_h = grain_rate_mt_h * 1000.0
    inlet_moist_frac = inlet_moist_pct / 100.0
    
    water_initial = grain_flow_kg_h * inlet_moist_frac
    water_effectively_absorbed = water_added_L_h * efficiency
    
    total_water_after_temp = water_initial + water_effectively_absorbed + water_from_steam_kg_h
    total_mass_after_temp = grain_flow_kg_h + water_added_L_h + water_from_steam_kg_h
    
    tempered_moisture_frac = total_water_after_temp / total_mass_after_temp
    tempered_moisture_pct = tempered_moisture_frac * 100.0
    
    final_atta_moisture_pct = tempered_moisture_pct - moisture_loss_milling

    # 3. Biology
    lipase_inact = predict_enzyme(final_temp, tempered_moisture_pct, "Lipase", enz_params)
    perox_inact = predict_enzyme(final_temp, tempered_moisture_pct, "Peroxidase", enz_params)
    gluten_retention = predict_gluten(final_temp, glut_params)

    return {
        "final_temperature_c": final_temp,
        "tempered_wheat_moisture_pct": tempered_moisture_pct,
        "final_atta_moisture_pct": final_atta_moisture_pct,
        "lipase_inactivated_pct": lipase_inact,
        "peroxidase_inactivated_pct": perox_inact,
        "gluten_retained_pct": gluten_retention,
        "steam_condensed_L_h": water_from_steam_kg_h
    }

# ==============================================================================
# 5. VISUALIZATION ENGINES
# ==============================================================================
def generate_plotly_fig(moisture, t_lip, t_pox, t_glut, target_l, target_p, target_g, enz_params, glut_params):
    max_t = max([100, t_lip or 0, t_pox or 0, t_glut or 0]) + 10
    temps_array = np.linspace(20, max_t, 300)
    temps = temps_array.tolist() 
    
    lip_vals = [predict_enzyme(t, moisture, "Lipase", enz_params) for t in temps]
    pox_vals = [predict_enzyme(t, moisture, "Peroxidase", enz_params) for t in temps]
    glut_vals = [predict_gluten(t, glut_params) for t in temps]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=temps, y=lip_vals, mode='lines', name='Lipase Inact.', line=dict(color='blue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=pox_vals, mode='lines', name='Peroxidase Inact.', line=dict(color='orange', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=glut_vals, mode='lines', name='Gluten Retention', line=dict(color='purple', width=3, dash='dash')), secondary_y=True)

    min_required_temp = max(t_lip or 0, t_pox or 0) if (t_lip or t_pox) else None
    max_allowed_temp = t_glut

    if min_required_temp and max_allowed_temp:
        if min_required_temp <= max_allowed_temp:
            fig.add_vrect(x0=min_required_temp, x1=max_allowed_temp, fillcolor="green", opacity=0.15, layer="below", line_width=0, annotation_text="Valid Window")
        else:
            fig.add_vrect(x0=max_allowed_temp, x1=min_required_temp, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="Conflict Zone")

    fig.add_hline(y=target_l, line_dash="dot", line_color="blue", opacity=0.4, secondary_y=False)
    fig.add_hline(y=target_p, line_dash="dot", line_color="orange", opacity=0.4, secondary_y=False)
    fig.add_hline(y=target_g, line_dash="dot", line_color="purple", opacity=0.4, secondary_y=True)

    if t_lip: fig.add_trace(go.Scatter(x=[t_lip], y=[target_l], mode='markers', name=f'Lipase Target ({t_lip:.1f}°C)', marker=dict(color='blue', size=12, symbol='x')), secondary_y=False)
    if t_pox: fig.add_trace(go.Scatter(x=[t_pox], y=[target_p], mode='markers', name=f'Perox Target ({t_pox:.1f}°C)', marker=dict(color='orange', size=12, symbol='x')), secondary_y=False)
    if t_glut: fig.add_trace(go.Scatter(x=[t_glut], y=[target_g], mode='markers', name=f'Gluten Limit ({t_glut:.1f}°C)', marker=dict(color='purple', size=12, symbol='x')), secondary_y=True)

    fig.update_layout(
        title=f"<b>Optimization Constraints</b><br><sup>Evaluating biological constraints at {moisture}% Target Moisture</sup>",
        xaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=40),
        hovermode="x unified",
        height=550
    )
    
    fig.update_yaxes(title_text="Enzyme Inactivation (%)", range=[-5, max(105, target_l + 5, target_p + 5)], secondary_y=False)
    fig.update_yaxes(title_text="Dry Gluten Retention (%)", range=[0, max(12, target_g + 2)], secondary_y=True)
    return fig


def generate_simulation_fig(final_temp, moisture, lipase_val, perox_val, gluten_val, enz_params, glut_params):
    max_t = max(100.0, final_temp + 15.0)
    temps_array = np.linspace(20, max_t, 300)
    temps = temps_array.tolist()
    
    lip_vals = [predict_enzyme(t, moisture, "Lipase", enz_params) for t in temps]
    pox_vals = [predict_enzyme(t, moisture, "Peroxidase", enz_params) for t in temps]
    glut_vals = [predict_gluten(t, glut_params) for t in temps]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=temps, y=lip_vals, mode='lines', name='Lipase Inact.', line=dict(color='blue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=pox_vals, mode='lines', name='Peroxidase Inact.', line=dict(color='orange', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=glut_vals, mode='lines', name='Gluten Retention', line=dict(color='purple', width=3, dash='dash')), secondary_y=True)

    # Add a bold vertical line marking exactly where the simulation landed
    fig.add_vline(x=final_temp, line_width=2, line_dash="solid", line_color="red", annotation_text=f"Simulated Temp ({final_temp:.1f}°C)")

    # Add markers at the exact predicted values
    fig.add_trace(go.Scatter(x=[final_temp], y=[lipase_val], mode='markers', name=f'Lipase ({lipase_val:.1f}%)', marker=dict(color='blue', size=12, symbol='x')), secondary_y=False)
    fig.add_trace(go.Scatter(x=[final_temp], y=[perox_val], mode='markers', name=f'Perox ({perox_val:.1f}%)', marker=dict(color='orange', size=12, symbol='x')), secondary_y=False)
    fig.add_trace(go.Scatter(x=[final_temp], y=[gluten_val], mode='markers', name=f'Gluten ({gluten_val:.2f}%)', marker=dict(color='purple', size=12, symbol='x')), secondary_y=True)

    fig.update_layout(
        title=f"<b>Simulation Biological Outcomes</b><br><sup>Displaying enzyme decay and gluten retention at {moisture:.2f}% Tempered Moisture</sup>",
        xaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=40, r=40),
        hovermode="x unified",
        height=550
    )
    
    fig.update_yaxes(title_text="Enzyme Inactivation (%)", range=[-5, 105], secondary_y=False)
    fig.update_yaxes(title_text="Dry Gluten Retention (%)", range=[0, 12], secondary_y=True)
    return fig

# ==============================================================================
# 6. STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Integrated Plant Optimizer", layout="wide")

st.title("🏭 Steam Conveyor Machine Optimizer")
st.markdown("Calculate process minimums to achieve biological targets or predict what will happen with specific machine settings.")

# --- SIDEBAR: CALIBRATION ---
st.sidebar.header("⚙️ Mechanical Calibration")
st.sidebar.markdown("Baseline factory trial data used to calculate tempering efficiency and milling loss.")
cal_grain_flow = st.sidebar.number_input("Grain Flow (kg/h)", value=14000.0, step=100.0)
cal_inlet_moist = st.sidebar.number_input("Calibration Inlet Moisture (%)", value=10.04, step=0.1)
cal_water_added = st.sidebar.number_input("Calibration Water Added (L/h)", value=450.0, step=10.0)
cal_wheat_after_temp = st.sidebar.number_input("Wheat Moisture After Temp (%)", value=11.67, step=0.1)
cal_atta_final = st.sidebar.number_input("Final Atta Moisture (%)", value=9.00, step=0.1)

sys_eff, sys_loss = calibrate_system(cal_grain_flow, cal_inlet_moist, cal_wheat_after_temp, cal_water_added, cal_atta_final)

st.sidebar.markdown("---")
st.sidebar.info(f"**Tempering Efficiency:** {sys_eff*100:.2f}%\n\n**Milling Moisture Loss:** {sys_loss:.2f}%")

enz_params = calibrate_enzymes()
glut_params = calibrate_gluten()

# --- MAIN PAGE: SHARED MACHINE STATE ---
st.subheader("⚙️ Current Operating Conditions")
st.caption("These conditions apply to both the Optimization and Simulator modes below.")
colA, colB, colC, colD = st.columns(4)
curr_grain_rate = colA.number_input("Grain Rate (MT/h)", value=5.0, step=0.5)
curr_rpm = colB.number_input("Conveyor RPM", value=40.0, step=1.0)
curr_inlet_temp = colC.number_input("Inlet Grain Temp (°C)", value=38.0, step=1.0)
curr_inlet_moist = colD.number_input("Inlet Moisture (%)", value=10.04, step=0.1)

st.markdown("---")

# --- TABS FOR MODES ---
tab1, tab2 = st.tabs(["🎯 Optimization Mode", "🔮 Simulator Mode"])

# ------------------------------------------------------------------------------
# TAB 1: OPTIMIZATION MODE
# ------------------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Biological Targets")
        tgt_lipase = st.number_input("Desired Lipase Inactivation (%)", value=25.0, step=1.0, key="opt_lipase")
        tgt_perox = st.number_input("Desired Peroxidase Inactivation (%)", value=25.0, step=1.0, key="opt_perox")
        tgt_gluten = st.number_input("Minimum Dry Gluten Retention (%)", value=7.50, step=0.1, key="opt_gluten")
    with col2:
        st.subheader("💧 Moisture Targets")
        tgt_moisture = st.number_input("Target Final Atta Moisture (%)", value=10.7, step=0.1, key="opt_moist")

    if st.button("🚀 Optimize Process & Calculate Mechanical Inputs", use_container_width=True):
        with st.spinner("Simulating Biological Constraints & Solving Thermodynamics..."):
            
            t_lip, t_pox, t_glut = get_optimal_window(tgt_lipase, tgt_perox, tgt_gluten, tgt_moisture, enz_params, glut_params)
            
            fig = generate_plotly_fig(tgt_moisture, t_lip, t_pox, t_glut, tgt_lipase, tgt_perox, tgt_gluten, enz_params, glut_params)
            st.plotly_chart(fig, use_container_width=True)

            if t_lip is None or t_pox is None:
                st.error("❌ Enzyme targets cannot be reached within realistic physical bounds (10-150°C).")
            elif t_glut is None:
                st.error("❌ Gluten target cannot be matched.")
            else:
                min_required_temp = max(t_lip, t_pox)
                max_allowed_temp = t_glut
                
                if min_required_temp <= max_allowed_temp:
                    st.success(f"✅ **Valid Operational Window Found:** Target temperatures range from **{min_required_temp:.1f}°C** to **{max_allowed_temp:.1f}°C**.")
                    status = "Valid"
                else:
                    st.error(f"⚠️ **Conflict Zone Detected:** Minimum temp to kill enzymes is **{min_required_temp:.1f}°C**, but gluten degrades entirely by **{max_allowed_temp:.1f}°C**.")
                    status = "Conflict"

                steam_min_temp, water_min_temp, cond_min = get_required_inputs_with_steam_moisture(
                    target_temp=min_required_temp, target_moist=tgt_moisture, moist_type='Atta', 
                    rpm=curr_rpm, grain_rate=curr_grain_rate, efficiency=sys_eff, 
                    moisture_loss=sys_loss, inlet_temp=curr_inlet_temp, inlet_moist_pct=curr_inlet_moist
                )

                steam_max_temp, water_max_temp, cond_max = get_required_inputs_with_steam_moisture(
                    target_temp=max_allowed_temp, target_moist=tgt_moisture, moist_type='Atta', 
                    rpm=curr_rpm, grain_rate=curr_grain_rate, efficiency=sys_eff, 
                    moisture_loss=sys_loss, inlet_temp=curr_inlet_temp, inlet_moist_pct=curr_inlet_moist
                )

                st.subheader("⚙️ Recommended Mechanical Operating Ranges")
                if status == "Valid":
                    st.caption(f"Operate within these mechanical limits to maintain the temperature safely between **{min_required_temp:.1f}°C** (Lower Limit) and **{max_allowed_temp:.1f}°C** (Upper Limit).")
                    
                    actual_water_low = min(water_min_temp, water_max_temp)
                    actual_water_high = max(water_min_temp, water_max_temp)
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        st.metric(label="Required Steam Flow Range", value=f"{steam_min_temp:.0f} - {steam_max_temp:.0f} kg/h")
                    with res_col2:
                        st.metric(label="Tempering Water Range", value=f"{actual_water_low:.0f} - {actual_water_high:.0f} L/h")
                    with res_col3:
                        st.metric(label="Water Added via Condensation", value=f"{cond_min:.1f} - {cond_max:.1f} L/h")
                else:
                    st.warning("Because your biological targets conflict, we are showing the mechanical inputs required for the two conflicting extremes.")
                    
                    st.markdown(f"**🔴 Option A: Save the Enzymes (Hits {min_required_temp:.1f}°C, but destroys gluten):**")
                    col_a1, col_a2, col_a3 = st.columns(3)
                    col_a1.metric("Required Steam", f"{steam_min_temp:.0f} kg/h")
                    col_a2.metric("Required Water", f"{water_min_temp:.0f} L/h")
                    col_a3.metric("Condensation Added", f"{cond_min:.2f} L/h")
                    
                    st.markdown("---")
                    
                    st.markdown(f"**🔵 Option B: Save the Gluten (Hits {max_allowed_temp:.1f}°C, but fails to kill enzymes):**")
                    col_b1, col_b2, col_b3 = st.columns(3)
                    col_b1.metric("Required Steam", f"{steam_max_temp:.0f} kg/h")
                    col_b2.metric("Required Water", f"{water_max_temp:.0f} L/h")
                    col_b3.metric("Condensation Added", f"{cond_max:.2f} L/h")

# ------------------------------------------------------------------------------
# TAB 2: SIMULATOR MODE
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🕹️ Machine Inputs")
    st.caption("Input your desired steam and water rates to predict what will happen to the grain.")
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_steam = st.number_input("Steam Added (kg/h)", value=250.0, step=10.0, key="sim_steam")
    with sim_col2:
        sim_water = st.number_input("Tempering Water Added (L/h)", value=150.0, step=10.0, key="sim_water")

    if st.button("🔬 Run Forward Simulation", use_container_width=True):
        with st.spinner("Calculating physical state and biological outcomes..."):
            
            results = predict_outcomes_from_inputs(
                steam_rate_kg_h=sim_steam,
                water_added_L_h=sim_water,
                grain_rate_mt_h=curr_grain_rate,
                rpm=curr_rpm,
                inlet_temp=curr_inlet_temp,
                inlet_moist_pct=curr_inlet_moist,
                efficiency=sys_eff,
                moisture_loss_milling=sys_loss,
                enz_params=enz_params,
                glut_params=glut_params
            )
            
            # --- Render Graphs ---
            sim_fig = generate_simulation_fig(
                final_temp=results['final_temperature_c'], 
                moisture=results['tempered_wheat_moisture_pct'], 
                lipase_val=results['lipase_inactivated_pct'], 
                perox_val=results['peroxidase_inactivated_pct'], 
                gluten_val=results['gluten_retained_pct'], 
                enz_params=enz_params, 
                glut_params=glut_params
            )
            st.plotly_chart(sim_fig, use_container_width=True)

            # --- Render Metrics ---
            st.markdown("---")
            st.subheader("📊 Predicted Physical State")
            phys_1, phys_2, phys_3, phys_4 = st.columns(4)
            phys_1.metric("Grain Temperature", f"{results['final_temperature_c']:.1f} °C")
            phys_2.metric("Tempered Moisture", f"{results['tempered_wheat_moisture_pct']:.2f} %")
            phys_3.metric("Final Atta Moisture", f"{results['final_atta_moisture_pct']:.2f} %")
            phys_4.metric("Condensation Added", f"{results['steam_condensed_L_h']:.2f} L/h")
            
            st.markdown("---")
            st.subheader("🧬 Predicted Biological Outcomes")
            bio_1, bio_2, bio_3 = st.columns(3)
            bio_1.metric("Lipase Inactivated", f"{results['lipase_inactivated_pct']:.1f} %")
            bio_2.metric("Peroxidase Inactivated", f"{results['peroxidase_inactivated_pct']:.1f} %")
            bio_3.metric("Dry Gluten Retained", f"{results['gluten_retained_pct']:.2f} %")
