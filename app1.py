import streamlit as st
import numpy as np
from scipy.optimize import minimize, brentq, minimize_scalar
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# 1. FIXED CONSTANTS & FACTORY DATA
# ==============================================================================
R = 8.314  # Universal Gas Constant (J/mol·K)

BASE_PARAMS = {
    "Lipase": {"Eab": 45945.38, "beta": 0.801218, "dH": 117420.10, "dS": 365.4445},
    "Peroxidase": {"Eab": 32662.39, "beta": 0.650912, "dH": 287284.26, "dS": 841.7062},
}

ENZ_TRAIN_DATA = [
    {"temp": 25.00, "moist": 9.40,  "Lipase": 0,     "Peroxidase": 0},
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

GLUT_TEST_TEMPS = np.array([68.0, 68.0, 62.0, 68.0, 65.0, 72.0, 69.0, 66.0, 59.0, 70.0])
GLUT_TEST_GLUTENS = np.array([8.66, 4.50, 8.65, 7.86, 7.48, 3.35, 7.27, 8.77, 7.46, 8.3])

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
    def mse_glut(params):
        w_res, w_drop1, w_drop2, t1, t2, k1, k2 = params
        preds = w_res + (w_drop1 / (1.0 + np.exp(k1 * (GLUT_TEST_TEMPS - t1)))) + (w_drop2 / (1.0 + np.exp(k2 * (GLUT_TEST_TEMPS - t2))))
        return np.sqrt(np.mean((preds - GLUT_TEST_GLUTENS)**2))

    bounds = [(1.0, 4.0), (0.1, 5.0), (0.1, 7.0), (50.0, 75.0), (70.0, 95.0), (0.1, 1.5), (0.1, 1.5)]
    res = minimize(mse_glut, [3.0, 2.0, 4.0, 60.0, 80.0, 0.3, 0.4], bounds=bounds, method='L-BFGS-B')
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
# 4. THERMODYNAMIC SOLVER
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

# ==============================================================================
# 5. VISUALIZATION ENGINE
# ==============================================================================
def generate_plotly_fig(moisture, t_lip, t_pox, t_glut, target_l, target_p, target_g, enz_params, glut_params):
    # Dynamically scale X-Axis to ensure points are never cut off
    max_t = max([100, t_lip or 0, t_pox or 0, t_glut or 0]) + 10
    
    # Convert NumPy array strictly to a Python list for Plotly compatibility
    temps_array = np.linspace(20, max_t, 300)
    temps = temps_array.tolist() 
    
    lip_vals = [predict_enzyme(t, moisture, "Lipase", enz_params) for t in temps]
    pox_vals = [predict_enzyme(t, moisture, "Peroxidase", enz_params) for t in temps]
    glut_vals = [predict_gluten(t, glut_params) for t in temps]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Explicitly bind the converted lists to the X and Y axes
    fig.add_trace(go.Scatter(x=temps, y=lip_vals, mode='lines', name='Lipase Inact.', line=dict(color='blue', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=pox_vals, mode='lines', name='Peroxidase Inact.', line=dict(color='orange', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=temps, y=glut_vals, mode='lines', name='Gluten Retention', line=dict(color='purple', width=3, dash='dash')), secondary_y=True)

    # Shaded operational windows
    min_required_temp = max(t_lip or 0, t_pox or 0) if (t_lip or t_pox) else None
    max_allowed_temp = t_glut

    if min_required_temp and max_allowed_temp:
        if min_required_temp <= max_allowed_temp:
            fig.add_vrect(x0=min_required_temp, x1=max_allowed_temp, fillcolor="green", opacity=0.15, layer="below", line_width=0, annotation_text="Valid Window")
        else:
            fig.add_vrect(x0=max_allowed_temp, x1=min_required_temp, fillcolor="red", opacity=0.15, layer="below", line_width=0, annotation_text="Conflict Zone")

    # Target Horizontal Lines
    fig.add_hline(y=target_l, line_dash="dot", line_color="blue", opacity=0.4, secondary_y=False)
    fig.add_hline(y=target_p, line_dash="dot", line_color="orange", opacity=0.4, secondary_y=False)
    fig.add_hline(y=target_g, line_dash="dot", line_color="purple", opacity=0.4, secondary_y=True)

    # Intersection markers
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

# ==============================================================================
# 6. STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Integrated Plant Optimizer", layout="wide")

st.title("🏭 Steam Conveyor Machine Optimizer")
st.markdown("Calculate the exact temperature window required for biological targets and automatically resolve the necessary mechanical inputs (steam and water) to achieve them.")

# --- SIDEBAR: CALIBRATION ---
st.sidebar.header("⚙️ Mechanical Calibration")
st.sidebar.markdown("Baseline factory trial data used to calculate tempering efficiency and milling loss.")
cal_grain_flow = st.sidebar.number_input("Grain Flow (kg/h)", value=14000.0, step=100.0)
cal_inlet_moist = st.sidebar.number_input("Inlet Moisture (%)", value=10.04, step=0.1)
cal_water_added = st.sidebar.number_input("Water Added (L/h)", value=450.0, step=10.0)
cal_wheat_after_temp = st.sidebar.number_input("Wheat Moisture After Temp (%)", value=11.67, step=0.1)
cal_atta_final = st.sidebar.number_input("Final Atta Moisture (%)", value=9.00, step=0.1)

# Dynamically calculate mechanical efficiency
sys_eff, sys_loss = calibrate_system(cal_grain_flow, cal_inlet_moist, cal_wheat_after_temp, cal_water_added, cal_atta_final)

st.sidebar.markdown("---")
st.sidebar.info(f"**Tempering Efficiency:** {sys_eff*100:.2f}%\n\n**Milling Moisture Loss:** {sys_loss:.2f}%")

# Initialize digital twins
enz_params = calibrate_enzymes()
glut_params = calibrate_gluten()

# --- MAIN PAGE: INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Biological Targets")
    st.caption("Determine process minimums based on laboratory benchmarks.")
    tgt_lipase = st.number_input("Desired Lipase Inactivation (%)", value=25.0, step=1.0)
    tgt_perox = st.number_input("Desired Peroxidase Inactivation (%)", value=25.0, step=1.0)
    tgt_gluten = st.number_input("Minimum Dry Gluten Retention (%)", value=7.50, step=0.1)
    tgt_moisture = st.number_input("Target Final Atta Moisture (%)", value=10.7, step=0.1)

with col2:
    st.subheader("⚙️ Mechanical Operating Conditions")
    st.caption("Current state of the production floor.")
    curr_grain_rate = st.number_input("Grain Rate (MT/h)", value=5.0, step=0.5)
    curr_rpm = st.number_input("Conveyor RPM", value=40.0, step=1.0)
    curr_inlet_temp = st.number_input("Inlet Grain Temp (°C)", value=38.0, step=1.0)
    curr_inlet_moist = st.number_input("Current Inlet Moisture (%)", value=10.04, step=0.1)

st.markdown("---")

# --- MAIN PAGE: EXECUTION ---
if st.button("🚀 Optimize Process & Calculate Mechanical Inputs", use_container_width=True):
    with st.spinner("Simulating Biological Constraints & Solving Thermodynamics..."):
        
        # 1. Biological Optimization
        t_lip, t_pox, t_glut = get_optimal_window(tgt_lipase, tgt_perox, tgt_gluten, tgt_moisture, enz_params, glut_params)
        
        # Render Graphical Result
        fig = generate_plotly_fig(tgt_moisture, t_lip, t_pox, t_glut, tgt_lipase, tgt_perox, tgt_gluten, enz_params, glut_params)
        st.plotly_chart(fig, use_container_width=True)

        if t_lip is None or t_pox is None:
            st.error("❌ Enzyme targets cannot be reached within realistic physical bounds (10-150°C).")
        elif t_glut is None:
            st.error("❌ Gluten target cannot be matched.")
        else:
            min_required_temp = max(t_lip, t_pox)
            max_allowed_temp = t_glut
            
            # Display Window Results
            if min_required_temp <= max_allowed_temp:
                st.success(f"✅ **Valid Operational Window Found:** Target temperatures range from **{min_required_temp:.1f}°C** to **{max_allowed_temp:.1f}°C**.")
                status = "Valid"
            else:
                st.error(f"⚠️ **Conflict Zone Detected:** Minimum temp to kill enzymes is **{min_required_temp:.1f}°C**, but gluten degrades entirely by **{max_allowed_temp:.1f}°C**.")
                status = "Conflict"

            # 2. Thermodynamic Mass Balance Solver (LOWER LIMIT)
            steam_min_temp, water_min_temp, cond_min = get_required_inputs_with_steam_moisture(
                target_temp=min_required_temp, 
                target_moist=tgt_moisture, 
                moist_type='Atta', 
                rpm=curr_rpm, 
                grain_rate=curr_grain_rate, 
                efficiency=sys_eff, 
                moisture_loss=sys_loss, 
                inlet_temp=curr_inlet_temp, 
                inlet_moist_pct=curr_inlet_moist
            )

            # 3. Thermodynamic Mass Balance Solver (UPPER LIMIT)
            steam_max_temp, water_max_temp, cond_max = get_required_inputs_with_steam_moisture(
                target_temp=max_allowed_temp, 
                target_moist=tgt_moisture, 
                moist_type='Atta', 
                rpm=curr_rpm, 
                grain_rate=curr_grain_rate, 
                efficiency=sys_eff, 
                moisture_loss=sys_loss, 
                inlet_temp=curr_inlet_temp, 
                inlet_moist_pct=curr_inlet_moist
            )

            # Display Mechanical Outputs
            st.subheader("⚙️ Recommended Mechanical Operating Ranges")
            
            if status == "Valid":
                st.caption(f"Operate within these mechanical limits to maintain the temperature safely between **{min_required_temp:.1f}°C** (Lower Limit) and **{max_allowed_temp:.1f}°C** (Upper Limit).")
                
                # Sort values so they display properly as "Low - High"
                actual_water_low = min(water_min_temp, water_max_temp)
                actual_water_high = max(water_min_temp, water_max_temp)
                
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric(label="Required Steam Flow Range", value=f"{steam_min_temp:.1f} - {steam_max_temp:.1f} kg/h")
                with res_col2:
                    st.metric(label="Tempering Water Range", value=f"{actual_water_low:.1f} - {actual_water_high:.1f} L/h")
                with res_col3:
                    st.metric(label="Water Added via Condensation", value=f"{cond_min:.1f} - {cond_max:.1f} L/h")
                    
                st.info("💡 **Note:** Hitting the upper limits of steam will result in more condensation, which automatically means you need slightly less liquid tempering water to hit your final moisture target.")
            
            else:
                st.warning("Because your biological targets conflict, we are showing the mechanical inputs required for the two conflicting extremes.")
                
                st.markdown(f"**🔴 Option A: Save the Enzymes (Hits {min_required_temp:.1f}°C, but destroys gluten):**")
                col_a1, col_a2, col_a3 = st.columns(3)
                col_a1.metric("Required Steam", f"{steam_min_temp:.1f} kg/h")
                col_a2.metric("Required Water", f"{water_min_temp:.1f} L/h")
                col_a3.metric("Condensation Added", f"{cond_min:.2f} L/h")
                
                st.markdown("---")
                
                st.markdown(f"**🔵 Option B: Save the Gluten (Hits {max_allowed_temp:.1f}°C, but fails to kill enzymes):**")
                col_b1, col_b2, col_b3 = st.columns(3)
                col_b1.metric("Required Steam", f"{steam_max_temp:.1f} kg/h")
                col_b2.metric("Required Water", f"{water_max_temp:.1f} L/h")
                col_b3.metric("Condensation Added", f"{cond_max:.2f} L/h")