"""
Single-Cell Rejuvenation Atlas: Interactive Streamlit Interface
==============================================================
Web-based interactive exploration of cellular rejuvenation landscapes
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

# Configure page
st.set_page_config(
    page_title="Single-Cell Rejuvenation Atlas",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🧬 Single-Cell Rejuvenation Atlas")
st.markdown("Interactive exploration of cellular rejuvenation landscapes")

# Sidebar for navigation
st.sidebar.title("Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    [
        "Data Overview",
        "Trajectory Analysis",
        "Reprogramming Predictor",
        "Senescence Analysis",
        "Pluripotency Scoring",
        "Cell Communication",
        "Digital Twin Simulator",
    ],
)


# Load sample data (in real application, this would be user-uploaded)
@st.cache_data
def load_sample_data():
    """Load sample single-cell data for demonstration"""
    # Generate synthetic single-cell data
    n_cells = 1000
    n_genes = 2000

    # Create synthetic expression matrix
    np.random.seed(42)
    expression_data = np.random.lognormal(mean=1, sigma=1, size=(n_cells, n_genes))

    # Create cell metadata
    cell_types = np.random.choice(
        ["Young", "Aged", "Rejuvenated"], size=n_cells, p=[0.4, 0.4, 0.2]
    )
    batch = np.random.choice(["Batch1", "Batch2", "Batch3"], size=n_cells)

    cell_metadata = pd.DataFrame(
        {
            "cell_id": [f"Cell_{i}" for i in range(n_cells)],
            "cell_type": cell_types,
            "batch": batch,
            "senescence_score": np.random.beta(2, 5, n_cells),
            "pluripotency_score": np.random.beta(3, 3, n_cells),
            "reprogramming_potential": np.random.beta(2, 3, n_cells),
        }
    )

    # Gene metadata
    gene_metadata = pd.DataFrame(
        {
            "gene_id": [f"Gene_{i}" for i in range(n_genes)],
            "gene_type": np.random.choice(
                ["protein_coding", "lncRNA", "miRNA"], size=n_genes, p=[0.8, 0.15, 0.05]
            ),
        }
    )

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(expression_data)

    cell_metadata["UMAP_1"] = pca_coords[:, 0]
    cell_metadata["UMAP_2"] = pca_coords[:, 1]

    return expression_data, cell_metadata, gene_metadata


# Load data
expression_data, cell_metadata, gene_metadata = load_sample_data()

# Analysis sections
if analysis_type == "Data Overview":
    st.header("📊 Data Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cells", len(cell_metadata))
    with col2:
        st.metric("Total Genes", len(gene_metadata))
    with col3:
        st.metric("Cell Types", cell_metadata["cell_type"].nunique())

    # Cell type distribution
    fig_dist = px.histogram(
        cell_metadata,
        x="cell_type",
        title="Cell Type Distribution",
        color="cell_type",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # UMAP plot
    fig_umap = px.scatter(
        cell_metadata,
        x="UMAP_1",
        y="UMAP_2",
        color="cell_type",
        title="UMAP: Cell Type Distribution",
        hover_data=["cell_id"],
    )
    st.plotly_chart(fig_umap, use_container_width=True)

elif analysis_type == "Trajectory Analysis":
    st.header("🔄 Trajectory Analysis: Aging → Rejuvenation")

    # Trajectory visualization
    fig_traj = px.scatter(
        cell_metadata,
        x="UMAP_1",
        y="UMAP_2",
        color="senescence_score",
        size="pluripotency_score",
        title="Rejuvenation Trajectory: Senescence vs Pluripotency",
        color_continuous_scale="RdYlBu_r",
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # Pseudotime analysis
    st.subheader("Pseudotime Ordering")
    pseudotime = np.argsort(
        cell_metadata["senescence_score"] - cell_metadata["pluripotency_score"]
    )
    cell_metadata["pseudotime"] = pseudotime

    fig_pseudo = px.scatter(
        cell_metadata,
        x="pseudotime",
        y="senescence_score",
        color="cell_type",
        title="Pseudotime vs Senescence Score",
    )
    st.plotly_chart(fig_pseudo, use_container_width=True)

elif analysis_type == "Reprogramming Predictor":
    st.header("🎯 Cellular Reprogramming Predictor")

    # Reprogramming potential analysis
    fig_reprog = px.scatter(
        cell_metadata,
        x="senescence_score",
        y="pluripotency_score",
        color="reprogramming_potential",
        size="reprogramming_potential",
        title="Reprogramming Potential Landscape",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_reprog, use_container_width=True)

    # Top reprogramming candidates
    st.subheader("Top Reprogramming Candidates")
    top_candidates = cell_metadata.nlargest(10, "reprogramming_potential")[
        ["cell_id", "cell_type", "reprogramming_potential"]
    ]
    st.dataframe(top_candidates)

elif analysis_type == "Senescence Analysis":
    st.header("🔬 Senescence Marker Analysis")

    # Senescence score distribution
    fig_sen_dist = px.histogram(
        cell_metadata,
        x="senescence_score",
        color="cell_type",
        title="Senescence Score Distribution by Cell Type",
    )
    st.plotly_chart(fig_sen_dist, use_container_width=True)

    # Senescence vs cell type
    fig_sen_box = px.box(
        cell_metadata,
        x="cell_type",
        y="senescence_score",
        title="Senescence Score by Cell Type",
    )
    st.plotly_chart(fig_sen_box, use_container_width=True)

elif analysis_type == "Pluripotency Scoring":
    st.header("🌱 Stem Cell Pluripotency Scoring")

    # Pluripotency score analysis
    fig_pluri_dist = px.histogram(
        cell_metadata,
        x="pluripotency_score",
        color="cell_type",
        title="Pluripotency Score Distribution",
    )
    st.plotly_chart(fig_pluri_dist, use_container_width=True)

    # High pluripotency cells
    high_pluripotency = cell_metadata[cell_metadata["pluripotency_score"] > 0.7]
    st.subheader("High Pluripotency Cells")
    st.write(f"Found {len(high_pluripotency)} cells with pluripotency score > 0.7")

    fig_high_pluri = px.scatter(
        high_pluripotency,
        x="UMAP_1",
        y="UMAP_2",
        color="pluripotency_score",
        size="pluripotency_score",
        title="High Pluripotency Cells in UMAP Space",
    )
    st.plotly_chart(fig_high_pluri, use_container_width=True)

elif analysis_type == "Cell Communication":
    st.header("📡 Cell-Cell Communication Network")

    # Mock communication network data
    communication_data = pd.DataFrame(
        {
            "source_cell_type": ["Young", "Young", "Aged", "Aged", "Rejuvenated"],
            "target_cell_type": [
                "Aged",
                "Rejuvenated",
                "Young",
                "Rejuvenated",
                "Young",
            ],
            "ligand": ["WNT3A", "BMP4", "TGFB1", "FGF2", "NOTCH1"],
            "receptor": ["FZD1", "BMPR1A", "TGFBR1", "FGFR1", "NOTCH2"],
            "interaction_strength": [0.8, 0.6, 0.9, 0.7, 0.5],
        }
    )

    # Communication heatmap
    pivot_data = communication_data.pivot_table(
        values="interaction_strength",
        index="source_cell_type",
        columns="target_cell_type",
        fill_value=0,
    )

    fig_heatmap = px.imshow(
        pivot_data,
        title="Cell-Cell Communication Strength Matrix",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Communication Details")
    st.dataframe(communication_data)

elif analysis_type == "Digital Twin Simulator":
    st.header("🤖 Digital Twin Cell Simulator")

    st.subheader("Advanced ODE-based Rejuvenation Kinetics Modeling")

    # Simulation parameters
    col1, col2 = st.columns(2)
    with col1:
        treatment_strength = st.slider("Treatment Strength", 0.0, 2.0, 1.0, 0.1)
        simulation_time = st.slider("Simulation Time (days)", 1, 60, 21)
        initial_senescence = st.slider("Initial Senescence Level", 0.0, 1.0, 0.8, 0.05)
        initial_pluripotency = st.slider(
            "Initial Pluripotency Level", 0.0, 1.0, 0.2, 0.05
        )

    with col2:
        autophagy_rate = st.slider("Autophagy Enhancement", 0.0, 2.0, 1.0, 0.1)
        inflammation_suppression = st.slider(
            "Inflammation Suppression", 0.0, 2.0, 1.0, 0.1
        )
        dna_repair_boost = st.slider("DNA Repair Enhancement", 0.0, 2.0, 1.0, 0.1)
        metabolic_optimization = st.slider("Metabolic Optimization", 0.0, 2.0, 1.0, 0.1)

    # Advanced ODE-based simulation
    @st.cache_data
    def simulate_cellular_dynamics(params):
        """
        Advanced cellular rejuvenation dynamics using system of ODEs
        """
        import numpy as np
        from scipy.integrate import odeint

        def rejuvenation_ode_system(state, t, params):
            """
            System of ODEs describing cellular rejuvenation dynamics
            """
            S, P, A, I, D, M = (
                state  # Senescence, Pluripotency, Autophagy, Inflammation, DNA damage, Metabolism
            )

            # Parameters
            treatment = params["treatment_strength"]
            autophagy_boost = params["autophagy_rate"]
            inflammation_supp = params["inflammation_suppression"]
            dna_repair = params["dna_repair_boost"]
            metabolic_opt = params["metabolic_optimization"]

            # Cross-regulatory network dynamics

            # Senescence dynamics (decreases with treatment)
            dS_dt = (
                -0.1 * treatment * S * (1 - S)
                - 0.05 * A * S
                + 0.03 * I * (1 - S)
                + 0.02 * D * (1 - S)
            )

            # Pluripotency dynamics (increases with treatment, inhibited by senescence)
            dP_dt = (
                0.08 * treatment * (1 - P) * (1 - 0.5 * S)
                + 0.04 * A * (1 - P)
                - 0.02 * P * I
            )

            # Autophagy dynamics (enhanced by treatment)
            dA_dt = (
                0.12 * autophagy_boost * treatment * (1 - A)
                - 0.08 * A
                + 0.03 * P * (1 - A)
            )

            # Inflammation dynamics (suppressed by treatment)
            dI_dt = (
                -0.15 * inflammation_supp * treatment * I
                + 0.05 * S * (1 - I)
                - 0.06 * A * I
            )

            # DNA damage dynamics (repaired by treatment)
            dD_dt = -0.2 * dna_repair * treatment * D - 0.1 * A * D + 0.04 * I * (1 - D)

            # Metabolic health dynamics (optimized by treatment)
            dM_dt = (
                0.1 * metabolic_opt * treatment * (1 - M)
                + 0.06 * A * (1 - M)
                - 0.03 * S * M
                - 0.02 * I * M
            )

            return [dS_dt, dP_dt, dA_dt, dI_dt, dD_dt, dM_dt]

        # Initial conditions
        initial_conditions = [
            params["initial_senescence"],  # Senescence
            params["initial_pluripotency"],  # Pluripotency
            0.3,  # Autophagy
            0.6,  # Inflammation
            0.5,  # DNA damage
            0.4,  # Metabolic health
        ]

        # Time points
        time_points = np.linspace(0, params["simulation_time"], 200)

        # Solve ODE system
        solution = odeint(
            rejuvenation_ode_system, initial_conditions, time_points, args=(params,)
        )

        return time_points, solution

    # Run simulation
    simulation_params = {
        "treatment_strength": treatment_strength,
        "autophagy_rate": autophagy_rate,
        "inflammation_suppression": inflammation_suppression,
        "dna_repair_boost": dna_repair_boost,
        "metabolic_optimization": metabolic_optimization,
        "initial_senescence": initial_senescence,
        "initial_pluripotency": initial_pluripotency,
        "simulation_time": simulation_time,
    }

    time_points, solution = simulate_cellular_dynamics(simulation_params)

    # Extract trajectories
    senescence_traj = solution[:, 0]
    pluripotency_traj = solution[:, 1]
    autophagy_traj = solution[:, 2]
    inflammation_traj = solution[:, 3]
    dna_damage_traj = solution[:, 4]
    metabolic_health_traj = solution[:, 5]

    # Calculate composite rejuvenation score
    rejuvenation_score = (
        (1 - senescence_traj) * 0.25
        + pluripotency_traj * 0.25
        + autophagy_traj * 0.2
        + (1 - inflammation_traj) * 0.15
        + (1 - dna_damage_traj) * 0.1
        + metabolic_health_traj * 0.05
    )

    # Create comprehensive visualization
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "Cellular Aging Markers",
            "Rejuvenation Markers",
            "Cellular Processes",
            "Composite Rejuvenation Score",
            "Treatment Response",
            "Cellular State Phase Plot",
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Plot 1: Aging markers
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=senescence_traj,
            name="Senescence",
            line={"color": "red", "width": 3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=inflammation_traj,
            name="Inflammation",
            line={"color": "orange", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=dna_damage_traj,
            name="DNA Damage",
            line={"color": "darkred", "width": 2},
        ),
        row=1,
        col=1,
    )

    # Plot 2: Rejuvenation markers
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=pluripotency_traj,
            name="Pluripotency",
            line={"color": "green", "width": 3},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=autophagy_traj,
            name="Autophagy",
            line={"color": "blue", "width": 2},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=metabolic_health_traj,
            name="Metabolism",
            line={"color": "cyan", "width": 2},
        ),
        row=1,
        col=2,
    )

    # Plot 3: Cellular processes dynamics
    process_efficiency = (
        autophagy_traj * (1 - inflammation_traj) * (1 - dna_damage_traj)
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=process_efficiency,
            name="Process Efficiency",
            line={"color": "purple", "width": 3},
        ),
        row=2,
        col=1,
    )

    # Plot 4: Composite rejuvenation score
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=rejuvenation_score,
            name="Rejuvenation Score",
            line={"color": "darkgreen", "width": 4},
        ),
        row=2,
        col=2,
    )

    # Plot 5: Treatment response curve
    treatment_response = (
        np.exp(-0.1 * time_points) * (1 - treatment_strength) + treatment_strength
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=treatment_response,
            name="Treatment Effect",
            line={"color": "gold", "width": 3},
        ),
        row=3,
        col=1,
    )

    # Plot 6: Phase plot (Senescence vs Pluripotency)
    fig.add_trace(
        go.Scatter(
            x=senescence_traj,
            y=pluripotency_traj,
            mode="lines+markers",
            name="Cellular Trajectory",
            line={"color": "magenta", "width": 2},
            marker={"size": 3},
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        height=900, title_text="Digital Twin: Comprehensive Cellular Dynamics"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quantitative analysis
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        initial_score = rejuvenation_score[0]
        final_score = rejuvenation_score[-1]
        improvement = final_score - initial_score
        st.metric(
            "Rejuvenation Improvement",
            f"{improvement:.3f}",
            delta=f"{improvement / initial_score * 100:.1f}%",
        )

    with col2:
        senescence_reduction = initial_senescence - senescence_traj[-1]
        st.metric(
            "Senescence Reduction",
            f"{senescence_reduction:.3f}",
            delta=f"-{senescence_reduction / initial_senescence * 100:.1f}%",
        )

    with col3:
        pluripotency_gain = pluripotency_traj[-1] - initial_pluripotency
        st.metric(
            "Pluripotency Gain",
            f"{pluripotency_gain:.3f}",
            delta=f"+{pluripotency_gain / (1 - initial_pluripotency) * 100:.1f}%",
        )

    with col4:
        half_life_idx = np.argmin(np.abs(senescence_traj - initial_senescence / 2))
        half_life = time_points[half_life_idx] if half_life_idx > 0 else simulation_time
        st.metric("Senescence Half-Life", f"{half_life:.1f} days")

    # Advanced analytics
    st.subheader("🔬 Advanced Cellular Analytics")

    # Calculate cellular age
    cellular_ages = []
    for i in range(len(time_points)):
        # Weighted cellular age based on multiple factors
        age_score = (
            senescence_traj[i] * 0.4
            + inflammation_traj[i] * 0.3
            + dna_damage_traj[i] * 0.2
            + (1 - metabolic_health_traj[i]) * 0.1
        )
        # Convert to biological age (assuming baseline of 40 years)
        biological_age = 20 + age_score * 60  # Range 20-80 years
        cellular_ages.append(biological_age)

    # Rejuvenation velocity
    rejuv_velocity = np.gradient(rejuvenation_score, time_points)

    # Display key insights
    st.markdown("### 📊 Key Simulation Insights:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        **Cellular Transformation:**
        - Initial biological age: {cellular_ages[0]:.1f} years
        - Final biological age: {cellular_ages[-1]:.1f} years
        - Age reversal: {cellular_ages[0] - cellular_ages[-1]:.1f} years

        **Treatment Efficacy:**
        - Peak rejuvenation velocity: {np.max(rejuv_velocity):.4f}/day
        - Time to peak effect: {time_points[np.argmax(rejuv_velocity)]:.1f} days
        - Overall treatment success: {(improvement > 0.2) * 100:.0f}%
        """)

    with col2:
        st.markdown(f"""
        **Molecular Mechanisms:**
        - Autophagy activation: {(autophagy_traj[-1] - autophagy_traj[0]):.3f}
        - Inflammation reduction: {(inflammation_traj[0] - inflammation_traj[-1]):.3f}
        - DNA repair efficiency: {(dna_damage_traj[0] - dna_damage_traj[-1]):.3f}

        **Stability Analysis:**
        - Senescence stability: {1 - np.std(senescence_traj[-20:]):.3f}
        - Pluripotency maintenance: {1 - np.std(pluripotency_traj[-20:]):.3f}
        """)

    # Treatment optimization recommendations
    st.subheader("💊 Treatment Optimization Recommendations")

    if improvement < 0.1:
        st.warning("⚠️ Low treatment efficacy detected. Consider:")
        st.markdown("- Increasing treatment strength or duration")
        st.markdown("- Combining with autophagy enhancers")
        st.markdown("- Adding anti-inflammatory agents")
    elif improvement > 0.5:
        st.success("✅ Excellent treatment response! Consider:")
        st.markdown("- Maintaining current protocol")
        st.markdown("- Monitoring for long-term stability")
        st.markdown("- Potential for dose optimization")
    else:
        st.info("ℹ️ Moderate treatment response. Consider:")
        st.markdown("- Fine-tuning treatment parameters")
        st.markdown("- Adding metabolic optimization")
        st.markdown("- Extended treatment duration")

    # Export simulation data
    if st.button("📥 Export Simulation Data"):
        simulation_df = pd.DataFrame(
            {
                "Time_days": time_points,
                "Senescence": senescence_traj,
                "Pluripotency": pluripotency_traj,
                "Autophagy": autophagy_traj,
                "Inflammation": inflammation_traj,
                "DNA_Damage": dna_damage_traj,
                "Metabolic_Health": metabolic_health_traj,
                "Rejuvenation_Score": rejuvenation_score,
                "Biological_Age": cellular_ages,
            }
        )

        csv = simulation_df.to_csv(index=False)
        st.download_button(
            label="Download simulation data as CSV",
            data=csv,
            file_name=f"cellular_simulation_data_{treatment_strength}x_{simulation_time}d.csv",
            mime="text/csv",
        )

# Footer
st.markdown("---")
st.markdown(
    "Single-Cell Rejuvenation Atlas | Interactive Cellular Aging Analysis Platform"
)
