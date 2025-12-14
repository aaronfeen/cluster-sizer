# app.py
#
# Streamlit-based AI cluster topology planner
# Run with: streamlit run app.py

import math
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from graphviz import Digraph


# ------------------------------
# Core constants and helpers
# ------------------------------

GPUS_PER_NODE = 8
GPU_POWER_KW = 1.4  # per MI355X GPU


@dataclass
class RackProfile:
    rack_units: int = 42
    gpu_node_u: int = 4
    storage_node_u: int = 2
    mgmt_node_u: int = 1
    switch_u: int = 1


@dataclass
class NetworkDesignInput:
    name: str
    topology: str  # "Rail", "Dragonfly+", "Fat Tree"
    tiers: int     # 1=leaf only, 2=leaf+spine, 3=leaf+spine+core
    oversub_ratio: float   # server-side bw : uplink bw (>=1.0)

    num_endpoints: int     # number of nodes (or logical endpoints) attached
    nics_per_endpoint: int # NICs per endpoint on this network

    leaf_ports: int
    spine_ports: Optional[int] = None
    core_ports: Optional[int] = None


@dataclass
class NetworkDesignResult:
    name: str
    topology: str
    tiers: int
    oversub_ratio: float
    num_endpoints: int
    nics_per_endpoint: int
    total_server_ports: int
    leaf_ports: int
    spine_ports: Optional[int]
    core_ports: Optional[int]
    num_leaf: int
    num_spine: int
    num_core: int


def ceil_div(a: int, b: int) -> int:
    return math.ceil(a / b) if b > 0 else 0


def capacity_per_switch(ports: int, oversub_ratio: float) -> tuple[int, int]:
    """
    Given total ports on a switch and a desired oversubscription ratio R
    (server-side bandwidth : uplink bandwidth), compute:

    - D: number of downlink (server-facing) ports
    - U: number of uplink (upwards) ports

    Subject to:
      D + U <= ports
      D ≈ R * U

    This is a simple heuristic for sizing.
    """
    if ports <= 1:
        return ports, 0

    R = max(1.0, oversub_ratio)

    # Ideal D from algebra: D + D/R <= ports  =>  D(1 + 1/R) <= ports
    ideal_D = int(ports / (1.0 + 1.0 / R))
    if ideal_D < 1:
        ideal_D = 1

    ideal_U = math.ceil(ideal_D / R)

    # Ensure we do not exceed port budget
    while ideal_D + ideal_U > ports and ideal_D > 1:
        ideal_D -= 1
        ideal_U = math.ceil(ideal_D / R)

    # Final safety adjustment
    if ideal_D + ideal_U > ports:
        ideal_U = max(0, ports - ideal_D)

    return ideal_D, ideal_U


def design_network(cfg: NetworkDesignInput) -> NetworkDesignResult:
    total_server_ports = cfg.num_endpoints * cfg.nics_per_endpoint
    if total_server_ports == 0:
        return NetworkDesignResult(
            name=cfg.name,
            topology=cfg.topology,
            tiers=cfg.tiers,
            oversub_ratio=cfg.oversub_ratio,
            num_endpoints=cfg.num_endpoints,
            nics_per_endpoint=cfg.nics_per_endpoint,
            total_server_ports=0,
            leaf_ports=cfg.leaf_ports,
            spine_ports=cfg.spine_ports,
            core_ports=cfg.core_ports,
            num_leaf=0,
            num_spine=0,
            num_core=0,
        )

    R = max(1.0, cfg.oversub_ratio)

    # Leaf tier (always present)
    leaf_down_per_switch, leaf_up_per_switch = capacity_per_switch(cfg.leaf_ports, R)
    num_leaf = ceil_div(total_server_ports, leaf_down_per_switch)
    total_leaf_uplinks = num_leaf * leaf_up_per_switch

    num_spine = 0
    num_core = 0
    spine_ports = cfg.spine_ports
    core_ports = cfg.core_ports

    if cfg.tiers == 1:
        # Leaf-only: uplinks are unused (or aggregate), model stops at leaf.
        num_spine = 0
        num_core = 0

    elif cfg.tiers == 2:
        # Leaf + Spine: spine ports face leaves only (no further uplinks)
        if spine_ports is None or spine_ports < 1:
            spine_ports = cfg.leaf_ports  # fallback
        num_spine = ceil_div(total_leaf_uplinks, spine_ports)
        num_core = 0

    else:
        # Tiers == 3: Leaf + Spine + Core
        if spine_ports is None or spine_ports < 1:
            spine_ports = cfg.leaf_ports  # fallback
        if core_ports is None or core_ports < 1:
            core_ports = spine_ports      # fallback

        # Spine tier: some ports down to leaves, some up to cores
        spine_down_per_switch, spine_up_per_switch = capacity_per_switch(spine_ports, R)

        # First, enough spines to terminate all leaf uplinks
        num_spine = ceil_div(total_leaf_uplinks, spine_down_per_switch)
        total_spine_uplinks = num_spine * spine_up_per_switch

        # Core tier: terminate all spine uplinks
        num_core = ceil_div(total_spine_uplinks, core_ports)

    return NetworkDesignResult(
        name=cfg.name,
        topology=cfg.topology,
        tiers=cfg.tiers,
        oversub_ratio=cfg.oversub_ratio,
        num_endpoints=cfg.num_endpoints,
        nics_per_endpoint=cfg.nics_per_endpoint,
        total_server_ports=total_server_ports,
        leaf_ports=cfg.leaf_ports,
        spine_ports=spine_ports,
        core_ports=core_ports,
        num_leaf=num_leaf,
        num_spine=num_spine,
        num_core=num_core,
    )


def build_network_graph(design: NetworkDesignResult, server_label: str) -> Digraph:
    """
    Build a simplified, aggregated Graphviz diagram:

      Core switches (if any)  (top)
      Spine switches (if any)
      Leaf switches
      Servers / endpoints      (bottom)
    """
    dot = Digraph(comment=f"{design.name} network")
    dot.attr(rankdir="TB", splines="polyline")
    dot.attr("node", shape="box")

    previous_node = None

    if design.num_core > 0:
        core_node = f"{design.name}_core"
        dot.node(
            core_node,
            f"Core switches\n(count = {design.num_core})",
        )
        previous_node = core_node

    if design.num_spine > 0:
        spine_node = f"{design.name}_spine"
        dot.node(
            spine_node,
            f"Spine switches\n(count = {design.num_spine})",
        )
        if previous_node:
            dot.edge(previous_node, spine_node)
        previous_node = spine_node

    if design.num_leaf > 0:
        leaf_node = f"{design.name}_leaf"
        dot.node(
            leaf_node,
            f"Leaf switches\n(count = {design.num_leaf})",
        )
        if previous_node:
            dot.edge(previous_node, leaf_node)
        previous_node = leaf_node

    server_node = f"{design.name}_servers"
    if design.num_endpoints > 0 and design.total_server_ports > 0:
        dot.node(
            server_node,
            f"{server_label}\n(nodes = {design.num_endpoints}, "
            f"NICs/node = {design.nics_per_endpoint})",
        )
    else:
        dot.node(server_node, f"{server_label}\n(no endpoints)")

    if previous_node:
        dot.edge(previous_node, server_node)

    return dot


def racks_needed_for_nodes(
    num_nodes: int,
    node_u: int,
    node_power_kw: float,
    rack_profile: RackProfile,
    rack_power_kw: float,
) -> tuple[int, int, int, int]:
    """
    Compute how many racks are required for a homogeneous set of nodes,
    constrained by both:

    - Space: rack_profile.rack_units / node_u
    - Power: rack_power_kw / node_power_kw

    Returns:
        racks_needed, nodes_per_rack, cap_by_u, cap_by_power
    """
    if num_nodes == 0:
        return 0, 0, 0, 0

    # Capacity by rack units
    cap_by_u = rack_profile.rack_units // node_u if node_u > 0 else 0

    # Capacity by rack power
    if node_power_kw > 0:
        cap_by_power = int(rack_power_kw // node_power_kw)
    else:
        cap_by_power = cap_by_u

    if cap_by_u <= 0 or cap_by_power <= 0:
        # Cannot fit even one node per rack without violating U or power
        nodes_per_rack = 0
        racks_needed = num_nodes  # conceptually one rack per node but invalid
        return racks_needed, nodes_per_rack, cap_by_u, cap_by_power

    nodes_per_rack = min(cap_by_u, cap_by_power)
    racks_needed = ceil_div(num_nodes, nodes_per_rack)

    return racks_needed, nodes_per_rack, cap_by_u, cap_by_power


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(
    page_title="AI Cluster & Network Planner",
    layout="wide",
)

st.title("AI Cluster & Network Planner")

st.markdown(
    """
This tool estimates an AI cluster layout based on GPU, storage, and management nodes,
and designs three networks:

- **Backside GPU fabric**
- **Frontside storage network**
- **Out-of-band management network**

It assumes **MI355X GPUs** at **1.4 kW per GPU** with **8 GPUs per compute node**.
"""
)

# ---- Sidebar inputs ----
st.sidebar.header("Cluster inputs")

# Node counts
st.sidebar.subheader("Node counts")
num_gpu_nodes = st.sidebar.number_input(
    "Number of GPU compute nodes",
    min_value=0,
    value=64,
    step=1,
)
num_storage_nodes = st.sidebar.number_input(
    "Number of storage nodes",
    min_value=0,
    value=8,
    step=1,
)
num_mgmt_nodes = st.sidebar.number_input(
    "Number of management nodes",
    min_value=0,
    value=4,
    step=1,
)

# NIC configuration
st.sidebar.subheader("NIC configuration")

backside_nics_per_gpu = st.sidebar.selectbox(
    "Backside NICs per GPU (GPU fabric)",
    options=[1, 4, 8],
    index=1,  # default 4
)
# Note: backside NICs per node = GPUS_PER_NODE * backside_nics_per_gpu
backside_nics_per_node = GPUS_PER_NODE * backside_nics_per_gpu

frontside_nics_per_node = st.sidebar.selectbox(
    "Frontside NICs per node (GPU & storage)",
    options=[1, 2],
    index=1,  # default 2
)

mgmt_nics_per_node = st.sidebar.selectbox(
    "Management NICs per node (GPU, storage, mgmt)",
    options=[1, 2],
    index=0,  # default 1
)

# Datacenter constraints
st.sidebar.header("Datacenter constraints")

max_it_power_kw = st.sidebar.number_input(
    "Datacenter IT power limit (kW)",
    min_value=10.0,
    value=2000.0,
    step=10.0,
)

max_racks = st.sidebar.number_input(
    "Datacenter rack count limit",
    min_value=1,
    value=56,
    step=1,
)

max_rack_power_kw = st.sidebar.number_input(
    "Maximum rack power (kW)",
    min_value=1.0,
    value=51.0,
    step=1.0,
)

# Rack profile
st.sidebar.subheader("Rack configuration")

rack_units = st.sidebar.number_input(
    "Rack size (U)",
    min_value=20,
    value=42,
    step=1,
)

gpu_node_u = st.sidebar.number_input(
    "GPU node height (U)",
    min_value=1,
    value=4,
    step=1,
)

storage_node_u = st.sidebar.number_input(
    "Storage node height (U)",
    min_value=1,
    value=2,
    step=1,
)

mgmt_node_u = st.sidebar.number_input(
    "Management node height (U)",
    min_value=1,
    value=1,
    step=1,
)

switch_u = st.sidebar.number_input(
    "Switch height (U)",
    min_value=1,
    value=1,
    step=1,
)

rack_profile = RackProfile(
    rack_units=rack_units,
    gpu_node_u=gpu_node_u,
    storage_node_u=storage_node_u,
    mgmt_node_u=mgmt_node_u,
    switch_u=switch_u,
)

# Power model
st.sidebar.subheader("Per-node and per-switch power (kW)")
extra_gpu_node_power_kw = st.sidebar.number_input(
    "Additional power per GPU node (non-GPU, kW)",
    min_value=0.0,
    value=2.0,
    step=0.1,
)
storage_node_power_kw = st.sidebar.number_input(
    "Power per storage node (kW)",
    min_value=0.0,
    value=0.8,
    step=0.1,
)
mgmt_node_power_kw = st.sidebar.number_input(
    "Power per management node (kW)",
    min_value=0.0,
    value=0.5,
    step=0.1,
)
switch_power_kw = st.sidebar.number_input(
    "Power per network switch (kW)",
    min_value=0.0,
    value=0.5,
    step=0.1,
)

# Network configuration
st.sidebar.header("Network configuration")

topology_options = ["Rail", "Dragonfly+", "Fat Tree"]

with st.sidebar.expander("Backside network (GPU fabric)", expanded=True):
    backside_topology = st.selectbox(
        "Topology (backside)",
        options=topology_options,
        index=0,  # default Rail
        key="backside_topology",
    )
    backside_tiers = st.select_slider(
        "Number of switch tiers",
        options=[1, 2, 3],
        value=2,
        key="backside_tiers",
    )
    backside_oversub = st.number_input(
        "Oversubscription ratio (server:uplink)",
        min_value=1.0,
        value=1.0,
        step=0.5,
        key="backside_oversub",
    )
    backside_leaf_ports = st.number_input(
        "Leaf switch ports (backside)",
        min_value=8,
        value=128,
        step=1,
        key="backside_leaf_ports",
    )
    backside_spine_ports = st.number_input(
        "Spine switch ports (backside)",
        min_value=8,
        value=128,
        step=1,
        key="backside_spine_ports",
    )
    backside_core_ports = st.number_input(
        "Core switch ports (backside)",
        min_value=8,
        value=128,
        step=1,
        key="backside_core_ports",
    )

with st.sidebar.expander("Frontside network (GPU ↔ storage)", expanded=False):
    frontside_topology = st.selectbox(
        "Topology (frontside)",
        options=topology_options,
        index=2,
        key="frontside_topology",
    )
    frontside_tiers = st.select_slider(
        "Number of switch tiers",
        options=[1, 2, 3],
        value=1,
        key="frontside_tiers",
    )
    frontside_oversub = st.number_input(
        "Oversubscription ratio (server:uplink)",
        min_value=1.0,
        value=1.0,
        step=0.5,
        key="frontside_oversub",
    )
    frontside_leaf_ports = st.number_input(
        "Leaf switch ports (frontside)",
        min_value=8,
        value=64,
        step=1,
        key="frontside_leaf_ports",
    )
    frontside_spine_ports = st.number_input(
        "Spine switch ports (frontside)",
        min_value=8,
        value=64,
        step=1,
        key="frontside_spine_ports",
    )
    frontside_core_ports = st.number_input(
        "Core switch ports (frontside)",
        min_value=8,
        value=128,
        step=1,
        key="frontside_core_ports",
    )

with st.sidebar.expander("Management network (out-of-band)", expanded=False):
    mgmt_topology = st.selectbox(
        "Topology (management)",
        options=topology_options,
        index=0,
        key="mgmt_topology",
    )
    mgmt_tiers = st.select_slider(
        "Number of switch tiers",
        options=[1, 2, 3],
        value=1,
        key="mgmt_tiers",
    )
    mgmt_oversub = st.number_input(
        "Oversubscription ratio (server:uplink)",
        min_value=1.0,
        value=4.0,
        step=0.5,
        key="mgmt_oversub",
    )
    mgmt_leaf_ports = st.number_input(
        "Leaf switch ports (management)",
        min_value=8,
        value=64,
        step=1,
        key="mgmt_leaf_ports",
    )
    mgmt_spine_ports = st.number_input(
        "Spine switch ports (management)",
        min_value=8,
        value=64,
        step=1,
        key="mgmt_spine_ports",
    )
    mgmt_core_ports = st.number_input(
        "Core switch ports (management)",
        min_value=8,
        value=48,
        step=1,
        key="mgmt_core_ports",
    )

# ------------------------------
# Calculations
# ------------------------------

# Power calculations (per-node)
gpu_node_power_kw = GPUS_PER_NODE * GPU_POWER_KW + extra_gpu_node_power_kw
total_gpu_power_kw = num_gpu_nodes * gpu_node_power_kw
total_storage_power_kw = num_storage_nodes * storage_node_power_kw
total_mgmt_power_kw = num_mgmt_nodes * mgmt_node_power_kw

# Network design inputs
backside_cfg = NetworkDesignInput(
    name="Backside",
    topology=backside_topology,
    tiers=backside_tiers,
    oversub_ratio=backside_oversub,
    num_endpoints=num_gpu_nodes,
    nics_per_endpoint=backside_nics_per_node,
    leaf_ports=backside_leaf_ports,
    spine_ports=backside_spine_ports,
    core_ports=backside_core_ports,
)

frontside_cfg = NetworkDesignInput(
    name="Frontside",
    topology=frontside_topology,
    tiers=frontside_tiers,
    oversub_ratio=frontside_oversub,
    num_endpoints=num_gpu_nodes + num_storage_nodes,
    nics_per_endpoint=frontside_nics_per_node,
    leaf_ports=frontside_leaf_ports,
    spine_ports=frontside_spine_ports,
    core_ports=frontside_core_ports,
)

mgmt_cfg = NetworkDesignInput(
    name="Management",
    topology=mgmt_topology,
    tiers=mgmt_tiers,
    oversub_ratio=mgmt_oversub,
    num_endpoints=num_gpu_nodes + num_storage_nodes + num_mgmt_nodes,
    nics_per_endpoint=mgmt_nics_per_node,
    leaf_ports=mgmt_leaf_ports,
    spine_ports=mgmt_spine_ports,
    core_ports=mgmt_core_ports,
)

backside_design = design_network(backside_cfg)
frontside_design = design_network(frontside_cfg)
mgmt_design = design_network(mgmt_cfg)

# Network switch counts
total_switches = (
    backside_design.num_leaf + backside_design.num_spine + backside_design.num_core
    + frontside_design.num_leaf + frontside_design.num_spine + frontside_design.num_core
    + mgmt_design.num_leaf + mgmt_design.num_spine + mgmt_design.num_core
)

total_switch_power_kw = total_switches * switch_power_kw

# Total IT power (includes GPUs, nodes, storage, mgmt, NICs implicitly, and switches)
total_it_power_kw = (
    total_gpu_power_kw
    + total_storage_power_kw
    + total_mgmt_power_kw
    + total_switch_power_kw
)

# Rack counts using both power and U constraints
gpu_racks, gpu_nodes_per_rack, gpu_cap_by_u, gpu_cap_by_pwr = racks_needed_for_nodes(
    num_gpu_nodes,
    rack_profile.gpu_node_u,
    gpu_node_power_kw,
    rack_profile,
    max_rack_power_kw,
)

storage_racks, storage_nodes_per_rack, storage_cap_by_u, storage_cap_by_pwr = (
    racks_needed_for_nodes(
        num_storage_nodes,
        rack_profile.storage_node_u,
        storage_node_power_kw,
        rack_profile,
        max_rack_power_kw,
    )
)

mgmt_racks, mgmt_nodes_per_rack, mgmt_cap_by_u, mgmt_cap_by_pwr = (
    racks_needed_for_nodes(
        num_mgmt_nodes,
        rack_profile.mgmt_node_u,
        mgmt_node_power_kw,
        rack_profile,
        max_rack_power_kw,
    )
)

network_racks, switches_per_rack, sw_cap_by_u, sw_cap_by_pwr = racks_needed_for_nodes(
    total_switches,
    rack_profile.switch_u,
    switch_power_kw,
    rack_profile,
    max_rack_power_kw,
)

total_racks = gpu_racks + storage_racks + mgmt_racks + network_racks

# ------------------------------
# Output: high-level summary
# ------------------------------

st.header("Datacenter fit summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total IT power (kW)",
        f"{total_it_power_kw:.1f}",
    )
    if total_it_power_kw <= max_it_power_kw:
        st.success(
            f"Within datacenter IT power limit (≤ {max_it_power_kw:.1f} kW)."
        )
    else:
        st.error(
            f"Exceeds datacenter IT power limit by "
            f"{total_it_power_kw - max_it_power_kw:.1f} kW."
        )

with col2:
    st.metric(
        "Total racks required",
        f"{total_racks}",
    )
    if total_racks <= max_racks:
        st.success(
            f"Within datacenter rack limit (≤ {max_racks} racks)."
        )
    else:
        st.error(
            f"Exceeds datacenter rack limit by {total_racks - max_racks} racks."
        )

with col3:
    st.metric(
        "Network switches (all networks)",
        f"{total_switches}",
    )
    st.caption("Sum of leaf, spine, and core switches across all networks.")

st.subheader("Rack allocation by function (space and power constrained)")
st.write(
    f"- GPU racks: **{gpu_racks}** "
    f"(nodes per rack: {gpu_nodes_per_rack} | U-cap: {gpu_cap_by_u}, "
    f"power-cap: {gpu_cap_by_pwr})\n"
    f"- Storage racks: **{storage_racks}** "
    f"(nodes per rack: {storage_nodes_per_rack} | U-cap: {storage_cap_by_u}, "
    f"power-cap: {storage_cap_by_pwr})\n"
    f"- Management racks: **{mgmt_racks}** "
    f"(nodes per rack: {mgmt_nodes_per_rack} | U-cap: {mgmt_cap_by_u}, "
    f"power-cap: {mgmt_cap_by_pwr})\n"
    f"- Network racks: **{network_racks}** "
    f"(switches per rack: {switches_per_rack} | U-cap: {sw_cap_by_u}, "
    f"power-cap: {sw_cap_by_pwr})\n"
)

# Warnings if a node type does not fit within per-rack power
if gpu_nodes_per_rack == 0 and num_gpu_nodes > 0:
    st.warning(
        "GPU node power exceeds the maximum rack power or rack U capacity; "
        "no valid packing exists for GPU racks with current constraints."
    )
if storage_nodes_per_rack == 0 and num_storage_nodes > 0:
    st.warning(
        "Storage node power exceeds the maximum rack power or rack U capacity; "
        "no valid packing exists for storage racks with current constraints."
    )
if mgmt_nodes_per_rack == 0 and num_mgmt_nodes > 0:
    st.warning(
        "Management node power exceeds the maximum rack power or rack U capacity; "
        "no valid packing exists for management racks with current constraints."
    )
if switches_per_rack == 0 and total_switches > 0:
    st.warning(
        "Switch power exceeds the maximum rack power or rack U capacity; "
        "no valid packing exists for network racks with current constraints."
    )

st.subheader("Power breakdown (kW)")
st.write(
    f"- GPU nodes (including GPUs, CPUs, NICs, etc.): "
    f"**{total_gpu_power_kw:.1f} kW** "
    f"({num_gpu_nodes} nodes @ {gpu_node_power_kw:.1f} kW each)\n"
    f"- Storage nodes: **{total_storage_power_kw:.1f} kW** "
    f"({num_storage_nodes} nodes @ {storage_node_power_kw:.1f} kW each)\n"
    f"- Management nodes: **{total_mgmt_power_kw:.1f} kW** "
    f"({num_mgmt_nodes} nodes @ {mgmt_node_power_kw:.1f} kW each)\n"
    f"- Network switches (incl. switch-side NICs/ports): "
    f"**{total_switch_power_kw:.1f} kW** "
    f"({total_switches} switches @ {switch_power_kw:.1f} kW each)\n"
    f"- **Total IT load:** **{total_it_power_kw:.1f} kW** "
    f"(compared to datacenter limit {max_it_power_kw:.1f} kW)\n"
)

# ------------------------------
# Network details
# ------------------------------

def render_network_section(design: NetworkDesignResult, server_label: str):
    st.markdown(f"### {design.name} network")

    cols = st.columns(4)
    with cols[0]:
        st.metric("Topology", design.topology)
    with cols[1]:
        st.metric("Tiers", design.tiers)
    with cols[2]:
        st.metric("Endpoints (nodes)", design.num_endpoints)
    with cols[3]:
        st.metric("Total server ports", design.total_server_ports)

    st.write(
        f"**Oversubscription:** {design.oversub_ratio:.2f}:1 "
        f"(server:uplink bandwidth, approximated via ports)"
    )

    st.markdown("**Switch counts by tier**")
    st.write(
        f"- Leaf switches: **{design.num_leaf}** "
        f"({design.leaf_ports} ports per leaf)\n"
        f"- Spine switches: **{design.num_spine}** "
        f"({design.spine_ports if design.spine_ports else '-'} ports per spine)\n"
        f"- Core switches: **{design.num_core}** "
        f"({design.core_ports if design.core_ports else '-'} ports per core)\n"
    )

    st.markdown("**Simplified topology diagram**")
    graph = build_network_graph(design, server_label)
    st.graphviz_chart(graph)


st.header("Network designs")

col_back, col_front, col_mgmt = st.tabs(
    ["Backside (GPU fabric)", "Frontside (GPU ↔ storage)", "Management"]
)

with col_back:
    render_network_section(
        backside_design,
        server_label="GPU nodes (backside fabric)",
    )

with col_front:
    render_network_section(
        frontside_design,
        server_label="GPU + storage nodes (frontside)",
    )

with col_mgmt:
    render_network_section(
        mgmt_design,
        server_label="GPU + storage + mgmt nodes (OOB management)",
    )

st.info(
    "Note: The switch counts, rack allocations, and diagrams are **sizing approximations** based on "
    "port counts, oversubscription ratios, rack U, and per-rack power limits. They are intended as "
    "a planning aid, not as a fully constrained physical design."
)
