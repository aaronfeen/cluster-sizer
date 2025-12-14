leaf_down = tiers["leaf_downlink_ports"]
leaf_total = tiers["leaf_total_ports"]
spine_total = tiers["spine_total_ports"]
core_total = tiers["core_total_ports"] if tiers.get("enabled_core") else 0

if leaf_down <= 0 or leaf_total <= 0 or spine_total <= 0:
    results["warnings"].append("Invalid switch port configuration: ports must be > 0.")
    return results

uplinks_needed_per_leaf = math.ceil(leaf_down / max(oversub_leaf, 1e-6))
available_uplinks_per_leaf = max(leaf_total - leaf_down, 0)

if uplinks_needed_per_leaf > available_uplinks_per_leaf:
    results["warnings"].append(
        f"Leaf uplinks required ({uplinks_needed_per_leaf}) exceed available uplink ports per leaf ({available_uplinks_per_leaf})."
    )

results["leaf_count"] = ceildiv(total_server_ports, leaf_down)

total_leaf_uplinks = results["leaf_count"] * min(uplinks_needed_per_leaf, available_uplinks_per_leaf)

results["spine_count"] = ceildiv(total_leaf_uplinks, spine_total) if spine_total > 0 else 0

if tiers.get("enabled_core", False):
    total_spine_uplinks_to_core = math.ceil(total_leaf_uplinks / max(oversub_spine_to_core, 1e-6))
    if core_total <= 0:
        results["warnings"].append("Core tier enabled but core switch has 0 ports configured.")
        results["core_count"] = 0
    else:
        results["core_count"] = ceildiv(total_spine_uplinks_to_core, core_total)
else:
    results["core_count"] = 0

results["notes"].append(
    f"Leaf downlinks per leaf: {leaf_down}; uplinks per leaf (oversub): {uplinks_needed_per_leaf}; available uplinks per leaf: {available_uplinks_per_leaf}"
)
results["notes"].append(
    f"Total leaf uplinks to spine: {total_leaf_uplinks}; spine ports per switch: {spine_total}"
)
if tiers.get("enabled_core", False):
    results["notes"].append(
        f"Total spine uplinks to core (post-oversub): {math.ceil(total_leaf_uplinks / max(oversub_spine_to_core, 1e-6))}; core ports per switch: {core_total}"
    )

return results

# Create layer subgraphs with rank=same
core_nodes = []
spine_nodes = []
leaf_nodes = []
server_nodes = []

# Servers layer
with g.subgraph(name=f"cluster_{network_name}_servers") as c:
    c.attr(rank="same", label=f"{network_name} servers", style="rounded", color="lightgrey")
    for label, cnt in server_groups.items():
        node_id = f"{network_name}_{label}"
        c.node(node_id, f"{label}\nports:{cnt}", shape="box")
        server_nodes.append(node_id)

# Leaf layer
with g.subgraph(name=f"cluster_{network_name}_leaf") as c:
    c.attr(rank="same", label="Leaf", color="lightblue", style="rounded")
    for i in range(plan["leaf_count"]):
        nid = f"{network_name}_Leaf_{i+1}"
        c.node(nid, f"Leaf {i+1}\n{tiers['leaf_total_ports']} ports", shape="ellipse")
        leaf_nodes.append(nid)

# Spine layer
with g.subgraph(name=f"cluster_{network_name}_spine") as c:
    c.attr(rank="same", label="Spine", color="orange", style="rounded")
    for i in range(plan["spine_count"]):
        nid = f"{network_name}_Spine_{i+1}"
        c.node(nid, f"Spine {i+1}\n{tiers['spine_total_ports']} ports", shape="ellipse")
        spine_nodes.append(nid)

# Core layer
if tiers.get("enabled_core", False):
    with g.subgraph(name=f"cluster_{network_name}_core") as c:
        c.attr(rank="same", label="Core", color="red", style="rounded")
        for i in range(plan["core_count"]):
            nid = f"{network_name}_Core_{i+1}"
            c.node(nid, f"Core {i+1}\n{tiers['core_total_ports']} ports", shape="ellipse")
            core_nodes.append(nid)

# Connect servers to leaves (always)
for s in server_nodes:
    # For readability, connect each server group to all leaves
    for l in leaf_nodes:
        g.edge(s, l)

# Topology-based connections between tiers
# Fat tree: fully mesh leaves->spines and spines->cores
# Dragonfly+: assign leaves to a spine (group), spines fully mesh to cores
# Rail: assign leaves to spines round-robin, spines to cores round-robin; optionally chain leaves and spines

if topology == "Fat tree":
    # Leaves to spines
    for l in leaf_nodes:
        for s in spine_nodes:
            g.edge(l, s)
    # Spines to cores
    for s in spine_nodes:
        for c in core_nodes:
            g.edge(s, c)

elif topology == "Dragonfly+":
    # Leaves grouped to spines (each leaf connects to one spine, round-robin)
    for idx, l in enumerate(leaf_nodes):
        if spine_nodes:
            s = spine_nodes[idx % len(spine_nodes)]
            g.edge(l, s)
    # Spines fully mesh to cores
    for s in spine_nodes:
        for c in core_nodes:
            g.edge(s, c)

elif topology == "Rail":
    # Leaves to spines: round-robin mapping
    for idx, l in enumerate(leaf_nodes):
        if spine_nodes:
            s = spine_nodes[idx % len(spine_nodes)]
            g.edge(l, s)
    # Optionally chain spines (visual rail), then spines to cores round-robin
    for i in range(len(spine_nodes) - 1):
        g.edge(spine_nodes[i], spine_nodes[i + 1])
    for idx, s in enumerate(spine_nodes):
        if core_nodes:
            c = core_nodes[idx % len(core_nodes)]
            g.edge(s, c)

return g

st.sidebar.header("Workload and Facility Inputs")
gpu_servers = st.sidebar.number_input("GPU servers count", min_value=0, value=64, step=1)
storage_servers = st.sidebar.number_input("Storage servers count", min_value=0, value=16, step=1)
mgmt_servers = st.sidebar.number_input("Management servers count", min_value=0, value=8, step=1)

rack_u = st.sidebar.number_input("Rack size (U)", min_value=1, value=42, step=1)
dc_power_kw_max = st.sidebar.number_input("Datacenter IT power maximum (kW)", min_value=0.0, value=500.0, step=10.0)
dc_rack_max = st.sidebar.number_input("Datacenter rack count maximum", min_value=0, value=20, step=1)

st.sidebar.header("Per-Node and Per-GPU NIC Configuration")
gpus_per_gpu_server = st.sidebar.number_input("GPUs per GPU server", min_value=1, value=8, step=1)
backside_nics_per_gpu = st.sidebar.selectbox("Backside NICs per GPU", options=[1, 4, 8], index=1)
gpu_front_nics_per_node = st.sidebar.selectbox("Frontside NICs per GPU server", options=[1, 2], index=1)
storage_front_nics_per_node = st.sidebar.selectbox("Frontside NICs per storage server", options=[1, 2], index=0)
mgmt_nics_per_node = 1  # As specified

st.sidebar.header("Server Specs")
gpu_ru = st.sidebar.number_input("GPU server RU", min_value=1, value=6, step=1)
gpu_power_kw = st.sidebar.number_input("GPU server power (kW)", min_value=0.0, value=3.5, step=0.1)

storage_ru = st.sidebar.number_input("Storage server RU", min_value=1, value=4, step=1)
storage_power_kw = st.sidebar.number_input("Storage server power (kW)", min_value=0.0, value=0.8, step=0.1)

mgmt_ru = st.sidebar.number_input("Management server RU", min_value=1, value=1, step=1)
mgmt_power_kw = st.sidebar.number_input("Management server power (kW)", min_value=0.0, value=0.3, step=0.1)

st.sidebar.header("Switch Specs (per network and tier)")
# Management network specs
st.sidebar.subheader("Management network")
mgmt_topology = st.sidebar.selectbox("Management topology", options=["Fat tree", "Dragonfly+", "Rail"], index=0)
mgmt_enabled_core = st.sidebar.checkbox("Management network has core tier", value=False)
mgmt_leaf_total_ports = st.sidebar.number_input("Mgmt leaf total ports", min_value=1, value=48, step=1)
mgmt_leaf_downlink_ports = st.sidebar.number_input("Mgmt leaf downlink (server) ports per leaf", min_value=1, value=44, step=1)
mgmt_spine_total_ports = st.sidebar.number_input("Mgmt spine total ports", min_value=1, value=48, step=1)
mgmt_core_total_ports = st.sidebar.number_input("Mgmt core total ports", min_value=1, value=64, step=1)
mgmt_leaf_oversub = st.sidebar.number_input("Mgmt leaf oversubscription (down:up ratio)", min_value=1.0, value=4.0, step=0.5)
mgmt_spine_core_oversub = st.sidebar.number_input("Mgmt spine->core oversubscription ratio", min_value=1.0, value=2.0, step=0.5)
mgmt_leaf_ru = st.sidebar.number_input("Mgmt leaf switch RU", min_value=1, value=1, step=1)
mgmt_spine_ru = st.sidebar.number_input("Mgmt spine switch RU", min_value=1, value=2, step=1)
mgmt_core_ru = st.sidebar.number_input("Mgmt core switch RU", min_value=1, value=2, step=1)
mgmt_leaf_power_kw = st.sidebar.number_input("Mgmt leaf switch power (kW)", min_value=0.0, value=0.3, step=0.1)
mgmt_spine_power_kw = st.sidebar.number_input("Mgmt spine switch power (kW)", min_value=0.0, value=0.5, step=0.1)
mgmt_core_power_kw = st.sidebar.number_input("Mgmt core switch power (kW)", min_value=0.0, value=0.7, step=0.1)

# Frontside network specs
st.sidebar.subheader("Frontside network (GPU to Storage)")
front_topology = st.sidebar.selectbox("Frontside topology", options=["Fat tree", "Dragonfly+", "Rail"], index=0)
front_enabled_core = st.sidebar.checkbox("Frontside network has core tier", value=True)
front_leaf_total_ports = st.sidebar.number_input("Front leaf total ports", min_value=1, value=64, step=1)
front_leaf_downlink_ports = st.sidebar.number_input("Front leaf downlink (server) ports per leaf", min_value=1, value=48, step=1)
front_spine_total_ports = st.sidebar.number_input("Front spine total ports", min_value=1, value=64, step=1)
front_core_total_ports = st.sidebar.number_input("Front core total ports", min_value=1, value=128, step=1)
front_leaf_oversub = st.sidebar.number_input("Front leaf oversubscription (down:up ratio)", min_value=1.0, value=3.0, step=0.5)
front_spine_core_oversub = st.sidebar.number_input("Front spine->core oversubscription ratio", min_value=1.0, value=2.0, step=0.5)
front_leaf_ru = st.sidebar.number_input("Front leaf switch RU", min_value=1, value=2, step=1)
front_spine_ru = st.sidebar.number_input("Front spine switch RU", min_value=1, value=2, step=1)
front_core_ru = st.sidebar.number_input("Front core switch RU", min_value=1, value=4, step=1)
front_leaf_power_kw = st.sidebar.number_input("Front leaf switch power (kW)", min_value=0.0, value=0.8, step=0.1)
front_spine_power_kw = st.sidebar.number_input("Front spine switch power (kW)", min_value=0.0, value=1.2, step=0.1)
front_core_power_kw = st.sidebar.number_input("Front core switch power (kW)", min_value=0.0, value=1.5, step=0.1)

# Backside network specs
st.sidebar.subheader("Backside network (GPU fabric)")
back_topology = st.sidebar.selectbox("Backside topology", options=["Fat tree", "Dragonfly+", "Rail"], index=1)
back_enabled_core = st.sidebar.checkbox("Backside network has core tier", value=False)
back_leaf_total_ports = st.sidebar.number_input("Back leaf total ports", min_value=1, value=64, step=1)
back_leaf_downlink_ports = st.sidebar.number_input("Back leaf downlink (GPU) ports per leaf", min_value=1, value=48, step=1)
back_spine_total_ports = st.sidebar.number_input("Back spine total ports", min_value=1, value=128, step=1)
back_core_total_ports = st.sidebar.number_input("Back core total ports", min_value=1, value=128, step=1)
back_leaf_oversub = st.sidebar.number_input("Back leaf oversubscription (down:up ratio)", min_value=1.0, value=1.5, step=0.5)
back_spine_core_oversub = st.sidebar.number_input("Back spine->core oversubscription ratio", min_value=1.0, value=1.5, step=0.5)
back_leaf_ru = st.sidebar.number_input("Back leaf switch RU", min_value=1, value=2, step=1)
back_spine_ru = st.sidebar.number_input("Back spine switch RU", min_value=1, value=4, step=1)
back_core_ru = st.sidebar.number_input("Back core switch RU", min_value=1, value=4, step=1)
back_leaf_power_kw = st.sidebar.number_input("Back leaf switch power (kW)", min_value=0.0, value=1.5, step=0.1)
back_spine_power_kw = st.sidebar.number_input("Back spine switch power (kW)", min_value=0.0, value=2.0, step=0.1)
back_core_power_kw = st.sidebar.number_input("Back core switch power (kW)", min_value=0.0, value=2.5, step=0.1)

st.sidebar.header("Preferred Oversubscription Guidance")
st.sidebar.write("Oversubscription ratios are specified per network above.")

# Compute total NIC ports per network based on selections
total_back_ports = gpu_servers * gpus_per_gpu_server * backside_nics_per_gpu
total_front_ports = gpu_servers * gpu_front_nics_per_node + storage_servers * storage_front_nics_per_node
total_mgmt_ports = (gpu_servers + storage_servers + mgmt_servers) * mgmt_nics_per_node

# Build plans for each network
mgmt_tiers = {
    "enabled_core": mgmt_enabled_core,
    "leaf_downlink_ports": mgmt_leaf_downlink_ports,
    "leaf_total_ports": mgmt_leaf_total_ports,
    "spine_total_ports": mgmt_spine_total_ports,
    "core_total_ports": mgmt_core_total_ports,
}
front_tiers = {
    "enabled_core": front_enabled_core,
    "leaf_downlink_ports": front_leaf_downlink_ports,
    "leaf_total_ports": front_leaf_total_ports,
    "spine_total_ports": front_spine_total_ports,
    "core_total_ports": front_core_total_ports,
}
back_tiers = {
    "enabled_core": back_enabled_core,
    "leaf_downlink_ports": back_leaf_downlink_ports,
    "leaf_total_ports": back_leaf_total_ports,
    "spine_total_ports": back_spine_total_ports,
    "core_total_ports": back_core_total_ports,
}

mgmt_plan = build_network_plan(total_mgmt_ports, mgmt_tiers, mgmt_leaf_oversub, mgmt_spine_core_oversub)
front_plan = build_network_plan(total_front_ports, front_tiers, front_leaf_oversub, front_spine_core_oversub)
back_plan = build_network_plan(total_back_ports, back_tiers, back_leaf_oversub, back_spine_core_oversub)

# RU totals for network gear
mgmt_net_total_ru = mgmt_plan["leaf_count"] * mgmt_leaf_ru + mgmt_plan["spine_count"] * mgmt_spine_ru + mgmt_plan["core_count"] * (mgmt_core_ru if mgmt_enabled_core else 0)
front_net_total_ru = front_plan["leaf_count"] * front_leaf_ru + front_plan["spine_count"] * front_spine_ru + front_plan["core_count"] * (front_core_ru if front_enabled_core else 0)
back_net_total_ru = back_plan["leaf_count"] * back_leaf_ru + back_plan["spine_count"] * back_spine_ru + back_plan["core_count"] * (back_core_ru if back_enabled_core else 0)

rack_devices = [
    ("GPU servers (separate racks)", gpu_servers, gpu_ru),
    ("Storage servers (separate racks)", storage_servers, storage_ru),
    ("Management servers (separate racks)", mgmt_servers, mgmt_ru),
    ("Mgmt network equipment", 1, mgmt_net_total_ru),
    ("Front network equipment", 1, front_net_total_ru),
    ("Back network equipment", 1, back_net_total_ru),
]
rack_plan = plan_racks(rack_devices, rack_u)

# Power plan
power_devices = [
    ("GPU servers", gpu_servers, gpu_power_kw),
    ("Storage servers", storage_servers, storage_power_kw),
    ("Management servers", mgmt_servers, mgmt_power_kw),
    ("Mgmt leaf switches", mgmt_plan["leaf_count"], mgmt_leaf_power_kw),
    ("Mgmt spine switches", mgmt_plan["spine_count"], mgmt_spine_power_kw),
    ("Mgmt core switches", mgmt_plan["core_count"], mgmt_core_power_kw if mgmt_enabled_core else 0.0),
    ("Front leaf switches", front_plan["leaf_count"], front_leaf_power_kw),
    ("Front spine switches", front_plan["spine_count"], front_spine_power_kw),
    ("Front core switches", front_plan["core_count"], front_core_power_kw if front_enabled_core else 0.0),
    ("Back leaf switches", back_plan["leaf_count"], back_leaf_power_kw),
    ("Back spine switches", back_plan["spine_count"], back_spine_power_kw),
    ("Back core switches", back_plan["core_count"], back_core_power_kw if back_enabled_core else 0.0),
]
total_kw = total_power(power_devices)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Capacity Summary")
    st.write(f"Total power required (kW): {total_kw:.2f}")
    if total_kw <= dc_power_kw_max:
        st.success(f"Within datacenter power limit ({dc_power_kw_max} kW).")
    else:
        st.error(f"Exceeds datacenter power limit ({dc_power_kw_max} kW).")

    st.write(f"Total racks required: {rack_plan['total_racks']}")
    if rack_plan["total_racks"] <= dc_rack_max:
        st.success(f"Within datacenter rack count limit ({dc_rack_max}).")
    else:
        st.error(f"Exceeds datacenter rack count limit ({dc_rack_max}).")

    st.write(f"Total RU used across all groups: {rack_plan['total_ru']}")

with col2:
    st.subheader("Rack Group Breakdown")
    for category, data in rack_plan["groups"].items():
        st.write(f"{category}: devices={data['count']}, RU each={data['ru_each']}, total RU={data['total_ru']}, racks={data['racks']}")

st.subheader("Network Switch Counts")
st.write("Management network:")
st.write(f"Topology: {mgmt_topology}")
st.write(f"Leaf: {mgmt_plan['leaf_count']}, Spine: {mgmt_plan['spine_count']}, Core: {mgmt_plan['core_count']}")
for w in mgmt_plan["warnings"]:
    st.warning(f"Mgmt: {w}")
for n in mgmt_plan["notes"]:
    st.write(f"Mgmt note: {n}")

st.write("Frontside network (GPU to Storage):")
st.write(f"Topology: {front_topology}")
st.write(f"Leaf: {front_plan['leaf_count']}, Spine: {front_plan['spine_count']}, Core: {front_plan['core_count']}")
for w in front_plan["warnings"]:
    st.warning(f"Front: {w}")
for n in front_plan["notes"]:
    st.write(f"Front note: {n}")

st.write("Backside network (GPU fabric):")
st.write(f"Topology: {back_topology}")
st.write(f"Leaf: {back_plan['leaf_count']}, Spine: {back_plan['spine_count']}, Core: {back_plan['core_count']}")
for w in back_plan["warnings"]:
    st.warning(f"Back: {w}")
for n in back_plan["notes"]:
    st.write(f"Back note: {n}")

st.subheader("Simplified Network Diagrams (Layered)")
mgmt_servers_ports = {
    "GPU mgmt NICs": gpu_servers * mgmt_nics_per_node,
    "Storage mgmt NICs": storage_servers * mgmt_nics_per_node,
    "Mgmt server NICs": mgmt_servers * mgmt_nics_per_node,
}
front_servers_ports = {
    "GPU front NICs": gpu_servers * gpu_front_nics_per_node,
    "Storage front NICs": storage_servers * storage_front_nics_per_node,
}
back_servers_ports = {
    "GPU fabric NICs": gpu_servers * gpus_per_gpu_server * backside_nics_per_gpu,
}

mgmt_graph = make_network_graph("Mgmt", mgmt_servers_ports, mgmt_plan, mgmt_tiers, mgmt_topology)
front_graph = make_network_graph("Front", front_servers_ports, front_plan, front_tiers, front_topology)
back_graph = make_network_graph("Back", back_servers_ports, back_plan, back_tiers, back_topology)

st.graphviz_chart(mgmt_graph)
st.graphviz_chart(front_graph)
st.graphviz_chart(back_graph)

st.subheader("Assumptions and Notes")
st.write("- Servers (GPU, storage, management) are placed in separate rack groups; network equipment is grouped per network type.")
st.write("- Backside ports are GPUs per server times backside NICs per GPU; frontside ports are NICs per node for GPU and storage servers; management is 1 NIC per node.")
st.write("- Leaf uplinks required are computed from downlink ports and oversubscription ratio; warnings appear if leaf ports cannot satisfy uplinks.")
st.write("- Switch counts are computed by terminating aggregate uplinks onto the next tier; spine-to-core oversubscription applies if a core tier is enabled.")
st.write("- Diagrams are layered top-to-bottom: core, spine, leaf, then servers. Connections vary by selected topology (Fat tree, Dragonfly+, Rail).")

