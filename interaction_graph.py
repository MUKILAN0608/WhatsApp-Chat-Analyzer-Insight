import networkx as nx
import pandas as pd
import plotly.graph_objects as go

def get_interaction_graph(df):
    # Filter out notifications and sort by datetime
    df = df[df['user'] != 'notification'].copy()
    df = df.sort_values('datetime')
    df['next_user'] = df['user'].shift(-1)
    df['next_time'] = df['datetime'].shift(-1)

    # Filter rows where the next user is different
    df = df[df['user'] != df['next_user']]
    edges = df.groupby(['user', 'next_user']).size().reset_index(name='count')

    # Create a directed graph
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row['user'], row['next_user'], weight=row['count'])

    # Generate spring layout for positioning
    pos = nx.spring_layout(G)

    edge_x, edge_y, node_x, node_y = [], [], [], []

    # Process edges
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Process nodes
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Create the figure
    fig = go.Figure()

    # Add edge traces
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add node traces
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=list(G.nodes),
        textposition="bottom center"
    ))

    # Update layout with the corrected title format
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        title=dict(
            text="User Interaction Network",
            font=dict(
                size=16
            )
        )
    )

    # Compute node stats (like degree)
    node_stats = dict(G.degree())

    return G, node_stats, fig
