"""
Neural Network and Growth Visualization for Arbor
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import networkx as nx

class NetworkVisualizer:
    """Visualizes the Arbor neural network architecture."""
    
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def visualize_architecture(self, 
                             num_layers: int,
                             hidden_dim: int,
                             num_heads: int,
                             ffn_dims: List[int],
                             layer_utilization: Optional[Dict[int, float]] = None) -> plt.Figure:
        """Create a visual representation of the Arbor architecture."""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, num_layers + 2)
        ax.set_aspect('equal')
        
        # Colors for different components
        colors = {
            'embedding': '#FFE5B4',
            'attention': '#B4D4FF', 
            'ffn': '#B4FFB4',
            'output': '#FFB4B4',
            'growth': '#FFD700'
        }
        
        # Draw input embedding
        self._draw_layer(ax, 0.5, 0.5, 2, 0.3, "Input\nEmbedding", colors['embedding'])
        
        # Draw transformer layers
        for i in range(num_layers):
            y_pos = i + 1.5
            
            # Color based on utilization if available
            if layer_utilization and i in layer_utilization:
                util = layer_utilization[i]
                if util > 0.9:
                    color = colors['growth']  # High utilization
                elif util > 0.7:
                    color = colors['ffn']
                else:
                    color = colors['attention']
            else:
                color = colors['attention']
            
            # Multi-head attention
            self._draw_layer(ax, 2, y_pos, 1.5, 0.25, f"MHA-{num_heads}", color)
            
            # FFN layer with current dimension
            ffn_dim = ffn_dims[i] if i < len(ffn_dims) else 4096
            ffn_label = f"FFN\n{ffn_dim}"
            if ffn_dim > 4096:
                ffn_label += "\n(Grown)"
                ffn_color = colors['growth']
            else:
                ffn_color = colors['ffn']
                
            self._draw_layer(ax, 4.5, y_pos, 1.5, 0.25, ffn_label, ffn_color)
            
            # Layer number
            ax.text(0.2, y_pos, f"L{i}", fontsize=10, ha='center', va='center')
            
            # Draw connections
            if i > 0:
                ax.arrow(2.75, y_pos - 0.5, 0, 0.25, head_width=0.05, head_length=0.05, fc='gray', ec='gray')
        
        # Draw output layer
        self._draw_layer(ax, 7, num_layers + 1, 2, 0.3, "Output\nProjection", colors['output'])
        
        # Add growth indicators
        if layer_utilization:
            high_util_layers = [i for i, util in layer_utilization.items() if util > 0.9]
            if high_util_layers:
                ax.text(8.5, num_layers + 0.5, f"High Utilization:\nLayers {high_util_layers}", 
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['growth']))
        
        # Labels and title
        ax.set_title(f"Arbor Architecture\n{num_layers} Layers | {hidden_dim}D | {sum(ffn_dims)/len(ffn_dims):.0f} Avg FFN", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Model Components")
        ax.set_ylabel("Layer Index")
        ax.grid(True, alpha=0.3)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks(range(num_layers + 2))
        
        # Legend
        legend_elements = [
            patches.Patch(color=colors['attention'], label='Multi-Head Attention'),
            patches.Patch(color=colors['ffn'], label='FFN (Normal)'),
            patches.Patch(color=colors['growth'], label='Grown/High Utilization'),
            patches.Patch(color=colors['embedding'], label='Embedding'),
            patches.Patch(color=colors['output'], label='Output')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def _draw_layer(self, ax, x, y, width, height, label, color):
        """Draw a single layer box."""
        rect = patches.Rectangle((x, y), width, height, 
                               linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, label, fontsize=9, ha='center', va='center', weight='bold')

class GrowthVisualizer:
    """Visualizes model growth patterns and statistics."""
    
    def create_growth_dashboard(self, metrics_data: Dict) -> go.Figure:
        """Create a comprehensive growth visualization dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Loss', 'Model Parameters Growth', 
                          'Layer Count Growth', 'Layer Utilization',
                          'Performance Metrics', 'Growth Events Timeline'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # 1. Training Loss
        if 'loss_trends' in metrics_data:
            loss_data = metrics_data['loss_trends']
            fig.add_trace(
                go.Scatter(x=loss_data['steps'], y=loss_data['train_loss'],
                          name='Train Loss', line=dict(color='blue')),
                row=1, col=1
            )
            if loss_data['val_loss']:
                fig.add_trace(
                    go.Scatter(x=loss_data['val_steps'], y=loss_data['val_loss'],
                              name='Val Loss', line=dict(color='red')),
                    row=1, col=1
                )
        
        # 2. Parameters Growth
        if 'growth_timeline' in metrics_data:
            growth_data = metrics_data['growth_timeline']
            fig.add_trace(
                go.Scatter(x=growth_data['steps'], y=growth_data['parameters_m'],
                          name='Parameters (M)', line=dict(color='green')),
                row=1, col=2
            )
        
        # 3. Layer Count Growth
        if 'growth_timeline' in metrics_data:
            fig.add_trace(
                go.Scatter(x=growth_data['steps'], y=growth_data['num_layers'],
                          name='Layer Count', line=dict(color='purple')),
                row=2, col=1
            )
        
        # 4. Layer Utilization
        if 'utilization_trends' in metrics_data:
            util_data = metrics_data['utilization_trends']
            fig.add_trace(
                go.Scatter(x=util_data['steps'], y=util_data['avg_utilization'],
                          name='Avg Utilization', line=dict(color='orange')),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=util_data['steps'], y=util_data['max_utilization'],
                          name='Max Utilization', line=dict(color='red', dash='dash')),
                row=2, col=2
            )
            # Add growth threshold line
            if util_data['steps']:
                fig.add_hline(y=0.92, line_dash="dot", line_color="red",
                             annotation_text="Growth Threshold", row=2, col=2)
        
        # 5. Performance Metrics
        if 'performance_stats' in metrics_data:
            perf_data = metrics_data['performance_stats']
            # This would need time series data, showing current values for now
            fig.add_trace(
                go.Bar(x=['Tokens/sec', 'Memory (GB)', 'GPU %'],
                      y=[perf_data.get('avg_tokens_per_sec', 0),
                         perf_data.get('avg_memory_usage', 0),
                         perf_data.get('avg_gpu_utilization', 0)],
                      name='Performance'),
                row=3, col=1
            )
        
        # 6. Growth Events Timeline
        if 'growth_events' in metrics_data:
            events = metrics_data['growth_events']
            if events:
                event_steps = [e['step'] for e in events]
                event_types = [e.get('type', 'growth') for e in events]
                event_details = [e.get('message', f"Growth at step {e['step']}") for e in events]
                
                fig.add_trace(
                    go.Scatter(x=event_steps, y=[1]*len(event_steps),
                              mode='markers+text',
                              marker=dict(size=15, color='gold'),
                              text=event_types,
                              textposition="top center",
                              name='Growth Events',
                              hovertext=event_details),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Arbor Training Dashboard - Real-time Monitoring",
            height=900,
            showlegend=True,
            font=dict(size=10)
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Training Steps", row=3, col=1)
        fig.update_xaxes(title_text="Training Steps", row=3, col=2)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Parameters (M)", row=1, col=2)
        fig.update_yaxes(title_text="Layer Count", row=2, col=1)
        fig.update_yaxes(title_text="Utilization", row=2, col=2)
        
        return fig
    
    def create_layer_utilization_heatmap(self, layer_utilization_data: Dict) -> go.Figure:
        """Create a heatmap showing layer utilization over time."""
        
        if not layer_utilization_data:
            return go.Figure()
        
        # Prepare data for heatmap
        all_layers = sorted(layer_utilization_data.keys())
        all_steps = []
        utilization_matrix = []
        
        # Get all unique steps
        for layer_data in layer_utilization_data.values():
            all_steps.extend(layer_data['steps'])
        all_steps = sorted(list(set(all_steps)))
        
        # Create utilization matrix
        for layer_idx in all_layers:
            layer_data = layer_utilization_data[layer_idx]
            layer_utils = []
            
            for step in all_steps:
                if step in layer_data['steps']:
                    step_idx = layer_data['steps'].index(step)
                    layer_utils.append(layer_data['utilization'][step_idx])
                else:
                    layer_utils.append(0)  # No data for this step
            
            utilization_matrix.append(layer_utils)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=utilization_matrix,
            x=all_steps,
            y=[f"Layer {i}" for i in all_layers],
            colorscale='RdYlBu_r',
            zmin=0,
            zmax=1,
            colorbar=dict(title="Utilization")
        ))
        
        fig.update_layout(
            title="Layer Utilization Heatmap Over Training",
            xaxis_title="Training Steps",
            yaxis_title="Layer Index",
            height=600
        )
        
        return fig
    
    def create_architecture_comparison(self, initial_config: Dict, current_config: Dict) -> go.Figure:
        """Compare initial vs current architecture."""
        
        metrics = ['Layers', 'Parameters (M)', 'FFN Dim', 'Context Length']
        
        initial_values = [
            initial_config.get('num_layers', 24),
            initial_config.get('parameters', 699) / 1e6,
            initial_config.get('ffn_dim', 4096),
            initial_config.get('max_seq_length', 1024)
        ]
        
        current_values = [
            current_config.get('num_layers', 24),
            current_config.get('parameters', 699) / 1e6,
            current_config.get('ffn_dim', 4096),
            current_config.get('max_seq_length', 1024)
        ]
        
        fig = go.Figure(data=[
            go.Bar(name='Initial', x=metrics, y=initial_values, marker_color='lightblue'),
            go.Bar(name='Current', x=metrics, y=current_values, marker_color='darkblue')
        ])
        
        fig.update_layout(
            title="Architecture Evolution: Initial vs Current",
            barmode='group',
            height=400
        )
        
        return fig
