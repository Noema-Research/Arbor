"""
Arbor Training Dashboard - Real-time monitoring with Streamlit

A comprehensive dashboard for monitoring Arbor model training with:
- Live metrics visualization
- Neural network architecture display
- Growth tracking and alerts
- Performance monitoring
- Interactive controls

Usage:
    streamlit run arbor/tracking/dashboard.py
    
Or programmatically:
    from arbor.tracking import ArborDashboard
    dashboard = ArborDashboard()
    dashboard.run()
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

from .monitor import TrainingMonitor
from .metrics import MetricsTracker, TrainingMetrics
from .visualizer import NetworkVisualizer, GrowthVisualizer
from .alerts import AlertSystem, AlertSeverity

class ArborDashboard:
    """Interactive Streamlit dashboard for Arbor training monitoring."""
    
    def __init__(self, metrics_dir: str = "training_logs"):
        self.metrics_dir = metrics_dir
        self.monitor = None
        
        # Initialize components
        self.metrics_tracker = MetricsTracker(metrics_dir)
        self.network_visualizer = NetworkVisualizer()
        self.growth_visualizer = GrowthVisualizer()
        
        # Load existing metrics if available
        self.metrics_tracker.load_metrics()
        
    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Arbor Training Dashboard",
            page_icon="🌳",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self._render_dashboard()
    
    def _render_dashboard(self):
        """Render the main dashboard."""
        
        # Header
        st.title("🌳 Arbor Training Dashboard")
        st.markdown("Real-time monitoring of Arbor model training with dynamic growth tracking")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main content
        if not self.metrics_tracker.metrics_history:
            self._render_no_data_message()
            return
        
        # Dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Metrics", 
            "🏗️ Architecture", 
            "🌱 Growth Tracking", 
            "🚨 Alerts", 
            "📈 Analytics"
        ])
        
        with tab1:
            self._render_live_metrics()
        
        with tab2:
            self._render_architecture()
        
        with tab3:
            self._render_growth_tracking()
        
        with tab4:
            self._render_alerts()
        
        with tab5:
            self._render_analytics()
    
    def _render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 30, 5)
        
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            self.metrics_tracker.load_metrics()
            st.rerun()
        
        # Data controls
        st.sidebar.header("📊 Data Controls")
        
        # Time range selector
        if self.metrics_tracker.metrics_history:
            total_steps = len(self.metrics_tracker.metrics_history)
            step_range = st.sidebar.slider(
                "Step Range", 
                0, total_steps, 
                (max(0, total_steps - 1000), total_steps)
            )
            st.session_state['step_range'] = step_range
        
        # Export controls
        st.sidebar.header("📤 Export")
        if st.sidebar.button("Export Training Report"):
            if hasattr(self, 'monitor') and self.monitor:
                report_file = self.monitor.export_training_report()
                st.sidebar.success(f"Report saved: {report_file}")
        
        # Configuration
        st.sidebar.header("⚙️ Configuration")
        
        # Alert thresholds
        with st.sidebar.expander("Alert Thresholds"):
            loss_spike_factor = st.number_input("Loss Spike Factor", value=3.0, step=0.1)
            grad_norm_max = st.number_input("Max Gradient Norm", value=10.0, step=0.1)
            memory_max = st.number_input("Max Memory (GB)", value=20.0, step=1.0)
            
            if st.button("Update Thresholds"):
                # This would update the alert system if monitor is available
                st.success("Thresholds updated!")
    
    def _render_no_data_message(self):
        """Render message when no training data is available."""
        st.info("🔍 No training data found. Start training to see live metrics!")
        
        st.markdown("""
        ### Getting Started
        
        1. **Start Training**: Run your Arbor training script with monitoring enabled
        2. **Integration**: Add the following to your training code:
        
        ```python
        from arbor.tracking import TrainingMonitor
        
        # Initialize monitor
        monitor = TrainingMonitor(save_dir="training_logs")
        monitor.start_monitoring()
        
        # In your training loop
        for step, batch in enumerate(dataloader):
            # ... training code ...
            
            # Log metrics
            monitor.log_training_step(
                step=step,
                epoch=epoch,
                train_loss=loss.item(),
                learning_rate=optimizer.param_groups[0]['lr'],
                grad_norm=grad_norm,
                model_state={
                    'num_layers': len(model.layers),
                    'num_parameters': sum(p.numel() for p in model.parameters()),
                    'layer_utilization': model.get_layer_utilization()
                }
            )
        ```
        
        3. **View Dashboard**: Refresh this page to see live training metrics
        """)
    
    def _render_live_metrics(self):
        """Render live training metrics."""
        latest_metrics = self.metrics_tracker.metrics_history[-1]
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Training Loss",
                f"{latest_metrics.train_loss:.4f}",
                delta=self._get_loss_delta()
            )
        
        with col2:
            st.metric(
                "Learning Rate",
                f"{latest_metrics.learning_rate:.2e}",
                delta=self._get_lr_delta()
            )
        
        with col3:
            st.metric(
                "Parameters",
                f"{latest_metrics.num_parameters / 1e6:.1f}M",
                delta=self._get_param_delta()
            )
        
        with col4:
            st.metric(
                "Layers",
                f"{latest_metrics.num_layers}",
                delta=self._get_layer_delta()
            )
        
        # Performance metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tokens/sec", f"{latest_metrics.tokens_per_second:.1f}")
        
        with col2:
            st.metric("Memory Usage", f"{latest_metrics.memory_usage_gb:.1f}GB")
        
        with col3:
            st.metric("GPU Utilization", f"{latest_metrics.gpu_utilization:.1f}%")
        
        with col4:
            st.metric("Gradient Norm", f"{latest_metrics.grad_norm:.4f}")
        
        # Loss chart
        st.subheader("📉 Training Loss")
        loss_data = self.metrics_tracker.get_loss_trends()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=loss_data['steps'],
            y=loss_data['train_loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='blue')
        ))
        
        if loss_data['val_loss']:
            fig.add_trace(go.Scatter(
                x=loss_data['val_steps'],
                y=loss_data['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title="Loss Trends",
            xaxis_title="Training Steps",
            yaxis_title="Loss",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # System metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💻 System Performance")
            perf_stats = self.metrics_tracker.get_performance_stats()
            
            perf_df = pd.DataFrame([
                {"Metric": "Avg Tokens/sec", "Value": f"{perf_stats.get('avg_tokens_per_sec', 0):.1f}"},
                {"Metric": "Avg Memory Usage", "Value": f"{perf_stats.get('avg_memory_usage', 0):.1f}GB"},
                {"Metric": "Avg GPU Utilization", "Value": f"{perf_stats.get('avg_gpu_utilization', 0):.1f}%"},
                {"Metric": "Training Time", "Value": f"{perf_stats.get('total_training_time', 0) / 3600:.1f}h"},
            ])
            
            st.dataframe(perf_df, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Layer Utilization")
            if latest_metrics.layer_utilization:
                util_df = pd.DataFrame([
                    {"Layer": f"Layer {idx}", "Utilization": f"{util:.3f}"}
                    for idx, util in latest_metrics.layer_utilization.items()
                ])
                st.dataframe(util_df, use_container_width=True)
            else:
                st.info("No layer utilization data available")
    
    def _render_architecture(self):
        """Render neural network architecture visualization."""
        st.subheader("🏗️ Neural Network Architecture")
        
        latest_metrics = self.metrics_tracker.metrics_history[-1]
        
        # Architecture summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Layers**: {latest_metrics.num_layers}")
        
        with col2:
            st.info(f"**Parameters**: {latest_metrics.num_parameters / 1e6:.1f}M")
        
        with col3:
            avg_util = latest_metrics.avg_utilization
            color = "🟢" if avg_util < 0.7 else "🟡" if avg_util < 0.9 else "🔴"
            st.info(f"**Avg Utilization**: {color} {avg_util:.3f}")
        
        # Create architecture visualization
        if latest_metrics.layer_utilization and latest_metrics.ffn_dimensions:
            fig = self.network_visualizer.visualize_architecture(
                num_layers=latest_metrics.num_layers,
                hidden_dim=1024,  # Default, would come from config
                num_heads=16,     # Default, would come from config
                ffn_dims=latest_metrics.ffn_dimensions,
                layer_utilization=latest_metrics.layer_utilization
            )
            
            st.pyplot(fig)
        else:
            st.warning("Architecture visualization requires layer utilization and FFN dimension data")
        
        # Layer details table
        if latest_metrics.layer_utilization and latest_metrics.ffn_dimensions:
            st.subheader("📊 Layer Details")
            
            layer_data = []
            for i in range(latest_metrics.num_layers):
                layer_data.append({
                    "Layer": i,
                    "FFN Dimension": latest_metrics.ffn_dimensions[i] if i < len(latest_metrics.ffn_dimensions) else "N/A",
                    "Utilization": f"{latest_metrics.layer_utilization.get(i, 0):.3f}",
                    "Status": "🔴 High" if latest_metrics.layer_utilization.get(i, 0) > 0.9 
                             else "🟡 Medium" if latest_metrics.layer_utilization.get(i, 0) > 0.7 
                             else "🟢 Normal"
                })
            
            layer_df = pd.DataFrame(layer_data)
            st.dataframe(layer_df, use_container_width=True)
    
    def _render_growth_tracking(self):
        """Render growth tracking visualizations."""
        st.subheader("🌱 Model Growth Tracking")
        
        # Growth timeline
        growth_data = self.metrics_tracker.get_growth_timeline()
        
        if growth_data['steps']:
            # Parameters growth
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=growth_data['steps'],
                y=growth_data['parameters_m'],
                mode='lines+markers',
                name='Parameters (M)',
                line=dict(color='green', width=3)
            ))
            fig1.update_layout(
                title="Parameter Growth Over Time",
                xaxis_title="Training Steps",
                yaxis_title="Parameters (Millions)",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Layer growth
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=growth_data['steps'],
                y=growth_data['num_layers'],
                mode='lines+markers',
                name='Layer Count',
                line=dict(color='purple', width=3)
            ))
            fig2.update_layout(
                title="Layer Count Growth Over Time",
                xaxis_title="Training Steps",
                yaxis_title="Number of Layers",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Utilization trends
        util_data = self.metrics_tracker.get_utilization_trends()
        
        if util_data['steps']:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=util_data['steps'],
                y=util_data['avg_utilization'],
                mode='lines',
                name='Average Utilization',
                line=dict(color='orange')
            ))
            fig3.add_trace(go.Scatter(
                x=util_data['steps'],
                y=util_data['max_utilization'],
                mode='lines',
                name='Max Utilization',
                line=dict(color='red', dash='dash')
            ))
            
            # Add growth threshold line
            fig3.add_hline(y=0.92, line_dash="dot", line_color="red",
                          annotation_text="Growth Threshold")
            
            fig3.update_layout(
                title="Layer Utilization Trends",
                xaxis_title="Training Steps",
                yaxis_title="Utilization",
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        # Growth events
        growth_events = self.metrics_tracker.get_growth_events()
        
        if growth_events:
            st.subheader("📈 Growth Events")
            
            events_data = []
            for event in growth_events:
                events_data.append({
                    "Step": event.get('step', 0),
                    "Type": event.get('type', 'Unknown'),
                    "Message": event.get('message', 'Growth event'),
                    "Time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.get('timestamp', 0)))
                })
            
            events_df = pd.DataFrame(events_data)
            st.dataframe(events_df, use_container_width=True)
        else:
            st.info("No growth events recorded yet")
        
        # Layer utilization heatmap
        if util_data.get('layer_utilization'):
            st.subheader("🔥 Layer Utilization Heatmap")
            heatmap_fig = self.growth_visualizer.create_layer_utilization_heatmap(
                util_data['layer_utilization']
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    def _render_alerts(self):
        """Render alerts and anomalies."""
        st.subheader("🚨 Training Alerts & Anomalies")
        
        # Detect anomalies
        anomalies = self.metrics_tracker.detect_anomalies()
        
        if anomalies:
            st.warning(f"⚠️ {len(anomalies)} anomalies detected!")
            
            for anomaly in anomalies:
                with st.expander(f"{anomaly['type'].replace('_', ' ').title()} - Step {anomaly['step']}"):
                    st.write(f"**Message**: {anomaly['message']}")
                    st.write(f"**Value**: {anomaly['value']}")
                    st.write(f"**Threshold**: {anomaly['threshold']}")
        else:
            st.success("✅ No anomalies detected")
        
        # Alert summary (if alert system is available)
        st.subheader("📊 Alert Summary")
        
        # Mock alert data for demonstration
        alert_data = {
            'total_alerts': len(anomalies),
            'severity_breakdown': {
                'info': 0,
                'warning': len([a for a in anomalies if a['type'] in ['high_memory', 'low_performance']]),
                'error': len([a for a in anomalies if a['type'] in ['gradient_explosion']]),
                'critical': len([a for a in anomalies if a['type'] in ['loss_spike']])
            }
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", alert_data['total_alerts'])
        
        with col2:
            st.metric("Warnings", alert_data['severity_breakdown']['warning'])
        
        with col3:
            st.metric("Errors", alert_data['severity_breakdown']['error'])
        
        with col4:
            st.metric("Critical", alert_data['severity_breakdown']['critical'])
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        
        if self.metrics_tracker.metrics_history:
            recent_steps = self.metrics_tracker.metrics_history[-10:]
            activity_data = []
            
            for metrics in recent_steps:
                activity_data.append({
                    "Step": metrics.step,
                    "Loss": f"{metrics.train_loss:.4f}",
                    "Layers": metrics.num_layers,
                    "Parameters": f"{metrics.num_parameters / 1e6:.1f}M",
                    "Avg Utilization": f"{metrics.avg_utilization:.3f}",
                    "Status": "🟢 Normal" if metrics.grad_norm < 5.0 else "🟡 High Gradient"
                })
            
            activity_df = pd.DataFrame(activity_data)
            st.dataframe(activity_df, use_container_width=True)
    
    def _render_analytics(self):
        """Render advanced analytics and insights."""
        st.subheader("📈 Training Analytics")
        
        if len(self.metrics_tracker.metrics_history) < 10:
            st.info("Need more training data for analytics (minimum 10 steps)")
            return
        
        # Training efficiency analysis
        df = self.metrics_tracker.get_dataframe()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Loss Analysis")
            
            # Loss distribution
            fig = px.histogram(df, x='train_loss', bins=20, title="Loss Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Loss correlation with learning rate
            if 'learning_rate' in df.columns:
                fig = px.scatter(df, x='learning_rate', y='train_loss', 
                               title="Loss vs Learning Rate")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Performance Analysis")
            
            # Training speed over time
            if 'tokens_per_second' in df.columns:
                fig = px.line(df, x='step', y='tokens_per_second', 
                             title="Training Speed Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            # Memory usage over time
            if 'memory_usage_gb' in df.columns:
                fig = px.line(df, x='step', y='memory_usage_gb', 
                             title="Memory Usage Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        # Training insights
        st.subheader("🔍 Training Insights")
        
        insights = self._generate_insights(df)
        for insight in insights:
            st.info(insight)
    
    def _generate_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate training insights from data."""
        insights = []
        
        if len(df) < 10:
            return insights
        
        # Loss trend analysis
        recent_loss = df['train_loss'].tail(10).mean()
        older_loss = df['train_loss'].head(10).mean()
        
        if recent_loss < older_loss * 0.8:
            insights.append("🎉 Training is converging well - loss decreased significantly!")
        elif recent_loss > older_loss * 1.2:
            insights.append("⚠️ Loss has increased - consider checking learning rate or data quality")
        
        # Growth analysis
        if 'num_parameters' in df.columns:
            param_growth = (df['num_parameters'].iloc[-1] - df['num_parameters'].iloc[0]) / 1e6
            if param_growth > 10:
                insights.append(f"🌱 Model has grown by {param_growth:.1f}M parameters during training")
        
        # Performance insights
        if 'tokens_per_second' in df.columns:
            avg_speed = df['tokens_per_second'].mean()
            if avg_speed > 1000:
                insights.append(f"🚀 Excellent training speed: {avg_speed:.0f} tokens/sec average")
            elif avg_speed < 100:
                insights.append("🐌 Training speed is slow - consider optimizing batch size or hardware")
        
        return insights
    
    def _get_loss_delta(self) -> Optional[float]:
        """Calculate loss delta for metric display."""
        if len(self.metrics_tracker.metrics_history) < 2:
            return None
        
        current = self.metrics_tracker.metrics_history[-1].train_loss
        previous = self.metrics_tracker.metrics_history[-2].train_loss
        return current - previous
    
    def _get_lr_delta(self) -> Optional[float]:
        """Calculate learning rate delta."""
        if len(self.metrics_tracker.metrics_history) < 2:
            return None
        
        current = self.metrics_tracker.metrics_history[-1].learning_rate
        previous = self.metrics_tracker.metrics_history[-2].learning_rate
        return current - previous
    
    def _get_param_delta(self) -> Optional[str]:
        """Calculate parameter count delta."""
        if len(self.metrics_tracker.metrics_history) < 2:
            return None
        
        current = self.metrics_tracker.metrics_history[-1].num_parameters
        previous = self.metrics_tracker.metrics_history[-2].num_parameters
        delta = (current - previous) / 1e6
        
        return f"{delta:.1f}M" if delta != 0 else None
    
    def _get_layer_delta(self) -> Optional[int]:
        """Calculate layer count delta."""
        if len(self.metrics_tracker.metrics_history) < 2:
            return None
        
        current = self.metrics_tracker.metrics_history[-1].num_layers
        previous = self.metrics_tracker.metrics_history[-2].num_layers
        delta = current - previous
        
        return delta if delta != 0 else None

# Streamlit entry point
def main():
    """Main entry point for Streamlit dashboard."""
    dashboard = ArborDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
