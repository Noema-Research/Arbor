#!/usr/bin/env python3
"""
Arbor Training with Live Dashboard Integration

This example shows how to integrate the Arbor tracking dashboard
with your training process for real-time monitoring.

Usage:
    python examples/training_with_dashboard.py
    
    # In another terminal, start the dashboard:
    streamlit run arbor/tracking/dashboard.py
"""

import torch
import torch.nn as nn
import time
import random
from pathlib import Path

# Arbor imports
from arbor.modeling.model import ArborTransformer, ArborConfig
from arbor.data.synthetic import SyntheticDataset
from arbor.tracking import TrainingMonitor
from torch.utils.data import DataLoader

def simulate_arbor_training_with_tracking():
    """Simulate Arbor training with comprehensive tracking."""
    
    print("🌳 Starting Arbor Training with Live Dashboard")
    print("=" * 50)
    
    # Configuration
    config = ArborConfig(
        vocab_size=10000,
        dim=512,
        num_layers=24,  # Will grow to 64
        num_heads=8,
        ffn_dim=2048,
        max_seq_length=1024,
        
        # Enable growth
        layer_growth_enabled=True,
        min_layers=24,
        max_layers=64,
        layer_growth_threshold=0.92,
        layer_growth_factor=4,
        layer_growth_cooldown=50,  # Shorter for demo
        
        growth_enabled=True,
        growth_factor=2.0
    )
    
    # Create model
    model = ArborTransformer(config)
    print(f"📊 Created model with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    # Create dataset and dataloader
    dataset = SyntheticDataset(
        vocab_size=config.vocab_size,
        seq_length=config.max_seq_length,
        num_sequences=5000
    )
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Setup training components
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize tracking system
    print("🔧 Initializing tracking system...")
    monitor = TrainingMonitor(
        save_dir="training_logs",
        update_interval=1.0
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Add custom alert handler
    def custom_alert_handler(alert):
        print(f"🚨 ALERT: {alert.severity.value.upper()} - {alert.message}")
    
    monitor.add_custom_alert_handler(custom_alert_handler)
    
    print("🚀 Starting training loop...")
    print("   Open another terminal and run: streamlit run arbor/tracking/dashboard.py")
    print("   Dashboard will be available at: http://localhost:8501")
    print()
    
    # Training loop
    step = 0
    epoch = 0
    start_time = time.time()
    
    try:
        for epoch in range(10):  # 10 epochs for demo
            print(f"\n📚 Epoch {epoch + 1}/10")
            
            for batch_idx, batch in enumerate(dataloader):
                if step >= 500:  # Limit for demo
                    break
                
                # Simulate training step
                model.train()
                optimizer.zero_grad()
                
                # Prepare data
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                
                # Forward pass
                outputs = model(input_ids, labels=targets, return_dict=True)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Simulate layer utilization (normally this would be tracked automatically)
                layer_utilization = {}
                for i in range(len(model.layers)):
                    # Simulate increasing utilization over time with some randomness
                    base_util = min(0.95, 0.3 + (step / 500) * 0.6)  # Gradually increase
                    noise = random.uniform(-0.1, 0.1)
                    layer_utilization[i] = max(0.0, min(1.0, base_util + noise))
                
                # Calculate average utilization
                avg_utilization = sum(layer_utilization.values()) / len(layer_utilization)
                max_utilization = max(layer_utilization.values())
                
                # Simulate performance metrics
                tokens_per_second = random.uniform(800, 1200)
                memory_usage_gb = random.uniform(8, 16)
                gpu_utilization = random.uniform(80, 95)
                
                # Check for growth events
                growth_events = []
                if step > 0 and step % 100 == 0 and avg_utilization > 0.92:
                    # Simulate layer growth
                    if len(model.layers) < config.max_layers:
                        old_layers = len(model.layers)
                        # In real training, this would be: model.grow_layers(4)
                        # For simulation, we'll just track the event
                        growth_events.append({
                            'type': 'layer_growth',
                            'message': f'Added 4 layers: {old_layers} → {old_layers + 4}',
                            'old_layers': old_layers,
                            'new_layers': old_layers + 4
                        })
                        print(f"🌱 Layer growth simulated at step {step}")
                
                # Prepare model state for tracking
                model_state = {
                    'num_layers': len(model.layers),
                    'num_parameters': sum(p.numel() for p in model.parameters()),
                    'ffn_dimensions': [getattr(layer.ffn, 'current_ffn_dim', config.ffn_dim) 
                                     for layer in model.layers],
                    'layer_utilization': layer_utilization,
                    'avg_utilization': avg_utilization,
                    'max_utilization': max_utilization,
                    'tokens_per_second': tokens_per_second,
                    'growth_events': growth_events
                }
                
                # Log metrics to tracking system
                metrics = monitor.log_training_step(
                    step=step,
                    epoch=epoch,
                    train_loss=loss.item(),
                    learning_rate=optimizer.param_groups[0]['lr'],
                    grad_norm=grad_norm.item(),
                    model_state=model_state,
                    val_loss=None  # Would add validation loss if available
                )
                
                # Print progress
                if step % 20 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"  Step {step:3d} | Loss: {loss.item():.4f} | "
                          f"Layers: {len(model.layers):2d} | "
                          f"Params: {model_state['num_parameters'] / 1e6:.1f}M | "
                          f"Avg Util: {avg_utilization:.3f} | "
                          f"Time: {elapsed_time:.1f}s")
                
                step += 1
                
                # Small delay to make the demo more realistic
                time.sleep(0.1)
            
            if step >= 500:
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Generate final report
        print("\n📊 Generating training report...")
        report_file = monitor.export_training_report()
        print(f"📄 Training report saved: {report_file}")
        
        print("\n🎉 Training completed!")
        print("📊 Dashboard data saved to: training_logs/")
        print("🌐 View the dashboard at: http://localhost:8501")

def main():
    """Main function."""
    simulate_arbor_training_with_tracking()

if __name__ == "__main__":
    main()
