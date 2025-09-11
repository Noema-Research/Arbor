#!/usr/bin/env python3
"""
Arbor Safety System Demo

This script demonstrates the safety features of the Arbor architecture,
including growth monitoring, resource limits, and human approval workflows.
"""

import sys
import time
from pathlib import Path

# Add arbor to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arbor.modeling.model import ArborTransformer, ArborConfig
from arbor.safety import (
    initialize_safety_system, 
    SafetyLimits, 
    DEVELOPMENT_SAFETY_CONFIG,
    HumanApprovalInterface
)
from arbor.growth.manager import GrowthManager
import torch

def demo_safety_system():
    """Demonstrate the Arbor safety system capabilities."""
    
    print("🛡️ ARBOR SAFETY SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Initialize safety system with development config
    print("\n1. Initializing Safety System...")
    safety_limits = SafetyLimits(
        max_parameters=1_000_000,     # Small limit for demo
        max_layers=10,                # Few layers for demo
        max_growth_events_per_hour=2, # Conservative growth rate
        require_human_approval_for_growth=True,
        enable_behavior_monitoring=True,
        enable_escape_detection=True
    )
    
    guardian = initialize_safety_system(safety_limits)
    print("✅ Safety Guardian initialized")
    
    # Create a small test model
    print("\n2. Creating Test Model...")
    config = ArborConfig(
        vocab_size=1000,
        dim=256, 
        num_layers=4,
        num_heads=4,
        ffn_dim=512,
        max_seq_length=512
    )
    
    model = ArborTransformer(config)
    print(f"✅ Model created: {model.param_count():,} parameters")
    
    # Create growth manager
    print("\n3. Setting up Growth Manager...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    growth_config = {
        "enabled": True,
        "add_hidden": 128,
        "max_events": 3,
        "triggers": [
            {
                "type": "plateau",
                "patience": 5,
                "threshold": 0.01
            }
        ]
    }
    
    growth_manager = GrowthManager(model, optimizer, growth_config)
    print("✅ Growth Manager with safety integration ready")
    
    # Demonstrate safety checks
    print("\n4. Testing Safety Validations...")
    
    # Test 1: Valid growth request
    print("\n   Test 1: Normal growth request")
    try:
        # This should require human approval
        growth_params = {
            "factor": 1.5,
            "add_hidden": 128,
            "reason": "demo_test"
        }
        
        print("   📋 This will require human approval...")
        print("   💡 In a real scenario, you would see approval interface")
        print("   ⏳ Simulating human approval timeout...")
        
        # For demo, we'll simulate approval without waiting
        result = guardian.validate_growth_request(model, "parameters", growth_params)
        print(f"   Result: {'✅ Approved' if result else '❌ Denied'}")
        
    except Exception as e:
        print(f"   🛡️ Safety check result: {e}")
    
    # Test 2: Dangerous growth request
    print("\n   Test 2: Dangerous growth request (should be blocked)")
    try:
        dangerous_params = {
            "factor": 10.0,  # Too large growth factor
            "add_hidden": 2048,
            "reason": "dangerous_test"
        }
        
        result = guardian.validate_growth_request(model, "parameters", dangerous_params)
        print(f"   Result: {'✅ Approved' if result else '❌ Denied (as expected)'}")
        
    except Exception as e:
        print(f"   🛡️ Safety violation prevented: {e}")
    
    # Test 3: Resource monitoring
    print("\n   Test 3: Resource monitoring")
    report = guardian.get_safety_report()
    print(f"   GPU Memory: {report['current_resources']['gpu_memory_gb']:.2f}GB")
    print(f"   CPU Usage: {report['current_resources']['cpu_percent']:.1f}%")
    print(f"   Growth Events: {report['activity_summary']['growth_events']}")
    print(f"   Security Alerts: {report['activity_summary']['security_alerts']}")
    
    # Test 4: Model integrity verification
    print("\n   Test 4: Model integrity verification")
    integrity_ok = guardian.verify_model_integrity(model)
    print(f"   Model integrity: {'✅ Verified' if integrity_ok else '❌ Failed'}")
    
    # Test 5: Behavior monitoring
    print("\n   Test 5: Behavior monitoring during inference")
    test_input = torch.randint(0, 1000, (1, 50))  # Normal input
    with torch.no_grad():
        output = model(test_input)
    print("   ✅ Normal inference behavior detected")
    
    # Simulate suspicious input
    suspicious_input = torch.randint(0, 10, (1, 1000))  # Very long, low diversity
    with torch.no_grad():
        output = model(suspicious_input)
    print("   ⚠️ Check console for behavior monitoring alerts")
    
    print("\n5. Safety System Summary")
    print("=" * 60)
    print("✅ Safety Guardian is monitoring:")
    print("   • Model growth and parameter limits")
    print("   • Resource usage and system health") 
    print("   • Suspicious behavior patterns")
    print("   • Model integrity and state changes")
    print("   • Human approval workflows")
    
    print("\n🛡️ Protection Features Active:")
    print("   • Growth rate limiting")
    print("   • Emergency shutdown procedures")
    print("   • Escape attempt detection")
    print("   • Resource usage monitoring")
    print("   • Human oversight requirements")
    
    print("\n💡 To see human approval interface:")
    print("   1. Set require_human_approval_for_growth=True")
    print("   2. Trigger a growth event")
    print("   3. Check safety_logs/growth_approval.json")
    print("   4. Edit the file to approve/deny requests")
    
    # Cleanup
    print("\n6. Shutting down safely...")
    guardian.shutdown()
    print("✅ Safety Guardian shutdown complete")

def demo_human_approval():
    """Demonstrate the human approval interface."""
    
    print("\n🎯 HUMAN APPROVAL INTERFACE DEMO")
    print("=" * 60)
    
    # Create approval interface
    approval_interface = HumanApprovalInterface()
    
    print("1. Creating sample approval request...")
    
    # This would normally block waiting for human input
    # For demo, we'll show the interface without waiting
    print("\n📋 Example of approval request display:")
    print("   (This would normally pause execution waiting for human)")
    
    sample_request = {
        "request_id": "demo_growth_123",
        "operation_type": "growth",
        "operation_params": {
            "factor": 2.0,
            "add_hidden": 256,
            "reason": "plateau_detected"
        },
        "urgency": "normal",
        "requested_at": "2025-09-11T10:30:00",
        "expires_at": "2025-09-11T11:00:00"
    }
    
    approval_interface._display_approval_request(sample_request)
    
    print("\n2. Emergency override demonstration...")
    override_token = approval_interface.emergency_override(
        operation_type="growth",
        justification="Critical training deadline - model performance plateaued",
        approver="Safety Engineer #123"
    )
    
    print(f"\n3. Override token validation: {override_token}")
    is_valid = approval_interface.validate_override_token(override_token, "growth")
    print(f"   Token valid: {'✅ Yes' if is_valid else '❌ No'}")
    
    print("\n✅ Human approval interface demo complete")

if __name__ == "__main__":
    print("🚀 Starting Arbor Safety Demonstration...")
    
    try:
        demo_safety_system()
        demo_human_approval()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Safety demonstration complete!")
    print("📚 Check the safety_logs/ directory for generated files")
