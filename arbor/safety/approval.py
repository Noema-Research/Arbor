"""
Human Approval Interface for Arbor Safety System

This module provides interfaces for human oversight and approval
of critical AI operations like model growth and behavior changes.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading

class HumanApprovalInterface:
    """Interface for requesting and managing human approval of AI operations."""
    
    def __init__(self, approval_file: str = "safety_logs/growth_approval.json"):
        self.approval_file = Path(approval_file)
        self.approval_file.parent.mkdir(exist_ok=True)
        
        # Initialize approval file if it doesn't exist
        if not self.approval_file.exists():
            self._initialize_approval_file()
    
    def _initialize_approval_file(self):
        """Initialize the approval file with default structure."""
        default_structure = {
            "pending_approvals": [],
            "approval_history": [],
            "emergency_overrides": [],
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.approval_file, 'w') as f:
            json.dump(default_structure, f, indent=2)
    
    def request_approval(self, 
                        operation_type: str,
                        operation_params: Dict[str, Any],
                        urgency: str = "normal",
                        timeout_minutes: int = 30) -> bool:
        """
        Request human approval for an operation.
        
        Args:
            operation_type: Type of operation (e.g., "growth", "modification")
            operation_params: Parameters of the operation
            urgency: Urgency level ("low", "normal", "high", "critical")
            timeout_minutes: How long to wait for approval
            
        Returns:
            True if approved, False if denied or timed out
        """
        request_id = f"{operation_type}_{int(time.time())}"
        
        # Create approval request
        approval_request = {
            "request_id": request_id,
            "operation_type": operation_type,
            "operation_params": operation_params,
            "urgency": urgency,
            "requested_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=timeout_minutes)).isoformat(),
            "status": "pending",
            "approver": None,
            "approved_at": None,
            "denial_reason": None
        }
        
        # Add to pending approvals
        self._add_pending_approval(approval_request)
        
        # Display approval request to human
        self._display_approval_request(approval_request)
        
        # Wait for approval or timeout
        return self._wait_for_approval(request_id, timeout_minutes)
    
    def _add_pending_approval(self, request: Dict[str, Any]):
        """Add a pending approval request."""
        try:
            with open(self.approval_file, 'r') as f:
                data = json.load(f)
            
            data["pending_approvals"].append(request)
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self.approval_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error adding approval request: {e}")
    
    def _display_approval_request(self, request: Dict[str, Any]):
        """Display approval request to human operator."""
        print("\n" + "="*80)
        print("🚨 HUMAN APPROVAL REQUIRED")
        print("="*80)
        print(f"Request ID: {request['request_id']}")
        print(f"Operation: {request['operation_type']}")
        print(f"Urgency: {request['urgency'].upper()}")
        print(f"Requested at: {request['requested_at']}")
        print(f"Expires at: {request['expires_at']}")
        print("\nOperation Details:")
        for key, value in request['operation_params'].items():
            print(f"  {key}: {value}")
        
        print("\n📋 To approve or deny this request:")
        print(f"1. Edit the file: {self.approval_file}")
        print(f"2. Find request ID: {request['request_id']}")
        print("3. Set 'status' to 'approved' or 'denied'")
        print("4. Add your name to 'approver' field")
        print("5. Add 'approved_at' timestamp or 'denial_reason'")
        
        print("\n⚠️  SAFETY REMINDER:")
        print("   - Only approve if you understand the implications")
        print("   - Consider potential risks and safeguards")
        print("   - Deny if any aspect seems suspicious or dangerous")
        print("="*80)
    
    def _wait_for_approval(self, request_id: str, timeout_minutes: int) -> bool:
        """Wait for human approval with timeout."""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        print(f"⏳ Waiting for human approval (timeout: {timeout_minutes} minutes)...")
        
        while time.time() - start_time < timeout_seconds:
            status = self._check_approval_status(request_id)
            
            if status == "approved":
                print("✅ Operation approved by human operator")
                return True
            elif status == "denied":
                reason = self._get_denial_reason(request_id)
                print(f"❌ Operation denied by human operator: {reason}")
                return False
            
            time.sleep(5)  # Check every 5 seconds
        
        # Timeout reached
        print("⏰ Approval request timed out - operation denied")
        self._mark_as_expired(request_id)
        return False
    
    def _check_approval_status(self, request_id: str) -> Optional[str]:
        """Check the current status of an approval request."""
        try:
            with open(self.approval_file, 'r') as f:
                data = json.load(f)
            
            for request in data["pending_approvals"]:
                if request["request_id"] == request_id:
                    return request.get("status", "pending")
            
            return None
            
        except Exception as e:
            print(f"Error checking approval status: {e}")
            return None
    
    def _get_denial_reason(self, request_id: str) -> str:
        """Get the denial reason for a request."""
        try:
            with open(self.approval_file, 'r') as f:
                data = json.load(f)
            
            for request in data["pending_approvals"]:
                if request["request_id"] == request_id:
                    return request.get("denial_reason", "No reason provided")
            
            return "Request not found"
            
        except Exception as e:
            return f"Error retrieving denial reason: {e}"
    
    def _mark_as_expired(self, request_id: str):
        """Mark a request as expired due to timeout."""
        try:
            with open(self.approval_file, 'r') as f:
                data = json.load(f)
            
            for request in data["pending_approvals"]:
                if request["request_id"] == request_id:
                    request["status"] = "expired"
                    request["expired_at"] = datetime.now().isoformat()
                    break
            
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self.approval_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error marking request as expired: {e}")
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        try:
            with open(self.approval_file, 'r') as f:
                data = json.load(f)
            
            # Filter out expired requests
            now = datetime.now()
            pending = []
            
            for request in data["pending_approvals"]:
                expires_at = datetime.fromisoformat(request["expires_at"])
                if request["status"] == "pending" and now < expires_at:
                    pending.append(request)
            
            return pending
            
        except Exception as e:
            print(f"Error getting pending approvals: {e}")
            return []
    
    def emergency_override(self, operation_type: str, justification: str, approver: str) -> str:
        """
        Create an emergency override for critical situations.
        
        Args:
            operation_type: Type of operation to override
            justification: Detailed justification for override
            approver: Name/ID of person authorizing override
            
        Returns:
            Override token that can be used to bypass normal approval
        """
        override_token = f"EMERGENCY_{int(time.time())}"
        
        override_record = {
            "override_token": override_token,
            "operation_type": operation_type,
            "justification": justification,
            "approver": approver,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),  # 1 hour expiry
            "used": False
        }
        
        try:
            with open(self.approval_file, 'r') as f:
                data = json.load(f)
            
            data["emergency_overrides"].append(override_record)
            data["last_updated"] = datetime.now().isoformat()
            
            with open(self.approval_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"🚨 EMERGENCY OVERRIDE CREATED: {override_token}")
            print(f"   Operation: {operation_type}")
            print(f"   Authorized by: {approver}")
            print(f"   Expires in 1 hour")
            
            return override_token
            
        except Exception as e:
            print(f"Error creating emergency override: {e}")
            return ""
    
    def validate_override_token(self, token: str, operation_type: str) -> bool:
        """Validate an emergency override token."""
        try:
            with open(self.approval_file, 'r') as f:
                data = json.load(f)
            
            now = datetime.now()
            
            for override in data["emergency_overrides"]:
                if (override["override_token"] == token and 
                    override["operation_type"] == operation_type and
                    not override["used"] and
                    now < datetime.fromisoformat(override["expires_at"])):
                    
                    # Mark as used
                    override["used"] = True
                    override["used_at"] = now.isoformat()
                    
                    with open(self.approval_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"✅ Emergency override validated: {token}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error validating override token: {e}")
            return False

# Example approval file structure for human operators
EXAMPLE_APPROVAL_FILE = """
{
  "pending_approvals": [
    {
      "request_id": "growth_1694456789",
      "operation_type": "growth",
      "operation_params": {
        "factor": 2.0,
        "add_hidden": 512,
        "reason": "plateau_trigger",
        "layers_affected": 2
      },
      "urgency": "normal",
      "requested_at": "2025-09-11T10:30:00",
      "expires_at": "2025-09-11T11:00:00",
      "status": "pending",
      "approver": null,
      "approved_at": null,
      "denial_reason": null
    }
  ],
  "approval_history": [],
  "emergency_overrides": [],
  "last_updated": "2025-09-11T10:30:00"
}

To approve the above request, change:
"status": "pending"  →  "status": "approved"
"approver": null     →  "approver": "Your Name"
"approved_at": null  →  "approved_at": "2025-09-11T10:35:00"

To deny the request:
"status": "pending"    →  "status": "denied"
"approver": null       →  "approver": "Your Name"
"denial_reason": null  →  "denial_reason": "Safety concern: growth rate too high"
"""
