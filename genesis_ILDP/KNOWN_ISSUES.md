# Known Issues and Future Improvements

## 🚨 **Critical Issue: Temporal Data Leakage in MultiStepWrapper**

### **Problem Description:**
The current MultiStepWrapper has a temporal data leakage bug where the final observation from the action execution is included in the observation buffer and used for the next policy prediction.

### **Detailed Issue:**
```python
# Current problematic flow:
t=0: obs_buffer = [obs_t0]
t=0: Policy predicts using [obs_t0, obs_t0] (duplicated from reset)
t=0-8: Execute 8 actions, collecting ALL observations including final step
     obs_buffer = [obs_t0, obs_t1, obs_t2, ..., obs_t8]  ← INCLUDES obs_t8!
t=8: get_obs() returns [obs_t7, obs_t8] ← obs_t8 shouldn't be available yet!
t=8: Policy uses obs_t8 to predict actions for t8+ ← DATA LEAKAGE!
```

### **What Should Happen:**
```python
# Correct flow:
t=0-8: Execute 8 actions, collect intermediate observations only
     obs_buffer = [obs_t0, obs_t1, obs_t2, ..., obs_t7]  ← Stop at t7!
t=8: get_obs() returns [obs_t6, obs_t7] ← Correct temporal alignment
t=8: Policy uses observations up to t7 to predict actions for t8+
t=8: After prediction, add obs_t8 to buffer for next iteration
```

### **Root Cause:**
In `MultiStepWrapper.step()` line ~125:
```python
# BUG: Appends observation from EVERY step including final
self.obs[env_idx].append(env_obs)  # Including final step obs_t8
```

### **Impact:**
- **Training**: Policy sees "future" information during training, leading to data leakage
- **Temporal Consistency**: Breaks the causal relationship between observations and predictions
- **Real Robot Transfer**: Trained policy may rely on information not available in real deployment

### **Proposed Solutions:**

#### **Option 1: Exclude Final Observation (Simple)**
```python
# Only append observations from intermediate steps
if step_idx < self.n_action_steps - 1:
    self.obs[env_idx].append(env_obs)
# Handle final observation separately for next iteration
```

#### **Option 2: Progressive Execution (Realistic)**
```python
# Implement receding horizon like real diffusion_policy
# Execute only part of predicted actions per step
# More complex but aligns with real robot behavior
```

### **Priority:** 🚨 **HIGH** - Affects training validity and real robot transfer

### **Complexity:** 🔧 **MEDIUM** - Requires careful handling of observation timing

### **Recommendation:** 
For now, continue with current approach for development/testing. Address this issue when preparing for serious training or real robot deployment.

---

## 📋 **Other Improvement Areas:**

### **1. Environment Auto-Reset for Continuous Training**
- Current: Terminated environments are removed from active list
- Should: Automatically reset terminated environments to maintain constant batch size
- Benefits: Better GPU utilization, continuous training

### **2. Past Action Temporal Alignment**  
- Current: Uses 1 past action for n_obs_steps=2 observations
- Consider: Align past action count with observation count for consistency
- Impact: Minor - current approach is reasonable for simulation

### **3. Memory Optimization**
- Current: Stores all environment data even for inactive environments  
- Consider: Dynamic memory management for large-scale training

---

*Last Updated: 2025-09-14*  
*Identified by: User analysis of temporal data flow*