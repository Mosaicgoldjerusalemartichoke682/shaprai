"""Training modules for Elyan-class agents.

Three-phase training pipeline:
  1. SFT (Supervised Fine-Tuning) -- Teach the base model to speak like an Elyan agent
  2. DPO (Direct Preference Optimization) -- Align preferences toward principled behavior
  3. DriftLock -- Verify identity coherence across long conversations
"""
