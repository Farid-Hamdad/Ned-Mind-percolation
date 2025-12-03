# Ned-Mind-percolation
Discovery of cognitive percolation threshold p_c ≈ 0.055 in self-organizing neural networks
# NeD-Mind v10.2K - Cognitive Percolation Threshold

**Author**: Farid Hamdad ([ORCID](https://orcid.org/0009-0009-2097-1625))  
**Lab**: NeD-Mind Laboratory (Independent Research)  
**Contact**: farid_hamdad@ned-mind.org  
**Status**: Paper submitted to arXiv (next)

---

## THE DISCOVERY

Neural networks exhibit a **first-order phase transition** at critical density `p_c ≈ 0.055`.  
Below this threshold, **cognitive coherence collapses to zero in one step**.

![coherence collapse](coherence_plot.png)

**Experimental evidence**:
- **Before ablation**: Coherence = 0.967, Density = 0.097, Skewness = 0.54
- **After 90% hub ablation**: Coherence = 0.000, Density = 0.046, Skewness = 3.74
- **Recovery**: None (LCR loop ruptured)

---

##  QUICK START (Reproduce in 5 minutes)

```bash
# Clone & install
git clone https://github.com/faridhamdad/nedmind-percolation.git
cd nedmind-percolation
pip install -r requirements.txt

# Run the critical experiment
python nedmind_runner.py --config perc_test.yaml
