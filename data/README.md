# Data

| File | Contents |
|---|---|
| `posteriordb/iohmm_reg_simulated.json.zip` | IOHMM regression dataset ([posteriordb](https://github.com/stan-dev/posteriordb)) |
| `posteriordb/three_men1.json.zip` | LDA dataset (posteriordb) |
| `reference/ref_sur_case2.npz` | SUR long-NUTS reference: `oracle_samples` (80000, 5) |
| `reference/ref_iohmm_reg_simulated-iohmm_reg.npz` | IOHMM long-NUTS reference: `draws` (80000, 29), unconstrained coordinates, symmetrized over the 6 label permutations |
| `reference/ref_three_men1-ldaK2_chain{0,1,2,3}.npz` | LDA long-NUTS reference, one file per chain: `draws` (20000, 502) in Stan's unconstrained (ILR) coordinates; chains 0-1 lie in one topic-label basin, 2-3 in the other |
| `reference/*.diag.json` | run settings and convergence diagnostics of the reference runs |
| `sur/sur_case2_T10_M5_k1_seed300.npz` | the SUR Case 2 dataset (frozen: regenerating it from its seed depends on the LAPACK build) |
| `sur/sur_ref_modes.npz` | the three local maxima of the SUR posterior and the reference's basin weights |

The LDA reference is split by chain to keep every file under GitHub's 100 MB limit.
`experiments/common.lda_reference_stan()` concatenates the chains in order.
