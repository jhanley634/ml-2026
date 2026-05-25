#! /usr/bin/env uv run streamlit run

# Source - https://stackoverflow.com/q/79938463
# Posted by Vital Fernández
# Retrieved 2026-05-09, License - CC BY-SA 4.0

import pymc as pm
import streamlit as st
from pymc.progress_bar import ProgressBarManager

st.title("PyMC + Nutpie Sampler")

n_draws = 2_000
n_tune = 1000
n_chains = 4

if st.button("Run Sampling"):

    total_steps = n_draws * n_chains
    chain_draws = dict.fromkeys(range(n_chains), 0)

    progress_bar = st.progress(0, text=f"Sampling for {n_chains * n_draws} steps...")

    old_update = ProgressBarManager.update

    def new_update(self, chain_idx, is_last, draw, tuning, stats) -> None:
        if not tuning:
            chain_draws[chain_idx] += 1
            completed = sum(chain_draws.values())
            progress = min(completed / total_steps, 1.0)
            progress_bar.progress(
                progress,
                text=f"Sampling... {completed}/{total_steps} steps ({progress * 100:.1f}%)",
            )
        old_update(self, chain_idx, is_last, draw, tuning, stats)

    ProgressBarManager.update = new_update

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        obs = pm.Normal("obs", mu=mu, sigma=1, observed=[1, 2, 3])

        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=n_chains,
            nuts_sampler="nutpie",
            nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"},
        )

    # Restore original to avoid side effects on reruns
    ProgressBarManager.update = old_update

    progress_bar.progress(1.0, text="Sampling complete!")
    st.success("Sampling complete!")

    st.subheader("Posterior Summary")
    st.dataframe(pm.stats.summary(trace))
