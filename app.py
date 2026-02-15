import streamlit as st
import os
import torch
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from monai.data import Dataset, DataLoader
from train import prep, DEVICE
from engine import run_synthesis_logic

st.set_page_config(page_title="NeuroSynth WebApp", layout="wide")

st.title("NeuroSynth: AI Contrast Synthesis")
st.markdown("Generates synthetic T1-Gd MRI from T1-Pre, FLAIR, and BRAVO sequences.")

# 1. SIDEBAR
st.sidebar.header("📂 Upload Patient Data")
t1_file = st.sidebar.file_uploader("Upload t1_pre.nii.gz", type=["gz"])
flair_file = st.sidebar.file_uploader("Upload flair.nii.gz", type=["gz"])
bravo_file = st.sidebar.file_uploader("Upload bravo.nii.gz", type=["gz"])

st.sidebar.divider()
slice_idx = st.sidebar.slider("Axial Slice Number", 0, 95, 70)
n_samples = st.sidebar.number_input("Uncertainty Samples (MC)", 3, 10, 5)

if st.sidebar.button("✨ Run Inference"):
    if t1_file and flair_file and bravo_file:
        # Containers for Progress
        st.write("### Generation In Progress...")
        status_txt = st.empty()
        prog_bar = st.progress(0)

        with st.spinner("Preparing data..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = {
                    "t1": os.path.join(tmpdir, "t1_pre.nii.gz"),
                    "flair": os.path.join(tmpdir, "flair.nii.gz"),
                    "bravo": os.path.join(tmpdir, "bravo.nii.gz"),
                    "target": os.path.join(tmpdir, "t1_gd.nii.gz") 
                }
                for key, uploaded in zip(["t1", "flair", "bravo", "target"], 
                                         [t1_file, flair_file, bravo_file, t1_file]):
                    with open(paths[key], "wb") as f:
                        f.write(uploaded.getbuffer())

                data = [{"t1": paths["t1"], "flair": paths["flair"], 
                         "bravo": paths["bravo"], "target": paths["target"]}]
                
                ds = Dataset(data=data, transform=prep)
                loader = DataLoader(ds, batch_size=1)
                batch = next(iter(loader))
                inputs = batch["image"][:,:,:,:,slice_idx].to(DEVICE)

            # Execution with Progress Tracking
            fake, uncert, sid = run_synthesis_logic(inputs, n_samples=n_samples, 
                                                    prog_bar=prog_bar, 
                                                    status_txt=status_txt)
            
            # Clear progress trackers once done
            prog_bar.empty()
            status_txt.empty()

        # 3. DISPLAY RESULTS
        st.success(f"Inference Complete! Session ID: {sid}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Input T1")
            inp_t1 = inputs[0, 0].cpu().numpy()
            inp_t1 = (inp_t1 - inp_t1.min()) / (inp_t1.max() - inp_t1.min() + 1e-8)
            st.image(inp_t1, use_container_width=True)

        with col2:
            st.subheader("Synthetic T1-Gd")
            st.image(fake, use_container_width=True)

        with col3:
            st.subheader("Uncertainty Map")
            fig, ax = plt.subplots()
            ax.imshow(uncert, cmap='hot')
            plt.axis('off')
            st.pyplot(fig)
            st.caption("Brighter areas = High Uncertainty")
            
        # 4. DOWNLOAD OPTION
        st.divider()
        st.download_button(label="Download Generated NPY", 
                           data=fake.tobytes(), 
                           file_name=f"synth_{sid}.npy")

    else:
        st.error("Please upload all three .nii.gz sequences in the sidebar.")
        