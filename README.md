# Meta-Learning-Fronthaul-Compression-CRAN
Simulation code for "Meta-Learning-Based Fronthaul Compression for Cloud Radio Access Networks", by Ruihua Qiao, Tao Jiang, Wei Yu, IEEE Transactions on Wireless Communications. To appear.

If you have any questions, please feel free to reach out to 
Ruihua Qiao: [ruihua.qiao@gmail.com](mailto:ruihua.qiao@gmail.com).

# Abstract of Article
This paper investigates the fronthaul compression
problem in a user-centric cloud radio access network, in which
single-antenna users are served by a central processor (CP)
cooperatively via a cluster of remote radio heads (RRHs). To
satisfy the fronthaul capacity constraint, this paper proposes
a transform-compress-forward scheme, which consists of well-
designed transformation matrices and uniform quantizers. The
transformation matrices perform dimension reduction in the
uplink and dimension expansion in the downlink. To reduce
the communication overhead for designing the transformation
matrices, this paper further proposes a deep learning framework
to first learn a suboptimal transformation matrix at each RRH
based on the local channel state information (CSI), and then
to refine it iteratively. To facilitate the refinement process, we
propose an efficient signaling scheme that only requires the
transmission of low-dimensional effective CSI and its gradient
between the CP and RRH, and further, a meta-learning based
gated recurrent unit network to reduce the number of signaling
transmission rounds. For the sum-rate maximization problem,
simulation results show that the proposed two-stage neural net-
work can perform close to the fully cooperative global CSI based
benchmark with significantly reduced communication overhead
for both the uplink and the downlink. Moreover, using the first
stage alone can already outperform the existing local CSI based
benchmark.

# Content of Code Package
The code is organized as follows:

    .
    ├── uplink/
    │   ├── gen_test_UELocs.py (generate testing dataset)
    │   ├── LocalCSI_DNN.py (proposed local CSI based deep learning method)
    │   ├── meta_GRU.py (proposed local CSI+GRU meta learning method)
    │   ├── SingleCellProcess.py (single cell proessing benchmark)
    │   ├── EVD.py (EVD based benchmark)
    │   ├── Global_GD.py (global CSI GD benchamrk & local CSI DNN+GD benchmark)
    │   ├── funcs.py (util functions)
    │   ├── funcs_autograd.py (funcstions for autograd of matrices W with pytorch)
    │   ├── plot_result/
    │   │   ├── plot_cdf/
    │   │   │   ├── plot_cdf.py (reproduce Fig. 4 & 5)
    │   │   │   └── some .mat files (saved after running the above scripts)
    │   │   ├── plot_convergence/
    │   │   │   ├── plot_convergence.py (reproduce Fig. 6)
    │   │   │   ├── plot_convergence_1axis.py (reproduce Fig. 7)
    │   │   │   └── some .mat files
    │   │   ├── plot_CE_err/
    │   │   │   ├── plot_CE_err.py (reproduce Fig. 8)
    │   │   │   └── some .mat files
    │   │   └── plot_quant/
    │   │       ├── plot_quant.py (reproduce Fig. 9)
    │   │       └── some .mat files
    │   └── saved_model/
    │       └── ... (models will be saved here once trained)
    └── downlink/
        └── ... (similar to uplink)

