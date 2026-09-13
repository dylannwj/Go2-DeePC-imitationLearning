# Release environment

The frozen Genesis evidence records the following runtime:

- Python `3.12.13`
- Genesis `1.3.1`
- NumPy `2.4.4`
- PyTorch `2.10.0+cpu`
- CVXPY `1.8.2`
- OSQP `1.1.1`
- Quadrants `(1, 2, 0)`

Install the Python dependencies from `requirements.txt` in a Python 3.12
environment. Install the matching CPU PyTorch wheel and Genesis 1.3.1 using
the supported package source for the platform.

Genesis supplies the Go2 and plane URDFs and meshes at runtime. The freeze
records the expected URDF hashes; verify those files before running the
benchmark. The release does not vendor a second copy of the Genesis asset
tree.

The physical branch is an external deployment: it requires ROS 2 with
`rclpy`, `geometry_msgs`, and `std_msgs`, a licensed Gurobi installation, and
the `/mocap/go2_pose` publisher. The expert imitation pipeline additionally
requires the separately distributed/licensed expert checkpoint and external
teacher scene assets.

The processed imitation dataset is a GitHub Release asset, not a normal Git
file. Place it at `data/imitation/expert_imitation_dataset_v1.npz` and verify
the SHA-256 documented in `data/imitation/README.md` before running BC tools.
