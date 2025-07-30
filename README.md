## How Diffusion Prior Landscapes Shape the Posterior in Blind Deconvolution <br><sub>Code for reproducing numerical experiments</sub>

## Requirements
- `pytorch` (if you're using `cuda`, please make sure that your cuda runtime version matches your pytorch cuda verion, since the score model need to be compiled from cpp source)
- `deepinv` (https://deepinv.github.io/deepinv/)

## Getting started
The codebase is based on previous exellent repositories: https://github.com/yang-song/score_sde_pytorch and  https://github.com/NVlabs/edm/tree/main) 
- The notebook `sampling_and_compute_potential.ipynb` shows how to sample from diffusion models and how to evaluate the potential of a given image. It also shows that blurry images are more likely.
- The notebook `eigenvalues.ipynb` shows how to compute the spectra of the diffusion prior, by leveraging automatic differentiation, giving an estimation on the instrinsic dimension of the image manifold.
- The notebook `blind_deblurring.ipynb` shows the proposed initialization and optimization strategy for solving blind deblurring problem by mimimizing the posterior.
