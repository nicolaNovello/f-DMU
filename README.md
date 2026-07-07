<div align="center">
   
# A Unified Framework for Diffusion Model Unlearning with f-Divergence

[Nicola Novello](https://scholar.google.com/citations?user=4PPM0GkAAAAJ&hl=en), [Federico Fontana](https://scholar.google.com/citations?user=5W5RkQgAAAAJ&hl=en), [Luigi Cinque](https://scholar.google.com/citations?user=sDe4b9UAAAAJ&hl=en), [Deniz Gündüz](https://scholar.google.com/citations?user=MbmKROkAAAAJ&hl=en), [Andrea M. Tonello](https://scholar.google.com/citations?user=qBiseEsAAAAJ&hl=en)<br />

</div>

Official repository of the paper "A Unified Framework for Diffusion Model Unlearning with f-Divergence" published at ICML 2026.

> $f$-DMU is a unified framework for Diffusion Model Unlearning based on $f$-divergence. It comprises two classes of objective functions: i) ''closed-form losses'' (best choice for most scenarios) characterized by a good erasure-preservation trade-off; ii) ''variational losses'' that lead to a more aggressive erasure.

<div align="center">

[![license](https://img.shields.io/badge/License-MIT-red.svg)](https://github.com/nicolaNovello/f-DMU/blob/main/LICENSE)
[![Hits](https://hits.sh/github.com/nicolaNovello/f-DMU.svg?label=Visitors&color=30a704)](https://hits.sh/github.com/nicolaNovello/f-DMU/)

</div>

---

# 💻 How to run the code

The script 'run_experiment.sh' runs the code to erase Van Gogh's artistic style.

[Additional information and scripts will be uploaded soon...]

For SD v2.1, we used the original link to download the model and run all experiments. Now, the original repository is not available anymore, but it is possible to use the community version of the same model from [https://huggingface.co/sd2-community/stable-diffusion-2-1](https://huggingface.co/sd2-community/stable-diffusion-2-1).

## Closed-Form Losses

'run_experiment.sh' runs the experiment for MSE, squared Hellinger distance, and Pearson $\chi^2$ divergence.

## Variational Losses

Coming soon...

---

# 🤓 Guidelines

$f$-DMU comprises two classes of objective functions: i) ''closed-form losses'', which are derived from the closed-form expression of the $f$-divergence between Gaussian probability density functions (i.e., the pdf describing the denoising process of DMs); ii) ''variational losses'', which are derived from the variational representation of the $f$-divergence between the same Gaussian probability density functions. We provide the guidelines in the choice of the loss below.

## Closed-Form Losses

The $f$-DMU closed-form losses are the best choice in most scenarios, as they have a better erasure-preservation trade-off. Furthermore, these losses exactly correspond to the minimization of the $f$-divergence between two Gaussian pdfs for any batch size. However, not all $f$-divergences lead to a closed-form loss.

### H-DMU
Using the squared Hellinger distance, the loss becomes    

$$ \min_{\hat{\Phi}} \mathbb{E}_{\mathbf{x}, \mathbf{c}^*, \mathbf{c}, t} \Bigl[ - \omega_t \exp \Bigl( -\lVert \Phi(\mathbf{x} _{t}, \mathbf{c}, t) - \hat{\Phi}(\mathbf{x} _{t}, \mathbf{c}^{\star}, t) \rVert_2^2 \Bigr) \Bigr].$$

### P-DMU
Using the $\chi^2$ divergence, the loss becomes   

$$ \min_{\hat{\Phi}} \mathbb{E}_{\mathbf{x}, \mathbf{c}^*, \mathbf{c}, t} \Bigl[ \omega_t \exp \Bigl( \lVert \Phi(\mathbf{x} _{t}, \mathbf{c}, t) - \hat{\Phi}(\mathbf{x} _{t}, \mathbf{c}^{\star}, t)\rVert_2^2 \Bigr) \Bigr].$$


## Variational Losses

The $f$-DMU variational losses lead to an aggressive unlearning. The minimization of these losses does not exactly correspond to the minimization of the $f$-divergence for small batch sizes, as the variational representation of the $f$-divergence yields an estimate of such divergence, which is more accurate when the batch size is larger. Any $f$-divergence can lead to a variational loss.  

For any $f$-divergence, the objective function becomes   

$$\min_{\hat{\Phi}} \max_T  \mathbb{E}_{\mathbf{x}, \mathbf{c}^{\star}, \mathbf{c}, t} \Bigl[ \mathbb{E}_{p_{\Phi}(\mathbf{x} _{t-1}|\mathbf{x} _{t},\mathbf{c})} \Bigl[ T(\Phi) \Bigr] - \mathbb{E}_{p_{\hat{\Phi}}(\mathbf{x} _{t-1}|\mathbf{x} _{t},\mathbf{c}^{\star})} \Bigl[ f^{\star}(T(\hat{\Phi})) \Bigr] \Bigr].$$

---

## 📝 Reference 

If you use the code for your research, please cite our paper:
```bibtex
@article{novello2026unified,
  title={A Unified Framework for Diffusion Model Unlearning with f-Divergence},
  author={Novello, Nicola and Fontana, Federico and Cinque, Luigi and Gunduz, Deniz and Tonello, Andrea M},
  journal={International Conference on Machine Learning},
  year={2026}
}

```

## 📋 Acknowledgments

The implementation is based on / inspired by:

- [https://github.com/nupurkmr9/concept-ablation](https://github.com/nupurkmr9/concept-ablation)
- [https://github.com/yongliang-wu/DoCo](https://github.com/yongliang-wu/DoCo)


---

## 📧 Contact

[nicola.novello@aau.at](nicola.novello@aau.at)
