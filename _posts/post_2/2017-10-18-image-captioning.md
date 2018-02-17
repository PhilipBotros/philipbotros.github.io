---
layout: post
title: "Variational autoencoders"
date: 2018-02-16
cover: manifold.png
background: ../../../imcap_bg.png
---

### Generative Modelling
In most classic Machine Learning problems we are interested in learning a mapping from the input data to a label, more recently however, a lot of interest has sparked in the field of generative modelling.  While classification is concerned with learning a conditional distribution $p(\mathbf{y}|\mathbf{x})$,  generative modelling has as a goal to directly learn the data distribution $p(\mathbf{x})$ such that we can freely reproduce this data. In a world where unlabeled data is ubiquitous and labeled data is sparse/expensive, this seems to be an area of great importance.
####  Probabilistic perspective
We have to take a slight detour to arrive at the objective function of the VAE since it has strong theoretical foundations in Bayesian statistics and we can also look at the auto encoding perspective REWRITE. To start, we have to figure out what variational inference is, why we use a neural network to approximate the parameters and what constraints we put on the model to make it scalable.

##### LATENT VARIABLE MODELS
We first start with assuming that a latent variable model is responsible for the data since optimizing $p(\mathbf{x})$ with respect to our parameters $\theta$ is difficult,  the sum rule of probability states that we can rewrite our data distribution as:
$$ \begin{aligned}
p(\mathbf{x}) &= \int p(\mathbf{x}, \mathbf{z}) d\mathbf{z} \\
&=\int p(\mathbf{z}) p(\mathbf{x}| \mathbf{z}) d\mathbf{z}
\end{aligned}
 $$ 
Where the last step follows from the product rule of probability. So in essence, we have restated the problem in a way that makes it more tractable to optimize $p(\mathbf{x})$ by defining a joint distribution $p(\mathbf{x}, \mathbf{z})$ and then marginalizing over our latent variables $\mathbf{z}$. 

We now turn to Bayes' Theorem to find an expression for our latent variables $\mathbf{z}$ since we need a way to compute this given the original input $\mathbf{x}$ CHECK:
$$ \begin{aligned}
p(\mathbf{z}|\mathbf{x}) &= \frac{p(\mathbf{z})p(\mathbf{x}| \mathbf{z})}{p(\mathbf{x})}\\
&= \frac{p(\mathbf{z})p(\mathbf{x}| \mathbf{z})}{\int p(\mathbf{z}) p(\mathbf{x}| \mathbf{z}) d\mathbf{z}}
\end{aligned}
 $$ 
Unfortunately, this integral over $\mathbf{z}$ is in many practical applications intractable since we have to integrate over all latent dimensions for every data point.
##### VARIATIONAL INFERENCE
Luckily we can approximate the posterior $p(\mathbf{z}|\mathbf{x})$ using variational inference. Variational inference is a deterministic analytic approximation method that can be used to efficiently approximate intractable posterior distributions, as opposed to stochastic sampling methods like MCMC it will not converge in the asymptotic case to the true distribution but we accept this approximate distribution as 'good enough' since it scales well to higher dimensions and high volumes of data.

It works by assuming a parameterized distributed Q that we know to be as close as possible to the actual posterior P by defining a particular distance over these two distributions,  usually the KL-divergence:
$$ D_{KL}(P||Q) = \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})}d\mathbf{x} $$
The KL-divergence (also called relative entropy) is a measure from information theory which measures how much one distribution diverges from another distribution.  We can interpret it in the variational case as the amount of information that is lost by using the approximate distribution Q instead of the true distribution P. Note that it is not a distance metric in the pure sense since it is not symmetric.

##### OBJECTIVE FUNCTION
We can derive the objective function by looking at the properties of the KL-divergence and rearranging terms. First note that the KL-divergence is lower bounded by zero and this lower bound is reached if and only if the two distributions are equal,  starting from the definition of the KL-divergence we can derive:
$$ \begin{aligned}
D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}|\mathbf{x})\big] &= \int q(\mathbf{z}|\mathbf{x}) \log \frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})}d\mathbf{z} \\
&= \int q(\mathbf{z}|\mathbf{x}) \log q(\mathbf{z}|\mathbf{x})d\mathbf{z}  - \int q(\mathbf{z}|\mathbf{x}) \log \frac{p(\mathbf{z}) p(\mathbf{x}| \mathbf{z})}{p(\mathbf{x})}d\mathbf{z} \\
&= \int q(\mathbf{z}|\mathbf{x}) \log \frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})}d\mathbf{z}  - \int q(\mathbf{z}|\mathbf{x}) \log p(\mathbf{x}| \mathbf{z})d\mathbf{z} + p(\mathbf{x}) \\
&= D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z})\big] - \mathbb{E}_{q\mathbf{(z|x})}\big[\log p(\mathbf{x}| \mathbf{z})\big] + p(\mathbf{x}) \\
p(\mathbf{x}) - D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}|\mathbf{x})\big]&= \mathbb{E}_{q\mathbf{(z|x})}\big[\log p(\mathbf{x}| \mathbf{z})\big] -  D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z})\big] \\
p(\mathbf{x}) &\geq \mathbb{E}_{q\mathbf{(z|x})}\big[\log p(\mathbf{x}| \mathbf{z})\big] -  D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z})\big] 
\end{aligned}
 $$ 
 Where we have used the fact that $D_{KL}\geq 0$ to derive the lower bound on the log probability. The right hand side is usually called the ELBO (Evidence Lower BOund) since the log probability is also called the evidence. 

Since both of the terms of the ELBO are computable we have succesfully rewritten the intractable posterior in a way we can approximate it up to a nonnegative KL distance and turned it into an optimization problem.

#### Neural networks
Now the word optimization problem has fallen, we instantly think neural networks, since they are unrivaled as universal function approximators at the moment. We start by briefly looking at the non probabilistic auto-encoders.
##### ENCODER - DECOCER
A normal auto-encoder works by transforming an input $\mathbf{x}$ into a latent code $\mathbf{z}$ after which it then decodes the latent code $\mathbf{z}$ back to the input space denoted by the reconstructed input $\mathbf{x}'$.  The encoder is here defined as a nonlinear transformation $f_\theta(\mathbf{x})$ and the decoder as $g_\phi(\mathbf{z})$. By putting some constraints on the transformations that can be made or by adding noise we can force the network to learn meaningful hidden representations.  After reconstructing the input we can update the model by, for instance,  optimizing with respect to the squared Euclidean loss (if we are talking about images):
$$ \displaystyle\min_{\theta, \phi } ||\mathbf{x} - \mathbf{x'}||^2 + \lambda \Omega$$
Where $\lambda \Omega$ is any arbitrary regularization penalty.
Although standard auto encoders are very useful for tasks like compression or image denoising, we can not sample from them to generate new examples, for this we need a probabilistic auto encoder.

##### PUTTING IT TOGETHER
Finally we are getting to the crux of the matter, we know if we want to sample from the model that we need to learn a probabilistic mapping $p_{encoder}(\mathbf{z|x})$ and a decoder distribution $p_{decoder}(\mathbf{x|z})$. From the probabilistic perspective we remember that the encoder distribution $p_{encoder}(\mathbf{z|x})$ is intractable and we can use variational inference to approximate this distribution denoted by $q_{\phi}(\mathbf{z|x})$. From the probabilistic latent space we still need to map back the input space to generate real samples, this generator network is denoted by $p\mathbf{_\theta(x|z)}$.
A neural network is used for both, to learn the parameters of the variational posterior $q_\phi(\mathbf{z|x})$ and the parameters of the generator network $p\mathbf{_\theta(x|z)}$. The encoder-decoder structure of the VAE works by jointly optimizing the variational lower bound with respect to both the variational parameters $\phi$ and generative parameters $\theta$:
$$ \mathcal{L}(\mathbf{\theta, \phi; x}^{(i)}) = \mathbb{E}_{q_\phi\mathbf{(z|x}^{(i)})}\big[\log p\mathbf{_\theta(x}^{(i)}|\mathbf{z)}\big]-D_{KL}(q_\phi(\mathbf{z|x}^{(i)})||p{\mathbf{_\theta(z))}} $$
Where the first term is simply the reconstruction error of the model with respect to the original input and the second term acts as a regularizer by keeping the latent space close to the prior of, usually, a standard normal distribution. 
##### NORMAL DISTRIBUTION
What kind of distribution do we choose as a prior and/or constrain our variational posterior to be? It has to be a distribution that is easy to sample from and since we need to compute the KL-divergence for every produced code of our data examples in our training set with respect to the prior it would be beneficial to perform this integration in closed form. If we take two Gaussians for instance we can simply compute:
$$  \frac{(\mu_1 - \mu_2)^2 + \sigma_1^2 } {2 \sigma_2^2} + \log\big(\frac{\sigma_2}{ \sigma_1}\big) - \frac{1}{2} $$
If we then use the fact that our prior is a standard Gaussian this reduces to:
$$\frac{1}{2}\big((\mu_1)^2 + \sigma_1^2 -\log(\sigma_1^2) - 1\big)$$
Note that in the case of the VAE we are summing over $j$ latent dimensions. Since we also restrict our posterior to have a diagonal covariance we can simply compute it for the multivariate case by summing over the univariate Gaussians:
$$\frac{1}{2}\sum^J_{j=1}\big((\mu_j)^2 + \sigma_j^2 -\log(\sigma_j^2) - 1\big)$$
Interestingly, more recent it has been shown that it also efficient to approximate the KL divergence with respect to a single batch by taking a Monte Carlo sample, this method is ofcourse useful when there is no analytical solution possible:
$$   \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})}d\mathbf{x} \approx \frac{1}{N}\sum^N_{n=1}\log p(\mathbf{x_n}) - \log q(\mathbf{x_n})$$

##### EXPERIMENTS
Random samples
Conditional
Manifold
Reconstruction
##### PROBLEMS
- Maximum likelihood problems
	Compared to the other huge generative model at the moment (GAN) we can observe that the samples produced are slightly blurrier. This can be attributed to the use of maximum likelihood as a loss function. In essence, ML is unable to perfectly reproduce data that is higher dimensional than unimodal. This can be seen by looking at a bimodal Gaussian, by taking the KL-divergence (ML approach) between the proposal distribution and the actual distribution we will end up between the two modes.
- Diagonal covariance posterior
		Mostly for computational convenience, we restrict our Gaussians to have a diagonal covariance. This is ofcourse a restrictive assumption which disables our posterior to have very flexible functional forms. 
-  Standard normal prior
	We can also look at a more flexible prior to make our distributions less restrictive.
- Strong decoders
	If our decoders have a strong functional form (a RNN) it would be able to 'encode' all of the information in the decoder and the encoder would be unused.

MEAN FIELD VARIATIONAL INFERENCE NOT SCALABLE
##### REFERENCES