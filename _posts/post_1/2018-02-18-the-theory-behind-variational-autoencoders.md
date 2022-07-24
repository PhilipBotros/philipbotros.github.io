---
layout: post
title: "The theory behind Variational Autoencoders"
date: 2018-02-18
cover: sample_100.png
mathjax: true
excerpt: <br>In most classic Machine Learning problems we are interested in learning a mapping from the input data to a label, more recently however, a lot of interest has sparked in the field of generative modelling. We will look at one of the most popular models in depth, the Variational Autoencoder. 
background: ../../../cover2.png
--- 
In most classic Machine Learning problems we are interested in learning a mapping from the input data to a label, more recently however, a lot of interest has sparked in the field of generative modelling. While classification is concerned with learning a conditional distribution $p(\mathbf{y}|\mathbf{x})$,  generative modelling has as a goal to directly learn the data distribution $p(\mathbf{x})$ such that we can freely reproduce this data. In a world where unlabeled data is ubiquitous and labeled data is sparse/expensive, this seems to be an area of great importance.
<h2>Probabilistic perspective</h2>
We have to take a slight detour to arrive at the objective function of the VAE since it has strong theoretical foundations in Bayesian statistics while we can also look at the auto encoding perspective. Furthermore, a quick connection with the classic wake-sleep algorithm is given.

<h2>Latent variable models</h2>
We first start with assuming that some latent factors are responsible for the data since optimizing $p(\mathbf{x})$ with respect to our parameters $\theta$ is difficult,  the sum rule of probability states that we can rewrite our data distribution as:

$$ \begin{aligned}
p(\mathbf{x}) &= \int p(\mathbf{x}, \mathbf{z}) d\mathbf{z} \\
&=\int p(\mathbf{z}) p(\mathbf{x}| \mathbf{z}) d\mathbf{z}
\end{aligned}
$$ 

Where the last step follows from the product rule of probability. So in essence, we have restated the problem in a way that makes it more tractable to optimize $p(\mathbf{x})$ by defining a joint distribution $p(\mathbf{x}, \mathbf{z})$ and then marginalizing over our latent variables $\mathbf{z}$. 

We now turn to Bayes' Theorem to find an expression for our latent variables $\mathbf{z}$ since we need a way to compute this given the original input $\mathbf{x}$:

$$ \begin{aligned}
p(\mathbf{z}|\mathbf{x}) &= \frac{p(\mathbf{z})p(\mathbf{x}| \mathbf{z})}{p(\mathbf{x})}\\
&= \frac{p(\mathbf{z})p(\mathbf{x}| \mathbf{z})}{\int p(\mathbf{z}) p(\mathbf{x}| \mathbf{z}) d\mathbf{z}}
\end{aligned}$$ 

Unfortunately, this integral over $\mathbf{z}$ is in many practical applications intractable since we have to integrate over all latent dimensions for every data point.
<h2>Inference in graphical models</h2>
Luckily we can approximate the posterior $p(\mathbf{z}|\mathbf{x})$ using techniques from variational inference. Variational inference is a deterministic analytic approximation method that can be used to efficiently approximate intractable posterior distributions, as opposed to stochastic sampling methods like MCMC it will not converge in the asymptotic case to the true distribution but we accept this approximate distribution since the optimization is reasonably quick.

It works by assuming a parameterized distribution Q that we want to be as close as possible to the actual posterior P by defining a particular distance over these two distributions,  usually the KL-divergence:

$$D_{KL}(P||Q) = \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})}d\mathbf{x}$$

Note that in most cases, recovering the actual posterior is infeasible since we limit ourself to known parameterized distributions which in most cases are not flexible enough to cover the actual posterior. 

The KL-divergence $($also called relative entropy$)$ is a measure from information theory which measures how much one distribution diverges from another distribution.  We can interpret it in the variational case as the amount of information that is lost by using the approximate distribution Q instead of the true distribution P. Note that it is not a distance metric in the pure sense since it is not symmetric.

<h2>Objective function</h2>
We can derive the objective function by looking at the properties of the KL-divergence and rearranging terms. First note that the KL-divergence is lower bounded by zero and this lower bound is reached if and only if the two distributions are equal, starting from the definition of the KL-divergence we can derive:

$$ \begin{aligned}
D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}|\mathbf{x})\big] &= \int q(\mathbf{z}|\mathbf{x}) \log \frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})}d\mathbf{z} \\
&= \int q(\mathbf{z}|\mathbf{x}) \log q(\mathbf{z}|\mathbf{x})d\mathbf{z}  - \int q(\mathbf{z}|\mathbf{x}) \log \frac{p(\mathbf{z}) p(\mathbf{x}| \mathbf{z})}{p(\mathbf{x})}d\mathbf{z} \\
&= \int q(\mathbf{z}|\mathbf{x}) \log \frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z})}d\mathbf{z}  - \int q(\mathbf{z}|\mathbf{x}) \log p(\mathbf{x}| \mathbf{z})d\mathbf{z} + \log p(\mathbf{x}) \\
&= D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z})\big] - \mathbb{E}_{q\mathbf{(z|x})}\big[\log p(\mathbf{x}| \mathbf{z})\big] + \log p(\mathbf{x}) \\
\log p(\mathbf{x}) - D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}|\mathbf{x})\big]&= \mathbb{E}_{q\mathbf{(z|x})}\big[\log p(\mathbf{x}| \mathbf{z})\big] -  D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z})\big] \\
\log p(\mathbf{x}) &\geq \mathbb{E}_{q\mathbf{(z|x})}\big[\log p(\mathbf{x}| \mathbf{z})\big] -  D_{KL}\big[q(\mathbf{z}|\mathbf{x})||p(\mathbf{z})\big] 
\end{aligned}
$$ 

Where we have used the fact that $D_{KL}\geq 0$ to derive the lower bound on the log probability. The right hand side is usually called the ELBO $($Evidence Lower BOund$)$ since the log probability is also called the evidence. 

Since both of the terms of the ELBO are computable we have succesfully rewritten the intractable posterior in a way we can approximate it up to a nonnegative KL divergence and turned it into an optimization problem.

<h2>Encoder-Decoder</h2>
A quick recap on vamilla auto-encoders can provide a different insight into the workings of the VAE. Remember that a normal auto-encoder works by transforming an input $\mathbf{x}$ into a latent code $\mathbf{z}$ after which it then decodes the latent code $\mathbf{z}$ back to the input space denoted by the reconstructed input $\mathbf{x}'$. The encoder is here defined as a nonlinear transformation $f_\theta(\mathbf{x})$ and the decoder as $g_\phi(\mathbf{z})$. By putting some constraints on the transformations that can be made or by adding noise we can force the network to learn meaningful hidden representations.  After reconstructing the input we can update the model by, for instance,  optimizing with respect to the squared Euclidean loss $($if we are talking about images$)$:

$$ \displaystyle\min_{\theta, \phi } ||\mathbf{x} - \mathbf{x'}||^2 + \lambda \Omega$$

Where $\lambda \Omega$ is any arbitrary regularization penalty.
Although standard auto encoders are very useful for tasks like compression or image denoising, we can not sample from them to generate new examples, for this we need a probabilistic auto encoder.
<h2>Putting it together</h2>
Finally we are getting to the crux of the matter, we know that we need to learn a probabilistic mapping $p_{enc}(\mathbf{z|x})$, i.e. infer the posterior distribution over the latent variables, and a decoder distribution $p_{dec}(\mathbf{x|z})$, our generative model. Remember that exact inference of the posterior $p_{enc}(\mathbf{z|x})$ is intractable and we use variational methods to approximate the distribution $q_{\phi}(\mathbf{z|x})$.
A neural network is used for both, to learn the parameters of the variational posterior $q_\phi(\mathbf{z|x})$ and the parameters of the generator network $p\mathbf{_\theta(x|z)}$. The encoder-decoder structure of the VAE works by jointly optimizing the variational lower bound with respect to both the variational parameters $\phi$ and generative parameters $\theta$:

$$ \mathcal{L}(\mathbf{\theta, \phi; x}^{(i)}) = \mathbb{E}_{q_\phi\mathbf{(z|x}^{(i)})}\big[\log p\mathbf{_\theta(x}^{(i)}|\mathbf{z)}\big]-D_{KL}(q_\phi(\mathbf{z|x}^{(i)})||p{\mathbf{_\theta(z))}} $$

Where the first term is simply the reconstruction error of the model with respect to the original input and the second term acts as a regularizer by keeping the latent space close to the prior of, usually, a standard normal distribution. How does this regularization work? Well by constraining our posterior to be close to a standard normal we force the model to obtain generalize mappings for any given data point. If we would allow our model to take on any arbitrary $($complicated$)$ distribution it could easily encode and decode our input to minimize the reconstruction error but we would fail to learn any generalizing properties of the distribution.

<h2>Reparameterization trick</h2>
Before we can actually perform stochastic gradient descent techniques note that we have stochastic nodes in the network. How do we backpropagate through these stochastic nodes? Since the gradient will in essence also be a random variable standard backpropagation techniques are unsuitable. We can, however, move the stochasticity out of the model by reparameterizing such that we have an auxiliary variable that incorporates the stochasticity. By doing this we can take gradients with respect to our parameters and can use standard backpropagation to train our model end-to-end. How does this work? Its quite simple in the end, imagine that our latent space is Gaussian distributed, i.e. $z \sim \mathcal{N}(\mu, \sigma^2)$ then we can also write:

$$ z = \mu + \sigma \epsilon \text{ with }\epsilon \sim \mathcal{N}(0,1) $$ 

Then by moving the sampling to the input layer we create a deterministic mapping and obtain a Monte Carlo estimate which is differentiable with respect to the model parameters:

$$ \mathbb{E}_{\mathcal{N}(\epsilon;0,1)}[f(\mu + \sigma \epsilon)] \approx \frac{1}{M} \sum^M_{m=1}f(\mu + \sigma \epsilon^{(m)}) $$

Note that reparameterization tricks for discrete random variables have also been developed recently such that we can freely use discrete nodes in VAE architectures.

<h2>Choice of distributions</h2>
What kind of distribution do we choose as a prior and/or constrain our variational posterior to be? It has to be a distribution where we can easily sample from and since we need to compute the KL-divergence for every produced code of our data examples in our training set with respect to the prior it would be beneficial to perform this integration in closed form. If we take two Gaussians for instance we can simply compute:

$$  \frac{(\mu_1 - \mu_2)^2 + \sigma_1^2 } {2 \sigma_2^2} + \log\big(\frac{\sigma_2}{ \sigma_1}\big) - \frac{1}{2} $$

If we then use the fact that our prior is a standard Gaussian this reduces to:

$$\frac{1}{2}\big((\mu_1)^2 + \sigma_1^2 -\log(\sigma_1^2) - 1\big)$$

Note that in the case of the VAE we are summing over $j$ latent dimensions. Since we also restrict our posterior to have a diagonal covariance we can simply compute it for the multivariate case by summing over the univariate Gaussians:

$$\frac{1}{2}\sum^J_{j=1}\big((\mu_j)^2 + \sigma_j^2 -\log(\sigma_j^2) - 1\big)$$

Another advantage from choosing our distribution to be Gaussian is that is the maximum entropy distribution for continuous variables, i.e. it makes the least prior assumptions about the data we are modeling.

Interestingly, more recent it has been shown that it also efficient to approximate the KL divergence with respect to a single batch by taking a Monte Carlo sample, this method is of course useful when there is no analytical solution possible:

$$   \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})}d\mathbf{x} \approx \frac{1}{M}\sum^M_{m=1}\log p(\mathbf{x_m}) - \log q(\mathbf{x_m})$$

<h2>Visualizing the manifold</h2>
Observe how the latent manifold changes from random noise to a meaningful representation of MNIST over time:<br><br>

{:refdef: style="text-align: center;"}
![My helpful screenshot]({{ "../../../giff.gif"}})
{: refdef}
<br>
<h2>Improving the flexibility of the posterior</h2>
Since we restrict our posterior to be unimodal in the case of a standard normal prior, a lot of work has been applied to improving the flexibility of the posterior. This can either be done by transforming the posterior by inverse autoregressive flows, a series of simple transformations that iteratively transform a simple distribution into a more complex one. Another option is to improve the flexibility of the prior directly, i.e. by choosing a multimodal prior.
<h2>Connections to wake-sleep</h2>
Another interesting point of view to look at is to look at the relation between the wake-sleep algorithm and VAE's. In essence, both are algorithms that perform inference on $($assumed$)$ intractable posteriors in deep graphical models. 

Wake-sleep however, consists of two phases, the 'wake' phase and 'sleep' phase. We optimize the generative model in the wake phase, where we try to reconstruct the inferred hidden representation from the actual data:

$$\nabla_\theta \cdot \mathbb{E}_{p_{data}(\mathbf{x})q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] $$

In the sleep phase however, we 'dream' up data, i.e. sample from our prior, and use these samples to optimize the inference model:

$$ \nabla_\phi \cdot \mathbb{E}_{p(\mathbf{z})p_\theta(\mathbf{x}|\mathbf{z})}[\log q_\phi(\mathbf{z}|\mathbf{x})] $$

Note that, as with GANs, we here use samples from the model distribution to train the model, which is not used in VAEs.

Key difference between the two is that we can jointly optimize the inference and model parameters using the reparameterization trick in the case of VAE's, this joint optimization in turn allows for proper optimization of the ELBO.
