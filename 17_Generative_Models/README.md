# Generative Models (GAN / VAE / Diffusion) — Interview Questions & Answers

---

## Small Dataset Example

```
Toy 2D dataset for generative modeling (points on a 2D plane):

Real samples (x₁, x₂):
(0.5, 0.8), (0.3, 0.7), (0.7, 0.6), (0.4, 0.9),
(0.6, 0.7), (0.2, 0.8), (0.8, 0.5), (0.5, 0.6)

These form a "banana-shaped" cluster in 2D space.
Goal: Train a generative model to produce new (x₁, x₂) pairs
      that look like they came from this distribution.
```

### Dataset Questions

**Q: A GAN is trained on this 2D dataset. What does the Generator do? What does the Discriminator do?**
> **A:** The **Generator** takes a random noise vector z ~ N(0,1) and outputs a 2D point (x₁, x₂). It learns to map the simple noise distribution to the complex 2D data distribution. The **Discriminator** takes a 2D point and outputs a probability that it's real (from data) vs fake (from Generator). They play a minimax game — Generator tries to fool Discriminator; Discriminator tries to tell them apart.

**Q: A VAE encodes each data point to a latent space. For the point (0.5, 0.8), the encoder outputs μ=0.3, σ=0.1. What latent vector z is sampled?**
> **A:** z = μ + σ × ε, where ε ~ N(0,1). So z = 0.3 + 0.1 × ε. For ε=0.5: z = 0.35. The reparameterization trick makes this differentiable by separating the random part (ε) from the learnable parameters (μ, σ).

**Q: Why would a diffusion model work better than a GAN for this 2D distribution?**
> **A:** For simple 2D data both work, but conceptually: diffusion models are more stable (no adversarial training, no mode collapse), produce diverse samples, and scale better to complex distributions. GANs may suffer mode collapse here — only generating from the dense center instead of the full banana shape.

---

## Questions & Answers

---

### Q1. What is a Generative Model?

**A:** A generative model learns the underlying data distribution P(X) and can generate new samples that look like the training data. Two main types:
- **Explicit density models:** Directly model P(X) — VAEs, Normalizing Flows, Diffusion Models
- **Implicit density models:** Learn to sample without explicit P(X) — GANs

Applications: Image synthesis, text generation, music, drug discovery, data augmentation.

---

### Q2. What is a GAN?

**A:** A Generative Adversarial Network (GAN) consists of two networks in a minimax game:

- **Generator G:** Takes noise z ~ P(z) → generates fake samples G(z)
- **Discriminator D:** Takes input x → outputs P(real | x)

**Objective:**
```
min_G max_D E[log D(x)] + E[log(1 − D(G(z)))]
```

Generator wants to fool Discriminator (make D output high probability for fakes). Discriminator wants to correctly classify real vs fake.

---

### Q3. What is mode collapse in GANs?

**A:** Mode collapse occurs when the Generator produces only a few types of outputs, ignoring most of the data distribution. Example: a face GAN only generates one face type instead of diverse faces.

**Cause:** Generator finds a small region that fools the Discriminator and exploits it rather than learning the full distribution.

**Solutions:**
- **WGAN:** Use Wasserstein distance instead of JS divergence
- **Minibatch discrimination:** Let Discriminator see multiple samples together
- **Spectral normalization:** Normalize Discriminator weights
- **Unrolled GAN:** Unroll Discriminator updates for Generator training

---

### Q4. What is WGAN?

**A:** Wasserstein GAN replaces the standard discriminator with a **critic** that estimates the Wasserstein (Earth Mover's) distance between real and fake distributions:

```
W(P_r, P_g) = max_{||f||_L ≤ 1} E[f(x)] − E[f(G(z))]
```

- Critic is not bounded to (0,1) — no sigmoid at output
- Weight clipping or gradient penalty to enforce Lipschitz constraint
- **WGAN-GP** (gradient penalty): More stable training than weight clipping

**Advantage:** Provides meaningful gradient signal even when distributions don't overlap → no mode collapse, stable training.

---

### Q5. What is a VAE?

**A:** Variational Autoencoder (VAE) learns a probabilistic latent space. It has:

**Encoder (inference network):** `q(z|x)` — maps input x to distribution over latent z:
- Outputs μ and σ (parameters of a Gaussian)

**Decoder (generative network):** `p(x|z)` — maps latent z to reconstructed x

**Training:** Maximize the Evidence Lower Bound (ELBO):
```
ELBO = E[log p(x|z)] − KL(q(z|x) || p(z))
      = Reconstruction loss − KL divergence
```

---

### Q6. What is the ELBO?

**A:** The Evidence Lower Bound (ELBO) is the VAE training objective:
```
ELBO = E[log p(x|z)] − KL(q(z|x) || N(0,I))
```

**Term 1 — Reconstruction loss:** How well the decoder reconstructs x from sampled z (maximize)

**Term 2 — KL divergence:** Forces posterior q(z|x) to stay close to standard Gaussian prior N(0,1) (minimize). This regularizes the latent space.

**Tradeoff:** High β in β-VAE: stronger KL → smoother, more structured latent space but worse reconstruction.

---

### Q7. What is the reparameterization trick?

**A:** VAE samples z ~ q(z|x) = N(μ, σ²). Direct sampling is not differentiable (can't backpropagate through random operations).

**Trick:** Reparameterize:
```
z = μ + σ × ε,  where ε ~ N(0, 1)
```

Now: ε is fixed random noise; μ and σ are learnable. Gradients flow through μ and σ. The randomness (ε) is separated from the parameters.

This is the key innovation that makes VAEs trainable end-to-end.

---

### Q8. What is the difference between GAN and VAE?

**A:**
| | GAN | VAE |
|---|---|---|
| Output quality | Sharper, higher quality | Blurrier (reconstruction loss) |
| Training stability | Unstable (adversarial) | Stable |
| Mode coverage | Mode collapse risk | Good coverage |
| Latent space | Not structured | Smooth, interpolatable |
| Inference | Fast (one pass) | Fast |
| Evaluation | Hard (FID, IS) | Can compute ELBO |
| Applications | Image synthesis | Generation + compression |

---

### Q9. What is a Diffusion Model?

**A:** Diffusion models learn to reverse a noise process. Two processes:

**Forward (fixed):** Gradually add Gaussian noise to data over T steps until it's pure noise:
```
q(xₜ|xₜ₋₁) = N(xₜ; √(1−β_t)·xₜ₋₁, β_t·I)
```

**Reverse (learned):** Train a neural network to denoise step by step:
```
p_θ(xₜ₋₁|xₜ) = N(xₜ₋₁; μ_θ(xₜ,t), Σ_θ(xₜ,t))
```

Generation: Start from random noise, iteratively denoise T times → clean image.

---

### Q10. What is DDPM?

**A:** Denoising Diffusion Probabilistic Models (Ho et al., 2020) is the foundational diffusion model paper:

- Train a U-Net to predict the noise ε added at each step
- Loss: `L = E[||ε − ε_θ(xₜ, t)||²]` (simple MSE on noise prediction)
- Generation: T=1000 denoising steps (slow)

**Why it works:** Each denoising step is a small correction — easier to learn than one-shot generation (VAE/GAN).

---

### Q11. What is DDIM sampling?

**A:** DDIM (Denoising Diffusion Implicit Models) enables faster sampling by using a non-Markovian denoising process:
- Can generate with 50–250 steps instead of 1000 (DDPM)
- Deterministic (no random noise in each step)
- Same model as DDPM — only the sampling procedure changes

```
50 steps with DDIM ≈ quality of 1000 steps with DDPM
```

---

### Q12. What is classifier-free guidance?

**A:** Classifier-free guidance (CFG) generates conditioned samples (e.g., text → image) with controllable fidelity vs. diversity:

```
ε_guided = (1 + w) × ε_cond − w × ε_uncond
```

- `w=0`: No guidance (diverse but potentially off-prompt)
- `w=7.5`: Typical value (good balance)
- `w=15+`: Very faithful to prompt but less diverse

Used in Stable Diffusion, DALL-E, Midjourney.

---

### Q13. What is a latent diffusion model (LDM)?

**A:** Standard diffusion models work in pixel space (expensive). LDMs (Stable Diffusion) work in a compressed **latent space**:

1. Encode image: x → z (using pretrained VAE encoder)
2. Run diffusion in latent space z (~64×64 instead of 512×512)
3. Decode: z → x (using pretrained VAE decoder)

Speed: ~16× faster than pixel-space diffusion. Stable Diffusion is an LDM.

---

### Q14. What is FID score?

**A:** Fréchet Inception Distance measures the quality and diversity of generated images:

```
FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2√(Σ_rΣ_g))
```

Compares statistics of features extracted by InceptionV3 from real vs. generated images.
- Lower FID = better (closer to real data distribution)
- FID = 0: Generated images identical to real

---

### Q15. What is a Normalizing Flow?

**A:** Normalizing flows learn an invertible mapping f: X → Z where Z is a simple distribution (e.g., Gaussian):

```
z = f(x)   → can compute exact p(x) = p_z(f(x)) |det Jf(x)|
x = f⁻¹(z) → can generate new samples
```

**Requirement:** f must be invertible and have a tractable Jacobian determinant.

**Examples:** RealNVP, Glow, Neural Spline Flows.

**Advantage over GAN/VAE:** Exact likelihood computation.

---

### Q16. What is an Inception Score (IS)?

**A:** Inception Score measures GAN quality using InceptionV3 predictions:
```
IS = exp(E_x[KL(p(y|x) || p(y))])
```
- `p(y|x)`: Conditional label distribution (high confidence = good quality)
- `p(y)`: Marginal distribution (diverse = uniform)

Higher IS = better. Less reliable than FID. Doesn't measure how similar generated images are to real data.

---

### Q17. What is a conditional GAN (cGAN)?

**A:** A conditional GAN conditions generation on class labels or other information:

```
G(z, y) → fake sample for class y
D(x, y) → probability that x is real given class y
```

Applications: Generate specific digit (digit 5) in MNIST, text-to-image generation, image-to-image translation (pix2pix).

---

### Q18. What is CycleGAN?

**A:** CycleGAN performs unpaired image-to-image translation without paired examples. Uses two GANs and a cycle consistency loss:

```
G: X → Y  (e.g., horse → zebra)
F: Y → X  (zebra → horse)
Cycle: F(G(x)) ≈ x  and  G(F(y)) ≈ y
```

The cycle consistency loss enforces that translations are reversible — no need for paired (horse, zebra) image pairs.

---

### Q19. What is posterior collapse in VAE?

**A:** Posterior collapse occurs when the KL term in ELBO becomes 0 — the encoder ignores the input and always outputs the prior N(0,1):

```
q(z|x) = p(z) for all x → z contains no information about x
```

The decoder learns to generate without using latent z. Common in VAEs with powerful decoders (LSTM/Transformer decoders).

**Fix:** Anneal KL weight (start at 0, gradually increase), use δ-VAE, use a weaker decoder.

---

### Q20. What is score-based generative modeling?

**A:** Score-based models learn the score function (gradient of log density) of the data distribution:
```
s_θ(x) ≈ ∇_x log p(x)
```

Generation: Start from noise, follow the gradient of the learned score (Langevin dynamics):
```
x_{t+1} = x_t + α × s_θ(x_t) + √(2α) × noise
```

This is equivalent to diffusion models (Song et al., 2020) — DDPM noise prediction is a special case of score matching.

---

## Quick Code Example (VAE in PyTorch)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=1, hidden_dim=16):
        super().__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.enc_log_var = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # ε ~ N(0,1)
        return mu + eps * std        # z = μ + σε

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        return self.dec_out(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

def vae_loss(x_hat, x, mu, log_var):
    recon = F.mse_loss(x_hat, x, reduction='sum')
    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + kl

# Training
X = torch.tensor([[0.5,0.8],[0.3,0.7],[0.7,0.6],[0.4,0.9],
                   [0.6,0.7],[0.2,0.8],[0.8,0.5],[0.5,0.6]])
vae = VAE(input_dim=2, latent_dim=1)
opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(500):
    x_hat, mu, log_var = vae(X)
    loss = vae_loss(x_hat, X, mu, log_var)
    opt.zero_grad(); loss.backward(); opt.step()
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Generate new samples
with torch.no_grad():
    z = torch.randn(5, 1)  # sample from prior
    new_samples = vae.decode(z)
    print("Generated samples:\n", new_samples)
```
