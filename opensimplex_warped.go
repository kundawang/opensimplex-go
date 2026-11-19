package opensimplex

import (
	"fmt"
	"math"
)

// Domain-warped fractal (fBm) 2D texture sampling.
//
// A single layer of OpenSimplex noise looks uniform at large scales, and
// simply stacking octaves (plain fBm) tends to show regular, axis-aligned
// banding. Domain warping breaks those patterns up: the sample coordinates
// are first displaced by an independent noise field, and the fractal octaves
// are then evaluated at the displaced position.

const (
	// MaxWarpOctaves is the maximum number of fractal octaves a warped
	// sampler will evaluate. It bounds the per-sample computation cost.
	MaxWarpOctaves = 32

	// maxWarpFrequency bounds the effective frequency of the highest
	// octave: frequency * lacunarity^(octaves-1). Together with
	// maxWarpCoordinate it keeps every internal product well inside the
	// exact-integer range of float64 arithmetic.
	maxWarpFrequency = 1 << 20

	// maxWarpCoordinate bounds the absolute magnitude of sample
	// coordinates, before and after warping. Inputs beyond it are clamped,
	// so extreme (but finite) coordinates still produce a defined result.
	maxWarpCoordinate = 1 << 40
)

// Seeds for the two independent warp fields are derived from the configured
// seed by XOR-ing with these salts, so the warp field never correlates with
// the detail field built from the same seed.
const (
	warpSeedSaltX = 0x2545F4914F6CDD1D
	warpSeedSaltY = 0x6A09E667F3BCC909
)

// WarpConfig configures a domain-warped fractal sampler.
//
// The zero value is not valid; use DefaultWarpConfig and override the fields
// you care about, or fill every field explicitly.
type WarpConfig struct {
	// Seed determines the noise permutation. Two samplers built from equal
	// configs produce identical output at every point.
	Seed int64

	// Octaves is the number of fractal layers, in [1, MaxWarpOctaves].
	Octaves int

	// Frequency is the base sampling frequency. It must be finite and
	// greater than zero.
	Frequency float64

	// Lacunarity is the frequency multiplier between consecutive octaves.
	// It must be finite and at least 1. Additionally, the effective
	// frequency of the highest octave, Frequency*Lacunarity^(Octaves-1),
	// must not exceed 2^20.
	Lacunarity float64

	// Gain is the amplitude multiplier between consecutive octaves. It
	// must be finite and in (0, 1].
	Gain float64

	// WarpStrength is the magnitude of the coordinate displacement, in
	// noise-space units. It must be finite and non-negative. Zero disables
	// warping, reducing the sampler to plain fractal noise.
	WarpStrength float64
}

// DefaultWarpConfig returns a reasonable starting configuration for the
// given seed: 4 octaves, unit frequency, lacunarity 2, gain 0.5 and a
// moderate warp strength.
func DefaultWarpConfig(seed int64) WarpConfig {
	return WarpConfig{
		Seed:         seed,
		Octaves:      4,
		Frequency:    1,
		Lacunarity:   2,
		Gain:         0.5,
		WarpStrength: 0.5,
	}
}

// validate checks the config and returns a descriptive error for the first
// invalid field found.
func (cfg WarpConfig) validate() error {
	if cfg.Octaves < 1 {
		return fmt.Errorf("opensimplex: octaves must be at least 1, got %d", cfg.Octaves)
	}
	if cfg.Octaves > MaxWarpOctaves {
		return fmt.Errorf("opensimplex: octaves must be at most %d, got %d", MaxWarpOctaves, cfg.Octaves)
	}
	if math.IsNaN(cfg.Frequency) || math.IsInf(cfg.Frequency, 0) || cfg.Frequency <= 0 {
		return fmt.Errorf("opensimplex: frequency must be finite and positive, got %v", cfg.Frequency)
	}
	if math.IsNaN(cfg.Lacunarity) || math.IsInf(cfg.Lacunarity, 0) || cfg.Lacunarity < 1 {
		return fmt.Errorf("opensimplex: lacunarity must be finite and at least 1, got %v", cfg.Lacunarity)
	}
	if math.IsNaN(cfg.Gain) || math.IsInf(cfg.Gain, 0) || cfg.Gain <= 0 || cfg.Gain > 1 {
		return fmt.Errorf("opensimplex: gain must be finite and in (0, 1], got %v", cfg.Gain)
	}
	if math.IsNaN(cfg.WarpStrength) || math.IsInf(cfg.WarpStrength, 0) || cfg.WarpStrength < 0 {
		return fmt.Errorf("opensimplex: warp strength must be finite and non-negative, got %v", cfg.WarpStrength)
	}
	top := cfg.Frequency * math.Pow(cfg.Lacunarity, float64(cfg.Octaves-1))
	if math.IsInf(top, 0) || top > maxWarpFrequency {
		return fmt.Errorf("opensimplex: effective frequency of highest octave (%v) exceeds %d", top, maxWarpFrequency)
	}
	return nil
}

// WarpedNoise is a seeded, domain-warped fractal 2D sampler with float64
// inputs.
type WarpedNoise interface {
	// Eval2 samples the warped fractal texture. The result is normalized
	// by the total octave amplitude and lies in [-1, 1]. Non-finite
	// coordinates yield 0; finite coordinates of any magnitude yield a
	// finite, bounded result.
	Eval2(x, y float64) float64
}

// WarpedNoise32 is the float32 counterpart of WarpedNoise. It follows the
// same semantics: identical configs give identical results, non-finite
// coordinates yield 0, and results lie in [-1, 1].
type WarpedNoise32 interface {
	Eval2(x, y float32) float32
}

type warpedNoise struct {
	noise  Noise
	warpX  Noise // nil when warp strength is zero
	warpY  Noise // nil when warp strength is zero
	config WarpConfig
	// ampNorm is the sum of per-octave amplitudes, used to normalize the
	// output range.
	ampNorm float64
}

// NewWarped constructs a domain-warped fractal sampler from cfg. It returns
// an error if cfg is invalid; see WarpConfig for the constraints.
func NewWarped(cfg WarpConfig) (WarpedNoise, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	w := &warpedNoise{
		noise:  New(cfg.Seed),
		config: cfg,
	}

	amp := 1.0
	for i := 0; i < cfg.Octaves; i++ {
		w.ampNorm += amp
		amp *= cfg.Gain
	}

	if cfg.WarpStrength > 0 {
		w.warpX = New(cfg.Seed ^ warpSeedSaltX)
		w.warpY = New(cfg.Seed ^ warpSeedSaltY)
	}

	return w, nil
}

// NewWarped32 constructs the float32 counterpart of NewWarped. The config is
// validated exactly as in NewWarped.
func NewWarped32(cfg WarpConfig) (WarpedNoise32, error) {
	base, err := NewWarped(cfg)
	if err != nil {
		return nil, err
	}
	return &warpedNoise32{base: base}, nil
}

// clampCoordinate bounds v to [-limit, limit].
func clampCoordinate(v, limit float64) float64 {
	if v > limit {
		return limit
	}
	if v < -limit {
		return -limit
	}
	return v
}

func (w *warpedNoise) Eval2(x, y float64) float64 {
	// Non-finite input has a defined, neutral result rather than
	// propagating NaN through the texture.
	if math.IsNaN(x) || math.IsNaN(y) || math.IsInf(x, 0) || math.IsInf(y, 0) {
		return 0
	}

	x = clampCoordinate(x, maxWarpCoordinate)
	y = clampCoordinate(y, maxWarpCoordinate)

	if w.config.WarpStrength > 0 {
		// Displace the sample position with two independent noise
		// fields before layering the fractal detail.
		wx := x + w.config.WarpStrength*w.warpX.Eval2(x, y)
		wy := y + w.config.WarpStrength*w.warpY.Eval2(x, y)
		x = clampCoordinate(wx, maxWarpCoordinate)
		y = clampCoordinate(wy, maxWarpCoordinate)
	}

	sum := 0.0
	amp := 1.0
	freq := w.config.Frequency
	for i := 0; i < w.config.Octaves; i++ {
		sum += amp * w.noise.Eval2(x*freq, y*freq)
		amp *= w.config.Gain
		freq *= w.config.Lacunarity
	}
	return sum / w.ampNorm
}

type warpedNoise32 struct {
	base WarpedNoise
}

func (w *warpedNoise32) Eval2(x, y float32) float32 {
	return float32(w.base.Eval2(float64(x), float64(y)))
}
